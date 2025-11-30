"""
Tests for Ingestion Pipeline v2.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ingestion.v2.models import (
    BlockType,
    ConceptType,
    RelationType,
    PageBlock,
    StructuredPage,
    ExtractedConcept,
    LocalRelation,
    FormulaEntity,
    EnrichedChunk,
    DomainConcept,
    OntologyEdge,
)
from backend.ingestion.v2.extraction import (
    ExtractionConfig,
    HeuristicExtractor,
    ChunkBuilder,
)
from backend.ingestion.v2.graph_builder import (
    ConceptNode,
    BookGraphBuilder,
    BookGraphData,
)
from backend.ingestion.v2.ontology import (
    ConceptClusterer,
    NoiseDetector,
    OntologyBuilder,
)
from backend.ingestion.v2.pipeline_v2 import (
    IngestMode,
    IngestConfig,
    IngestPipelineV2,
)


# =============================================================================
# Model Tests
# =============================================================================

class TestModels:
    """Test data model classes."""
    
    def test_page_block_creation(self):
        block = PageBlock(
            block_type=BlockType.PARAGRAPH,
            text="Heat transfer is the exchange of thermal energy.",
        )
        assert block.block_type == BlockType.PARAGRAPH
        assert "thermal energy" in block.text
    
    def test_structured_page(self):
        page = StructuredPage(
            page_number=1,
            blocks=[
                PageBlock(BlockType.HEADING_1, "Chapter 1: Introduction"),
                PageBlock(BlockType.PARAGRAPH, "Heat transfer involves..."),
            ],
        )
        assert len(page.blocks) == 2
        full_text = page.full_text
        assert "Chapter 1" in full_text
        assert "Heat transfer" in full_text
    
    def test_extracted_concept(self):
        concept = ExtractedConcept(
            name="Heat Transfer",
            concept_type=ConceptType.PHENOMENON,
            confidence=0.9,
        )
        assert concept.canonical_name == "heat transfer"
        assert concept.concept_type == ConceptType.PHENOMENON
    
    def test_extracted_concept_positional(self):
        # Test that positional args work correctly
        concept = ExtractedConcept("Convection")
        assert concept.name == "Convection"
        assert concept.canonical_name == "convection"
    
    def test_enriched_chunk(self):
        chunk = EnrichedChunk(
            resource_id="test_book",
            page_number=5,
            full_text="Convection is heat transfer by fluid motion.",
            main_concepts=[
                ExtractedConcept(name="Convection", concept_type=ConceptType.PHENOMENON, is_main_concept=True),
            ],
            mentioned_concepts=[
                ExtractedConcept(name="Heat Transfer", concept_type=ConceptType.PHENOMENON),
            ],
        )
        assert len(chunk.all_concepts) == 2
        # ID is a hash, not prefixed with resource_id
        assert chunk.id is not None and len(chunk.id) > 0
    
    def test_domain_concept(self):
        dc = DomainConcept(
            id="physics.convection",
            display_name="Convection",
            canonical_name="convection",
            concept_type=ConceptType.PHENOMENON,
            domain="Physics",
            aliases=["convective heat transfer"],
        )
        assert dc.canonical_name == "convection"
        assert len(dc.aliases) == 1


# =============================================================================
# Extraction Tests
# =============================================================================

class TestHeuristicExtractor:
    """Test heuristic concept extraction."""
    
    def test_law_extraction(self):
        extractor = HeuristicExtractor()
        chunk = EnrichedChunk(
            resource_id="test",
            page_number=1,
            full_text="Fourier's Law describes heat conduction: q = -k∇T",
        )
        
        result = extractor.extract(chunk)
        
        concept_names = [c.name.lower() for c in result.main_concepts + result.mentioned_concepts]
        # Should find Fourier's Law
        assert any("fourier" in n for n in concept_names)
    
    def test_pedagogy_role_detection(self):
        extractor = HeuristicExtractor()
        
        # Definition
        chunk1 = EnrichedChunk(
            resource_id="test", page_number=1,
            full_text="Temperature is defined as a measure of thermal energy.",
        )
        result1 = extractor.extract(chunk1)
        assert result1.pedagogy_role == "definition"
        
        # Example
        chunk2 = EnrichedChunk(
            resource_id="test", page_number=1,
            full_text="For example, consider a hot cup of coffee cooling down.",
        )
        result2 = extractor.extract(chunk2)
        assert result2.pedagogy_role == "example"
    
    def test_difficulty_detection(self):
        extractor = HeuristicExtractor()
        
        # Advanced
        chunk = EnrichedChunk(
            resource_id="test", page_number=1,
            full_text="The partial derivative of temperature with respect to position gives the temperature gradient.",
        )
        result = extractor.extract(chunk)
        assert result.difficulty in ("intermediate", "advanced")
    
    def test_stop_word_filtering(self):
        extractor = HeuristicExtractor()
        
        # These should NOT become concepts
        assert not extractor._is_valid_concept("the")
        assert not extractor._is_valid_concept("example")
        assert not extractor._is_valid_concept("123")
        
        # These SHOULD be valid
        assert extractor._is_valid_concept("Temperature")
        assert extractor._is_valid_concept("Heat Transfer")


class TestChunkBuilder:
    """Test chunk building from structured pages."""
    
    def test_basic_chunking(self):
        pages = [
            StructuredPage(
                page_number=1,
                blocks=[
                    PageBlock(BlockType.HEADING_1, "Chapter 1: Heat Transfer"),
                    PageBlock(BlockType.PARAGRAPH, "Heat transfer is the movement of thermal energy. " * 20),
                    PageBlock(BlockType.PARAGRAPH, "There are three modes of heat transfer. " * 20),
                ],
            ),
        ]
        
        builder = ChunkBuilder(
            resource_id="test_book",
            max_chunk_tokens=100,
            min_chunk_tokens=10,
        )
        chunks = builder.build_chunks(pages)
        
        assert len(chunks) >= 1
        assert all(c.resource_id == "test_book" for c in chunks)
    
    def test_section_tracking(self):
        pages = [
            StructuredPage(
                page_number=1,
                blocks=[
                    PageBlock(BlockType.HEADING_1, "Chapter 1: Introduction"),
                    PageBlock(BlockType.PARAGRAPH, "Some introductory text here that is long enough."),
                    PageBlock(BlockType.HEADING_2, "1.1 Background"),
                    PageBlock(BlockType.PARAGRAPH, "Background information text that is also long enough."),
                ],
            ),
        ]
        
        builder = ChunkBuilder(resource_id="test", min_chunk_tokens=5)
        chunks = builder.build_chunks(pages)
        
        # First chunk should have chapter info
        if chunks:
            assert chunks[0].chapter_title == "Chapter 1: Introduction"


# =============================================================================
# Graph Builder Tests
# =============================================================================

class TestBookGraphBuilder:
    """Test per-book graph building."""
    
    def test_add_chunk(self):
        builder = BookGraphBuilder(resource_id="test_book")
        
        chunk = EnrichedChunk(
            resource_id="test_book",
            page_number=1,
            full_text="Convection is heat transfer by fluid motion.",
            pedagogy_role="definition",
            main_concepts=[
                ExtractedConcept(name="Convection", concept_type=ConceptType.PHENOMENON, is_main_concept=True, confidence=0.9),
            ],
        )
        
        builder.add_chunk(chunk)
        graph = builder.finalize()
        
        assert "convection" in graph.concepts
        assert graph.concepts["convection"].frequency == 1
    
    def test_cooccurrence_tracking(self):
        builder = BookGraphBuilder(resource_id="test", min_cooccurrence=1)
        
        # Add chunks where two concepts co-occur
        for i in range(3):
            chunk = EnrichedChunk(
                resource_id="test", page_number=i,
                full_text="Heat and temperature are related.",
                main_concepts=[
                    ExtractedConcept(name="Heat", concept_type=ConceptType.PHYSICAL_QUANTITY, is_main_concept=True),
                    ExtractedConcept(name="Temperature", concept_type=ConceptType.PHYSICAL_QUANTITY, is_main_concept=True),
                ],
            )
            builder.add_chunk(chunk)
        
        graph = builder.finalize()
        
        # Should have RELATED_TO edge
        assert len(graph.related_to) >= 1
    
    def test_teaches_relationship(self):
        builder = BookGraphBuilder(resource_id="test")
        
        chunk = EnrichedChunk(
            resource_id="test", page_number=1,
            full_text="Conduction is heat transfer through solid material.",
            pedagogy_role="definition",
            main_concepts=[
                ExtractedConcept(name="Conduction", concept_type=ConceptType.PHENOMENON, is_main_concept=True),
            ],
        )
        
        builder.add_chunk(chunk)
        graph = builder.finalize()
        
        # Definition chunk should TEACH the concept
        assert len(graph.teaches) >= 1


# =============================================================================
# Ontology Tests
# =============================================================================

class TestConceptClusterer:
    """Test concept clustering across books."""
    
    def test_exact_match_clustering(self):
        clusterer = ConceptClusterer()
        
        clusterer.add_concepts("book1", {
            "convection": {"display_name": "Convection", "concept_type": "phenomenon", "frequency": 5},
        })
        clusterer.add_concepts("book2", {
            "convection": {"display_name": "Convection", "concept_type": "phenomenon", "frequency": 3},
        })
        
        clusters = clusterer.resolve_clusters()
        
        # Should be in same cluster
        assert len(clusters) == 1
        assert clusters[0].total_frequency == 8
        assert clusters[0].book_count == 2
    
    def test_similar_name_clustering(self):
        clusterer = ConceptClusterer(similarity_threshold=0.8)
        
        clusterer.add_concepts("book1", {
            "heat_transfer": {"display_name": "Heat Transfer", "concept_type": "phenomenon", "frequency": 5},
        })
        clusterer.add_concepts("book2", {
            "heat transfer": {"display_name": "Heat Transfer", "concept_type": "phenomenon", "frequency": 3},
        })
        
        clusters = clusterer.resolve_clusters()
        
        # Should cluster together
        assert len(clusters) == 1


class TestNoiseDetector:
    """Test noise detection."""
    
    def test_generic_term_detection(self):
        detector = NoiseDetector()
        
        is_noise, reason = detector.is_noisy("example", frequency=5, confidence=0.8)
        assert is_noise
        # Could be generic_term or matches_noise_pattern
        assert reason in ("generic_term", "matches_noise_pattern")
    
    def test_short_name_detection(self):
        detector = NoiseDetector()
        
        is_noise, reason = detector.is_noisy("x", frequency=10, confidence=0.9)
        assert is_noise
        # Could be too_short or matches_noise_pattern for single char
        assert reason in ("too_short", "matches_noise_pattern")
    
    def test_low_confidence_detection(self):
        detector = NoiseDetector(min_frequency=3, min_confidence=0.5)
        
        is_noise, reason = detector.is_noisy("something", frequency=1, confidence=0.3)
        assert is_noise
        assert reason == "low_frequency_confidence"
    
    def test_valid_concept(self):
        detector = NoiseDetector()
        
        is_noise, _ = detector.is_noisy("convection", frequency=10, confidence=0.9)
        assert not is_noise


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestIngestConfig:
    """Test pipeline configuration."""
    
    def test_default_config(self):
        config = IngestConfig()
        assert config.mode == IngestMode.STANDARD
        assert config.domain == "STEM"
        assert config.store_postgres
        assert config.store_neo4j
    
    def test_fast_mode_config(self):
        config = IngestConfig(mode=IngestMode.FAST)
        assert config.mode == IngestMode.FAST


class TestIngestPipelineV2:
    """Test ingestion pipeline."""
    
    def test_resource_id_generation(self):
        pipeline = IngestPipelineV2()
        
        # Create temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            resource_id = pipeline._generate_resource_id(temp_path)
            assert resource_id
            assert "_" in resource_id  # Should have hash suffix
        finally:
            os.unlink(temp_path)
    
    @patch('backend.ingestion.v2.pipeline_v2.get_ocr_backend')
    def test_parse_phase_mock(self, mock_get_ocr):
        """Test parse phase with mocked OCR."""
        mock_backend = Mock()
        mock_backend.process_pdf.return_value = [
            StructuredPage(page_number=1, blocks=[
                PageBlock(BlockType.PARAGRAPH, "Test content for parsing."),
            ]),
        ]
        mock_get_ocr.return_value = mock_backend
        
        config = IngestConfig(store_postgres=False, store_neo4j=False, embed_chunks=False)
        pipeline = IngestPipelineV2(config)
        
        pages = pipeline._phase_parse("dummy.pdf")
        
        assert len(pages) == 1
        mock_backend.process_pdf.assert_called_once()


# =============================================================================
# Integration Tests (require databases)
# =============================================================================

@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Integration tests disabled (set RUN_INTEGRATION_TESTS=1)"
)
class TestIntegration:
    """Integration tests requiring actual databases."""
    
    def test_full_pipeline(self):
        """Test complete pipeline with sample PDF."""
        sample_pdf = "sample/hcv11th_part1.pdf"
        if not os.path.exists(sample_pdf):
            pytest.skip("Sample PDF not found")
        
        config = IngestConfig(
            mode=IngestMode.FAST,
            page_range=(1, 5),  # Only first 5 pages
        )
        pipeline = IngestPipelineV2(config)
        result = pipeline.ingest(sample_pdf)
        
        assert result.pages_processed == 5
        assert result.chunks_created > 0
        assert not result.errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
