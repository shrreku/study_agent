"""Tests for the unified ingestion pipeline."""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


class TestChunkData:
    """Tests for ChunkData dataclass."""
    
    def test_chunk_data_creation(self):
        from ingestion.pipeline import ChunkData
        
        chunk = ChunkData(
            resource_id="test-resource",
            page_number=1,
            full_text="This is a test chunk about Newton's Laws.",
            concepts=["Newton's Laws", "Force"],
            pedagogy_role="explanation"
        )
        
        assert chunk.resource_id == "test-resource"
        assert chunk.page_number == 1
        assert len(chunk.concepts) == 2
        assert chunk.pedagogy_role == "explanation"
    
    def test_to_tags_json(self):
        from ingestion.pipeline import ChunkData
        
        chunk = ChunkData(
            resource_id="test",
            pedagogy_role="definition",
            content_type="concept_intro",
            difficulty="intermediate",
            domain="Physics",
            topic="Mechanics",
            concepts=["Force", "Mass", "Acceleration"],
            prerequisites=["Basic Math"]
        )
        
        tags = chunk.to_tags_json()
        
        assert tags["pedagogy_role"] == "definition"
        assert tags["difficulty"] == "intermediate"
        assert tags["domain"] == "Physics"
        assert "Force" in tags["key_concepts"]
        assert "Basic Math" in tags["prerequisites"]


class TestIngestMode:
    """Tests for ingestion modes."""
    
    def test_mode_enum(self):
        from ingestion.pipeline import IngestMode
        
        assert IngestMode.FAST.value == "fast"
        assert IngestMode.STANDARD.value == "standard"
        assert IngestMode.FULL.value == "full"
    
    def test_mode_from_string(self):
        from ingestion.pipeline import IngestMode
        
        assert IngestMode("fast") == IngestMode.FAST
        assert IngestMode("standard") == IngestMode.STANDARD
        assert IngestMode("full") == IngestMode.FULL


class TestIngestPipeline:
    """Tests for the main pipeline."""
    
    @pytest.fixture
    def sample_pages(self) -> List[str]:
        """Sample document pages for testing."""
        return [
            """
            Chapter 1: Introduction to Mechanics
            
            1.1 Newton's Laws of Motion
            
            Newton's First Law states that an object at rest stays at rest and an object
            in motion stays in motion with the same speed and in the same direction 
            unless acted upon by an unbalanced force. This is also known as the Law of Inertia.
            
            Newton's Second Law is defined as: The acceleration of an object is directly
            proportional to the net force acting on it and inversely proportional to its mass.
            Mathematically, this is expressed as F = ma, where F is force, m is mass, and a is acceleration.
            """,
            """
            1.2 Applications of Newton's Laws
            
            Example 1.1: A 5 kg block is pushed with a force of 20 N. Calculate the acceleration.
            
            Solution: Using F = ma, we get a = F/m = 20/5 = 4 m/s².
            
            For example, consider a car accelerating from rest. The engine provides a force that 
            overcomes friction and air resistance to accelerate the vehicle.
            """
        ]
    
    def test_parse_fast_mode(self, sample_pages):
        """Test fast parsing without LLM."""
        from ingestion.pipeline import IngestPipeline, IngestMode
        
        with patch('ingestion.pipeline.IngestPipeline._get_mode_from_env', return_value=IngestMode.FAST):
            pipeline = IngestPipeline(
                resource_id="test-resource",
                file_path="/fake/path.pdf",
                mode=IngestMode.FAST
            )
            
            # Mock the parse_utils
            with patch('ingestion.parse_utils.extract_text_by_type', return_value=sample_pages):
                pipeline._phase_parse()
            
            # Should have created chunks
            assert len(pipeline.chunks) > 0
            
            # Each chunk should have required fields
            for chunk in pipeline.chunks:
                assert chunk.resource_id == "test-resource"
                assert chunk.full_text
                assert chunk.page_number >= 1
    
    def test_extract_heuristic(self, sample_pages):
        """Test heuristic extraction without LLM."""
        from ingestion.pipeline import IngestPipeline, IngestMode, ChunkData
        
        pipeline = IngestPipeline(
            resource_id="test",
            file_path="/fake/path.pdf",
            mode=IngestMode.FAST
        )
        
        # Create test chunks
        pipeline.chunks = [
            ChunkData(
                resource_id="test",
                full_text="Newton's First Law is defined as the Law of Inertia. It states that objects at rest stay at rest.",
                page_number=1
            ),
            ChunkData(
                resource_id="test",
                full_text="For example, consider a ball rolling on a frictionless surface. The ball will continue rolling forever.",
                page_number=1
            ),
            ChunkData(
                resource_id="test",
                full_text="Problem 1: Calculate the force needed to accelerate a 10 kg mass at 5 m/s².",
                page_number=2
            )
        ]
        
        pipeline._extract_heuristic()
        
        # Check pedagogy role detection
        assert pipeline.chunks[0].pedagogy_role == "definition"
        assert pipeline.chunks[1].pedagogy_role == "example"
        assert pipeline.chunks[2].pedagogy_role == "problem"
    
    def test_canonicalize(self):
        """Test concept name canonicalization."""
        from ingestion.pipeline import IngestPipeline
        
        pipeline = IngestPipeline(resource_id="test", file_path="/fake.pdf")
        
        # Test basic canonicalization
        assert pipeline._canonicalize("Newton's Laws") == "newtons laws"
        assert pipeline._canonicalize("Force") == "force"
        assert pipeline._canonicalize("F=ma") == "f ma"
        assert pipeline._canonicalize("") == ""
        assert pipeline._canonicalize("  Heat Transfer  ") == "heat transfer"
    
    def test_build_section_path(self):
        """Test section path building."""
        from ingestion.pipeline import IngestPipeline
        
        pipeline = IngestPipeline(resource_id="test", file_path="/fake.pdf")
        
        # Test with number
        assert pipeline._build_section_path("1.2.3", "") == ["1", "2", "3"]
        
        # Test with title only
        assert pipeline._build_section_path("", "Introduction") == ["Introduction"]
        
        # Test with both (number takes precedence)
        assert pipeline._build_section_path("2.1", "Methods") == ["2", "1"]
        
        # Test empty
        assert pipeline._build_section_path("", "") == []


class TestDatabasePool:
    """Tests for database connection pooling."""
    
    def test_pg_dsn_construction(self):
        """Test Postgres DSN construction."""
        from ingestion.pipeline import DatabasePool
        
        # Set test environment
        with patch.dict(os.environ, {
            "POSTGRES_USER": "testuser",
            "POSTGRES_PASSWORD": "testpass",
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "testdb"
        }, clear=False):
            dsn = DatabasePool._get_pg_dsn()
            assert "testuser" in dsn
            assert "testpass" in dsn
            assert "testdb" in dsn


class TestIngestResult:
    """Tests for ingestion result."""
    
    def test_result_success(self):
        from ingestion.pipeline import IngestResult
        
        result = IngestResult(
            resource_id="test",
            chunks_created=10,
            concepts_created=5
        )
        
        assert result.success is True
        assert result.chunks_created == 10
    
    def test_result_with_errors(self):
        from ingestion.pipeline import IngestResult
        
        result = IngestResult(resource_id="test")
        result.errors.append("Test error")
        
        assert result.success is False


class TestIntegration:
    """Integration tests (require mocks for external services)."""
    
    def test_full_pipeline_fast_mode(self):
        """Test full pipeline in fast mode with mocks."""
        from ingestion.pipeline import IngestPipeline, IngestMode, ingest_resource
        
        sample_text = """
        Introduction to Heat Transfer
        
        Heat transfer is the process of thermal energy moving from one object to another.
        There are three main modes: conduction, convection, and radiation.
        
        Conduction is defined as heat transfer through direct contact between molecules.
        
        For example, touching a hot stove transfers heat to your hand through conduction.
        """
        
        with patch('ingestion.parse_utils.extract_text_by_type', return_value=[sample_text]):
            with patch('ingestion.embed.embed_texts', return_value=[[0.1] * 384] * 10):
                with patch.object(IngestPipeline, '_store_postgres'):
                    with patch.object(IngestPipeline, '_store_neo4j'):
                        result = ingest_resource(
                            resource_id="test-resource-123",
                            file_path="/fake/path.pdf",
                            mode="fast"
                        )
        
        assert result.resource_id == "test-resource-123"
        assert result.elapsed_ms >= 0


class TestHeuristicExtraction:
    """Tests for heuristic-based metadata extraction."""
    
    def test_difficulty_detection(self):
        """Test difficulty level detection."""
        from ingestion.pipeline import IngestPipeline, ChunkData
        
        pipeline = IngestPipeline(resource_id="test", file_path="/fake.pdf")
        
        # Advanced content
        chunk1 = ChunkData(
            resource_id="test",
            full_text="The Laplacian operator is used in partial differential equations.",
            page_number=1
        )
        pipeline.chunks = [chunk1]
        pipeline._extract_heuristic()
        assert chunk1.difficulty == "advanced"
        
        # Intermediate content
        chunk2 = ChunkData(
            resource_id="test",
            full_text="The derivative of x² is 2x using calculus rules.",
            page_number=1
        )
        pipeline.chunks = [chunk2]
        pipeline._extract_heuristic()
        assert chunk2.difficulty == "intermediate"
        
        # Basic content
        chunk3 = ChunkData(
            resource_id="test",
            full_text="Speed is the distance traveled divided by time.",
            page_number=1
        )
        pipeline.chunks = [chunk3]
        pipeline._extract_heuristic()
        assert chunk3.difficulty == "introductory"
    
    def test_concept_extraction(self):
        """Test concept extraction from text."""
        from ingestion.pipeline import IngestPipeline, ChunkData
        
        pipeline = IngestPipeline(resource_id="test", file_path="/fake.pdf")
        
        chunk = ChunkData(
            resource_id="test",
            full_text="Newton's Laws describe the relationship between Force and Acceleration. Isaac Newton developed Classical Mechanics.",
            page_number=1
        )
        
        pipeline.chunks = [chunk]
        pipeline._extract_heuristic()
        
        # Should extract multi-word capitalized phrases
        assert any("Newton" in c for c in chunk.concepts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
