"""
Ingestion Pipeline v2 - Multi-book, Ontology-aware Knowledge Graph Builder

This package provides:
- Structured document parsing with OCR backends (API-based VLMs, local OCR)
- Ontology-aware concept extraction
- Multi-book knowledge graph construction
- Cross-book ontology consolidation
- Noise control and curation utilities

Architecture layers:
1. Document Understanding (OCR + layout) -> StructuredPage
2. Semantic Chunking -> EnrichedChunk  
3. Ontology-aware Extraction -> concepts, relations, formulas
4. Per-book Pedagogical Graph -> Neo4j Concept/Chunk nodes
5. Cross-book Ontology -> DomainConcept nodes + MAPS_TO edges
"""

from .models import (
    StructuredPage,
    PageBlock,
    BlockType,
    EnrichedChunk,
    ExtractedConcept,
    ConceptType,
    LocalRelation,
    RelationType,
    FormulaEntity,
    DomainConcept,
    OntologyEdge,
)

from .ocr_backends import (
    OCRBackend,
    get_ocr_backend,
    PyMuPDFBackend,
    GeminiVisionBackend,
    OpenAIVisionBackend,
    HybridOCRBackend,
)

from .extraction import (
    ExtractionConfig,
    HeuristicExtractor,
    LLMExtractor,
    ConceptExtractor,
    ChunkBuilder,
)

from .graph_builder import (
    ConceptNode,
    RelationshipData,
    BookGraphData,
    BookGraphBuilder,
    Neo4jGraphWriter,
)

from .ontology import (
    ConceptOccurrence,
    ConceptCluster,
    ConceptClusterer,
    OntologyBuilder,
    NoiseDetector,
    Neo4jOntologyWriter,
)

from .pipeline_v2 import (
    IngestMode,
    IngestConfig,
    IngestResult,
    IngestPipelineV2,
    MultiBookIngestor,
)

__all__ = [
    # Models
    "StructuredPage",
    "PageBlock", 
    "BlockType",
    "EnrichedChunk",
    "ExtractedConcept",
    "ConceptType",
    "LocalRelation",
    "RelationType",
    "FormulaEntity",
    "DomainConcept",
    "OntologyEdge",
    # OCR
    "OCRBackend",
    "get_ocr_backend",
    "PyMuPDFBackend",
    "GeminiVisionBackend",
    "OpenAIVisionBackend",
    "HybridOCRBackend",
    # Extraction
    "ExtractionConfig",
    "HeuristicExtractor",
    "LLMExtractor",
    "ConceptExtractor",
    "ChunkBuilder",
    # Graph Builder
    "ConceptNode",
    "RelationshipData",
    "BookGraphData",
    "BookGraphBuilder",
    "Neo4jGraphWriter",
    # Ontology
    "ConceptOccurrence",
    "ConceptCluster",
    "ConceptClusterer",
    "OntologyBuilder",
    "NoiseDetector",
    "Neo4jOntologyWriter",
    # Pipeline
    "IngestMode",
    "IngestConfig",
    "IngestResult",
    "IngestPipelineV2",
    "MultiBookIngestor",
]
