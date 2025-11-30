"""
Ingestion Pipeline v2 - Main orchestrator for multi-book ingestion.

This module provides:
- Multi-phase pipeline: Parse -> Extract -> Embed -> Store
- Multiple ingestion modes (FAST, STANDARD, FULL)
- Support for pluggable OCR backends
- Ontology-aware concept extraction
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import EnrichedChunk, StructuredPage
from .ocr_backends import OCRBackend, get_ocr_backend
from .extraction import ChunkBuilder, ConceptExtractor, ExtractionConfig
from .graph_builder import BookGraphBuilder, Neo4jGraphWriter
from .ontology import NoiseDetector

logger = logging.getLogger("backend.ingestion.v2.pipeline")


class IngestMode(str, Enum):
    """Ingestion modes with different quality/speed tradeoffs."""
    FAST = "fast"          # Heuristic extraction, no LLM per chunk
    STANDARD = "standard"  # LLM extraction for key pages
    FULL = "full"          # LLM for every chunk


@dataclass
class IngestConfig:
    """Configuration for ingestion pipeline."""
    mode: IngestMode = IngestMode.STANDARD
    
    # OCR
    ocr_backend: str = "auto"  # auto, pymupdf, gemini, openai, hybrid
    
    # Chunking
    max_chunk_tokens: int = 300
    min_chunk_tokens: int = 50
    
    # Extraction
    domain: str = "STEM"
    use_llm_extraction: bool = True
    known_concepts: List[str] = field(default_factory=list)
    
    # Graph
    min_cooccurrence: int = 3
    noise_frequency_threshold: int = 1
    build_prerequisites_llm: bool = True
    
    # Storage
    store_postgres: bool = True
    store_neo4j: bool = True
    embed_chunks: bool = True
    
    # Performance
    batch_size: int = 50
    page_range: Optional[Tuple[int, int]] = None  # (start, end) or None for all


@dataclass
class IngestResult:
    """Result of ingestion pipeline."""
    resource_id: str
    book_id: str
    pages_processed: int
    chunks_created: int
    concepts_extracted: int
    relationships_created: int
    duration_seconds: float
    errors: List[str] = field(default_factory=list)


class IngestPipelineV2:
    """
    Multi-phase ingestion pipeline for textbooks.
    
    Phases:
    1. Parse: OCR/text extraction -> StructuredPages
    2. Chunk: StructuredPages -> EnrichedChunks
    3. Extract: Concept extraction (heuristic + LLM)
    4. Embed: Generate embeddings for chunks
    5. Store: Write to Postgres + Neo4j
    """
    
    def __init__(self, config: IngestConfig = None):
        self.config = config or IngestConfig()
        
        # Initialize components
        self.ocr_backend: Optional[OCRBackend] = None
        self.extractor: Optional[ConceptExtractor] = None
        self.graph_builder: Optional[BookGraphBuilder] = None
        self.neo4j_writer: Optional[Neo4jGraphWriter] = None
        self.noise_detector = NoiseDetector()
    
    def ingest(
        self,
        file_path: str,
        resource_id: Optional[str] = None,
        book_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> IngestResult:
        """Run full ingestion pipeline on a file."""
        start_time = time.time()
        errors = []
        
        # Generate IDs
        if not resource_id:
            resource_id = self._generate_resource_id(file_path)
        if not book_id:
            book_id = resource_id
        
        logger.info(f"Starting ingestion: {file_path} (mode={self.config.mode.value})")
        
        # Phase 1: Parse
        try:
            pages = self._phase_parse(file_path)
            logger.info(f"Parsed {len(pages)} pages")
        except Exception as e:
            logger.error(f"Parse phase failed: {e}")
            errors.append(f"Parse: {e}")
            return IngestResult(
                resource_id=resource_id, book_id=book_id,
                pages_processed=0, chunks_created=0,
                concepts_extracted=0, relationships_created=0,
                duration_seconds=time.time() - start_time,
                errors=errors,
            )
        
        # Phase 2: Chunk
        try:
            chunks = self._phase_chunk(pages, resource_id, book_id)
            logger.info(f"Created {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Chunk phase failed: {e}")
            errors.append(f"Chunk: {e}")
            chunks = []
        
        # Phase 3: Extract
        try:
            chunks = self._phase_extract(chunks)
            logger.info(f"Extraction complete")
        except Exception as e:
            logger.error(f"Extract phase failed: {e}")
            errors.append(f"Extract: {e}")
        
        # Phase 4: Embed
        if self.config.embed_chunks and chunks:
            try:
                chunks = self._phase_embed(chunks)
                logger.info(f"Embedded {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Embed phase failed: {e}")
                errors.append(f"Embed: {e}")
        
        # Phase 5: Store
        concept_count = 0
        rel_count = 0
        
        if chunks:
            try:
                concept_count, rel_count = self._phase_store(
                    chunks, resource_id, book_id, title or Path(file_path).stem
                )
                logger.info(f"Stored {concept_count} concepts, {rel_count} relationships")
            except Exception as e:
                logger.error(f"Store phase failed: {e}")
                errors.append(f"Store: {e}")
        
        duration = time.time() - start_time
        logger.info(f"Ingestion complete in {duration:.1f}s")
        
        return IngestResult(
            resource_id=resource_id,
            book_id=book_id,
            pages_processed=len(pages),
            chunks_created=len(chunks),
            concepts_extracted=concept_count,
            relationships_created=rel_count,
            duration_seconds=duration,
            errors=errors,
        )
    
    # =========================================================================
    # Phase 1: Parse
    # =========================================================================
    
    def _phase_parse(self, file_path: str) -> List[StructuredPage]:
        """Parse PDF into structured pages."""
        if self.ocr_backend is None:
            self.ocr_backend = get_ocr_backend(self.config.ocr_backend)
        
        pages = self.ocr_backend.process_pdf(file_path)
        
        # Apply page range filter
        if self.config.page_range:
            start, end = self.config.page_range
            pages = [p for p in pages if start <= p.page_number <= end]
        
        return pages
    
    # =========================================================================
    # Phase 2: Chunk
    # =========================================================================
    
    def _phase_chunk(
        self,
        pages: List[StructuredPage],
        resource_id: str,
        book_id: str,
    ) -> List[EnrichedChunk]:
        """Convert structured pages to enriched chunks."""
        builder = ChunkBuilder(
            resource_id=resource_id,
            book_id=book_id,
            max_chunk_tokens=self.config.max_chunk_tokens,
            min_chunk_tokens=self.config.min_chunk_tokens,
        )
        return builder.build_chunks(pages)
    
    # =========================================================================
    # Phase 3: Extract
    # =========================================================================
    
    def _phase_extract(self, chunks: List[EnrichedChunk]) -> List[EnrichedChunk]:
        """Extract concepts from chunks."""
        # Determine if we use LLM based on mode
        use_llm = self.config.mode == IngestMode.FULL or (
            self.config.mode == IngestMode.STANDARD and self.config.use_llm_extraction
        )
        
        extraction_config = ExtractionConfig(
            use_llm=use_llm,
            known_concepts=self.config.known_concepts,
            domain=self.config.domain,
        )
        
        if self.extractor is None or self.extractor.config != extraction_config:
            self.extractor = ConceptExtractor(extraction_config)
        
        # Process in batches
        processed = []
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            
            if self.config.mode == IngestMode.FAST:
                # Heuristic only
                for chunk in batch:
                    self.extractor.heuristic.extract(chunk)
                    processed.append(chunk)
            else:
                # Full extraction with context
                batch_processed = self.extractor.extract_batch(batch)
                processed.extend(batch_processed)
            
            logger.debug(f"Extracted batch {i//self.config.batch_size + 1}")
        
        return processed
    
    # =========================================================================
    # Phase 4: Embed
    # =========================================================================
    
    def _phase_embed(self, chunks: List[EnrichedChunk]) -> List[EnrichedChunk]:
        """Generate embeddings for chunks."""
        try:
            from backend.ingestion.embed import embed_text
        except ImportError:
            logger.warning("Embedding module not available")
            return chunks
        
        for chunk in chunks:
            if not chunk.embedding:
                try:
                    chunk.embedding = embed_text(chunk.full_text[:2000])
                except Exception as e:
                    logger.warning(f"Embedding failed for chunk {chunk.id}: {e}")
        
        return chunks
    
    # =========================================================================
    # Phase 5: Store
    # =========================================================================
    
    def _phase_store(
        self,
        chunks: List[EnrichedChunk],
        resource_id: str,
        book_id: str,
        title: str,
    ) -> Tuple[int, int]:
        """Store chunks and graph to databases."""
        # Build graph
        self.graph_builder = BookGraphBuilder(
            resource_id=resource_id,
            book_id=book_id,
            min_cooccurrence=self.config.min_cooccurrence,
            noise_frequency_threshold=self.config.noise_frequency_threshold,
        )
        
        for chunk in chunks:
            self.graph_builder.add_chunk(chunk)
        
        graph_data = self.graph_builder.finalize()
        
        # LLM prerequisites
        if self.config.build_prerequisites_llm and self.config.mode != IngestMode.FAST:
            self.graph_builder.build_prerequisites_llm(
                use_lite_model=(self.config.mode == IngestMode.STANDARD)
            )
        
        # Count relationships
        rel_count = (
            len(graph_data.occurs_in) +
            len(graph_data.teaches) +
            len(graph_data.prerequisites) +
            len(graph_data.related_to)
        )
        
        # Store to Postgres
        if self.config.store_postgres:
            self._store_postgres(chunks, resource_id, title)
        
        # Store to Neo4j
        if self.config.store_neo4j:
            if self.neo4j_writer is None:
                self.neo4j_writer = Neo4jGraphWriter()
            self.neo4j_writer.write_book_graph(graph_data)
        
        return len(graph_data.concepts), rel_count
    
    def _store_postgres(
        self,
        chunks: List[EnrichedChunk],
        resource_id: str,
        title: str,
    ):
        """Store chunks to Postgres."""
        try:
            from backend.core.db import get_db_dsn
            import psycopg2
        except ImportError:
            logger.warning("Postgres module not available")
            return
        
        dsn = get_db_dsn()
        
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                # Upsert resource
                cur.execute("""
                    INSERT INTO resources (id, title, source_type, created_at)
                    VALUES (%s, %s, 'pdf', NOW())
                    ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title
                """, (resource_id, title))
                
                # Batch insert chunks
                for chunk in chunks:
                    cur.execute("""
                        INSERT INTO chunks (
                            id, resource_id, content, page_number,
                            embedding, metadata, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata
                    """, (
                        chunk.id,
                        resource_id,
                        chunk.full_text,
                        chunk.page_number,
                        chunk.embedding,
                        chunk.to_metadata_json(),
                    ))
                
                conn.commit()
        
        logger.info(f"Stored {len(chunks)} chunks to Postgres")
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _generate_resource_id(self, file_path: str) -> str:
        """Generate deterministic resource ID from file."""
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read(8192)).hexdigest()[:8]
        stem = Path(file_path).stem[:20].lower().replace(" ", "_")
        return f"{stem}_{file_hash}"


# =============================================================================
# Multi-Book Ingestion
# =============================================================================

class MultiBookIngestor:
    """Ingest multiple books and build unified ontology."""
    
    def __init__(self, config: IngestConfig = None):
        self.config = config or IngestConfig()
        self.pipeline = IngestPipelineV2(config)
        self.results: List[IngestResult] = []
    
    def ingest_books(
        self,
        file_paths: List[str],
        domain: str = "STEM",
        build_ontology: bool = True,
    ) -> List[IngestResult]:
        """Ingest multiple books."""
        self.config.domain = domain
        
        for path in file_paths:
            logger.info(f"Ingesting: {path}")
            result = self.pipeline.ingest(path)
            self.results.append(result)
        
        if build_ontology and len(self.results) > 1:
            self._build_cross_book_ontology()
        
        return self.results
    
    def _build_cross_book_ontology(self):
        """Build ontology from all ingested books."""
        from .ontology import OntologyBuilder, Neo4jOntologyWriter
        
        builder = OntologyBuilder(domain=self.config.domain)
        
        # Collect concepts from Neo4j
        try:
            from neo4j import GraphDatabase
            
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                for result in self.results:
                    concepts = session.run("""
                        MATCH (c:Concept {book_id: $book_id})
                        RETURN c.canonical_name AS name, c.display_name AS display,
                               c.concept_type AS type, c.frequency AS freq,
                               c.definition AS def
                    """, book_id=result.book_id)
                    
                    book_concepts = {
                        r["name"]: {
                            "display_name": r["display"],
                            "concept_type": r["type"],
                            "frequency": r["freq"],
                            "definition": r["def"] or "",
                        }
                        for r in concepts
                    }
                    builder.add_book_concepts(result.book_id, book_concepts)
            
            driver.close()
        except Exception as e:
            logger.error(f"Failed to collect concepts: {e}")
            return
        
        # Build and write ontology
        domain_concepts, edges = builder.build()
        
        if domain_concepts:
            writer = Neo4jOntologyWriter()
            writer.write_ontology(domain_concepts, edges)
            logger.info(f"Built ontology: {len(domain_concepts)} concepts, {len(edges)} edges")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest textbooks into knowledge graph")
    parser.add_argument("files", nargs="+", help="PDF files to ingest")
    parser.add_argument("--mode", choices=["fast", "standard", "full"], default="standard")
    parser.add_argument("--domain", default="STEM")
    parser.add_argument("--ocr", default="auto", help="OCR backend")
    parser.add_argument("--no-neo4j", action="store_true")
    parser.add_argument("--no-postgres", action="store_true")
    parser.add_argument("--page-range", type=str, help="Page range (e.g., 1-50)")
    
    args = parser.parse_args()
    
    config = IngestConfig(
        mode=IngestMode(args.mode),
        domain=args.domain,
        ocr_backend=args.ocr,
        store_neo4j=not args.no_neo4j,
        store_postgres=not args.no_postgres,
    )
    
    if args.page_range:
        start, end = map(int, args.page_range.split("-"))
        config.page_range = (start, end)
    
    if len(args.files) == 1:
        pipeline = IngestPipelineV2(config)
        result = pipeline.ingest(args.files[0])
        print(f"Ingested: {result.chunks_created} chunks, {result.concepts_extracted} concepts")
    else:
        ingestor = MultiBookIngestor(config)
        results = ingestor.ingest_books(args.files, domain=args.domain)
        for r in results:
            print(f"{r.book_id}: {r.chunks_created} chunks, {r.concepts_extracted} concepts")


if __name__ == "__main__":
    main()
