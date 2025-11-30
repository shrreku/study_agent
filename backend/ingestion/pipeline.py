"""Unified Ingestion Pipeline - Robust and Efficient.

This module provides a streamlined ingestion pipeline that:
1. Minimizes LLM calls (single unified extraction per chunk)
2. Batches database operations (embedding, Postgres writes, Neo4j writes)
3. Uses connection pooling for both Postgres and Neo4j
4. Provides clear phase separation: Parse → Extract → Store
5. Is idempotent and resumable

Usage:
    from backend.ingestion.pipeline import IngestPipeline
    
    pipeline = IngestPipeline(resource_id="...", file_path="/path/to/file.pdf")
    result = pipeline.run()
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("backend.ingestion.pipeline")


class IngestMode(Enum):
    """Ingestion mode controlling feature set."""
    FAST = "fast"           # Minimal processing, no LLM
    STANDARD = "standard"   # Balanced: LLM chunking + basic tagging
    FULL = "full"           # All features: hierarchical tags, formulas, KG


@dataclass
class ChunkData:
    """Intermediate representation of a chunk during processing."""
    # Core identifiers
    id: Optional[str] = None
    resource_id: str = ""
    
    # Position
    page_number: int = 0
    page_start: int = 0
    page_end: int = 0
    source_offset: int = 0
    
    # Content
    full_text: str = ""
    text_snippet: str = ""
    
    # Structure
    section_title: str = ""
    section_number: str = ""
    section_level: Optional[int] = None
    section_path: List[str] = field(default_factory=list)
    
    # Extracted metadata (filled by extraction phase)
    concepts: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    formulas: List[Dict[str, Any]] = field(default_factory=list)
    
    # Classification
    chunk_type: str = ""
    pedagogy_role: str = "explanation"
    content_type: str = "unknown"
    difficulty: str = "intermediate"
    cognitive_level: str = "understand"
    
    # Hierarchical taxonomy
    domain: str = ""
    topic: str = ""
    subtopic: str = ""
    
    # Flags
    has_figure: bool = False
    has_equation: bool = False
    figure_labels: List[str] = field(default_factory=list)
    equation_labels: List[str] = field(default_factory=list)
    
    # Computed
    token_count: int = 0
    embedding: Optional[List[float]] = None
    
    def to_tags_json(self) -> Dict[str, Any]:
        """Convert metadata to JSONB tags format for Postgres."""
        return {
            "pedagogy_role": self.pedagogy_role,
            "content_type": self.content_type,
            "difficulty": self.difficulty,
            "cognitive_level": self.cognitive_level,
            "domain": self.domain,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "key_concepts": self.concepts[:5],
            "prerequisites": self.prerequisites[:5],
        }


@dataclass
class IngestResult:
    """Result of ingestion pipeline run."""
    resource_id: str
    chunks_created: int = 0
    chunks_updated: int = 0
    chunks_deleted: int = 0
    concepts_created: int = 0
    relationships_created: int = 0
    elapsed_ms: int = 0
    errors: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class DatabasePool:
    """Connection pool manager for Postgres and Neo4j."""
    
    _pg_pool = None
    _neo4j_driver = None
    
    @classmethod
    def get_postgres_conn(cls):
        """Get Postgres connection from pool."""
        try:
            import psycopg2
            from psycopg2 import pool
        except ImportError:
            logger.warning("psycopg2 not available")
            return None
        
        if cls._pg_pool is None:
            dsn = cls._get_pg_dsn()
            try:
                cls._pg_pool = pool.ThreadedConnectionPool(1, 10, dsn)
            except Exception:
                logger.exception("Failed to create Postgres pool")
                return None
        
        try:
            return cls._pg_pool.getconn()
        except Exception:
            logger.exception("Failed to get Postgres connection")
            return None
    
    @classmethod
    def return_postgres_conn(cls, conn):
        """Return connection to pool."""
        if cls._pg_pool and conn:
            try:
                cls._pg_pool.putconn(conn)
            except Exception:
                pass
    
    @classmethod
    @contextmanager
    def postgres(cls):
        """Context manager for Postgres connection."""
        conn = cls.get_postgres_conn()
        try:
            yield conn
        finally:
            cls.return_postgres_conn(conn)
    
    @classmethod
    def get_neo4j_driver(cls):
        """Get shared Neo4j driver."""
        if cls._neo4j_driver is not None:
            return cls._neo4j_driver
        
        try:
            from neo4j import GraphDatabase
        except ImportError:
            logger.warning("neo4j driver not available")
            return None
        
        uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
        
        # Fallback to localhost if neo4j hostname not resolvable
        if "neo4j:7687" in uri:
            import socket
            try:
                socket.gethostbyname("neo4j")
            except socket.error:
                uri = uri.replace("neo4j:7687", "localhost:7687")
        
        try:
            cls._neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            return cls._neo4j_driver
        except Exception:
            logger.exception("Failed to create Neo4j driver")
            return None
    
    @classmethod
    @contextmanager
    def neo4j_session(cls):
        """Context manager for Neo4j session."""
        driver = cls.get_neo4j_driver()
        if driver is None:
            yield None
            return
        
        session = None
        try:
            session = driver.session()
            yield session
        except Exception:
            logger.exception("Neo4j session error")
            yield None
        finally:
            if session:
                try:
                    session.close()
                except Exception:
                    pass
    
    @classmethod
    def _get_pg_dsn(cls) -> str:
        """Build Postgres DSN."""
        dsn = os.getenv("DATABASE_URL")
        if dsn:
            return dsn
        
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        host = os.getenv("POSTGRES_HOST", "postgres")
        port = os.getenv("POSTGRES_PORT", "5432")
        
        if host == "postgres":
            import socket
            try:
                socket.gethostbyname("postgres")
            except socket.error:
                host = "localhost"
                if port == "5432":
                    port = "5433"
        
        db = os.getenv("POSTGRES_DB", "app")
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    
    @classmethod
    def shutdown(cls):
        """Clean up all connections."""
        if cls._pg_pool:
            try:
                cls._pg_pool.closeall()
            except Exception:
                pass
            cls._pg_pool = None
        
        if cls._neo4j_driver:
            try:
                cls._neo4j_driver.close()
            except Exception:
                pass
            cls._neo4j_driver = None


class IngestPipeline:
    """Unified ingestion pipeline for educational documents."""
    
    def __init__(
        self,
        resource_id: str,
        file_path: str,
        mode: Optional[IngestMode] = None,
        batch_size: int = 50,
    ):
        self.resource_id = resource_id
        self.file_path = file_path
        self.mode = mode or self._get_mode_from_env()
        self.batch_size = batch_size
        
        self.chunks: List[ChunkData] = []
        self.result = IngestResult(resource_id=resource_id)
    
    def _get_mode_from_env(self) -> IngestMode:
        """Determine ingestion mode from environment."""
        mode_str = os.getenv("INGEST_MODE", "standard").lower()
        try:
            return IngestMode(mode_str)
        except ValueError:
            return IngestMode.STANDARD
    
    def run(self) -> IngestResult:
        """Execute the full ingestion pipeline."""
        t0 = time.time()
        
        try:
            # Phase 1: Parse document into raw chunks
            logger.info("Phase 1: Parsing document", extra={"resource_id": self.resource_id})
            self._phase_parse()
            
            # Phase 2: Extract metadata (LLM calls)
            logger.info("Phase 2: Extracting metadata", extra={"chunks": len(self.chunks)})
            self._phase_extract()
            
            # Phase 3: Compute embeddings (batched)
            logger.info("Phase 3: Computing embeddings", extra={"chunks": len(self.chunks)})
            self._phase_embed()
            
            # Phase 4: Store in databases (batched)
            logger.info("Phase 4: Storing to databases", extra={"chunks": len(self.chunks)})
            self._phase_store()
            
        except Exception as e:
            logger.exception("Pipeline failed", extra={"resource_id": self.resource_id})
            self.result.errors.append(str(e))
        
        self.result.elapsed_ms = int((time.time() - t0) * 1000)
        
        logger.info(
            "Pipeline complete",
            extra={
                "resource_id": self.resource_id,
                "created": self.result.chunks_created,
                "updated": self.result.chunks_updated,
                "elapsed_ms": self.result.elapsed_ms,
                "errors": len(self.result.errors),
            }
        )
        
        return self.result
    
    # =========================================================================
    # Phase 1: Parse
    # =========================================================================
    
    def _phase_parse(self):
        """Parse document into raw chunks."""
        from .parse_utils import extract_text_by_type
        
        pages = extract_text_by_type(self.file_path, None)
        
        if self.mode == IngestMode.FAST:
            self._parse_fast(pages)
        else:
            self._parse_semantic(pages)
    
    def _parse_fast(self, pages: List[str]):
        """Fast parsing: simple sentence-based chunking, no LLM."""
        for page_idx, page_text in enumerate(pages, start=1):
            if not page_text or not page_text.strip():
                continue
            
            # Simple sentence-based splitting
            sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', page_text) if s.strip()]
            
            # Group sentences into chunks (target ~200 tokens)
            current_chunk = []
            current_tokens = 0
            source_offset = 0
            
            for sentence in sentences:
                tokens = len(sentence.split())
                
                if current_tokens + tokens > 200 and current_chunk:
                    # Emit chunk
                    text = " ".join(current_chunk)
                    self.chunks.append(ChunkData(
                        resource_id=self.resource_id,
                        page_number=page_idx,
                        page_start=page_idx,
                        page_end=page_idx,
                        source_offset=source_offset,
                        full_text=text,
                        text_snippet=text[:300],
                        token_count=current_tokens,
                    ))
                    current_chunk = []
                    current_tokens = 0
                    source_offset += len(text) + 1
                
                current_chunk.append(sentence)
                current_tokens += tokens
            
            # Flush remaining
            if current_chunk:
                text = " ".join(current_chunk)
                self.chunks.append(ChunkData(
                    resource_id=self.resource_id,
                    page_number=page_idx,
                    page_start=page_idx,
                    page_end=page_idx,
                    source_offset=source_offset,
                    full_text=text,
                    text_snippet=text[:300],
                    token_count=current_tokens,
                ))
    
    def _parse_semantic(self, pages: List[str]):
        """Semantic parsing: LLM-based structure detection."""
        from prompts import get as prompt_get, render as prompt_render
        from llm import call_llm_json
        
        for page_idx, page_text in enumerate(pages, start=1):
            if not page_text or not page_text.strip():
                continue
            
            # Trim page text for LLM
            max_chars = int(os.getenv("INGEST_PAGE_MAX_CHARS", "6000"))
            trimmed = page_text[:max_chars] if len(page_text) > max_chars else page_text
            
            # Get page structure from LLM
            tmpl = prompt_get("ingest.page_structure_v2") or prompt_get("ingest.page_structure")
            if tmpl:
                prompt = prompt_render(tmpl, {"page_number": page_idx, "page_text": trimmed})
                try:
                    structure = call_llm_json(prompt, {"sections": [], "chunks": []})
                except Exception:
                    logger.exception("LLM page structure failed", extra={"page": page_idx})
                    structure = {"sections": [], "chunks": []}
            else:
                structure = {"sections": [], "chunks": []}
            
            sections = structure.get("sections", [])
            chunk_entries = structure.get("chunks", [])
            
            # If LLM didn't return chunks, fall back to simple splitting
            if not chunk_entries:
                self._parse_fast([page_text])
                continue
            
            # Process LLM chunks
            text_len = len(page_text)
            for entry in chunk_entries:
                try:
                    start = int(entry.get("start", 0))
                    end = int(entry.get("end", 0))
                except (ValueError, TypeError):
                    continue
                
                if end <= start or start < 0 or end > text_len:
                    continue
                
                text = page_text[start:end].strip()
                if len(text) < 20:
                    continue
                
                # Find matching section
                section_title = entry.get("section_title", "")
                section_number = entry.get("section_number", "")
                section_level = None
                try:
                    section_level = int(entry.get("section_level", 0)) or None
                except (ValueError, TypeError):
                    pass
                
                # Check for figures/equations
                has_figure = bool(re.search(r'\b(?:Fig|Figure)\b', text, re.IGNORECASE))
                has_equation = bool(re.search(r'[=∫∑∏]|\$[^$]+\$|\\begin\{equation\}', text))
                
                chunk = ChunkData(
                    resource_id=self.resource_id,
                    page_number=page_idx,
                    page_start=page_idx,
                    page_end=page_idx,
                    source_offset=start,
                    full_text=text,
                    text_snippet=text[:300],
                    section_title=section_title,
                    section_number=section_number,
                    section_level=section_level,
                    section_path=self._build_section_path(section_number, section_title),
                    chunk_type=entry.get("type", ""),
                    has_figure=has_figure,
                    has_equation=has_equation,
                    token_count=len(text.split()),
                )
                
                self.chunks.append(chunk)
    
    def _build_section_path(self, number: str, title: str) -> List[str]:
        """Build section path from number and title."""
        if number:
            parts = [p for p in number.replace(" ", "").split(".") if p]
            if parts:
                return parts
        if title:
            return [title]
        return []
    
    # =========================================================================
    # Phase 2: Extract
    # =========================================================================
    
    def _phase_extract(self):
        """Extract metadata from chunks."""
        if self.mode == IngestMode.FAST:
            self._extract_heuristic()
        else:
            self._extract_unified()
    
    def _extract_heuristic(self):
        """Fast heuristic extraction without LLM."""
        for chunk in self.chunks:
            self._extract_heuristic_single(chunk)
    
    def _extract_unified(self):
        """Unified LLM extraction - single call per chunk."""
        from prompts import get as prompt_get, render as prompt_render
        from llm import call_llm_json
        
        # Create unified extraction prompt
        unified_prompt_template = self._get_unified_prompt()
        
        for chunk in self.chunks:
            # First, apply heuristics as baseline
            self._extract_heuristic_single(chunk)
            
            # Only call LLM for substantial chunks in FULL mode
            if self.mode == IngestMode.FULL and len(chunk.full_text) > 100:
                try:
                    result = self._extract_with_llm(chunk, unified_prompt_template)
                    self._apply_llm_result(chunk, result)
                except Exception:
                    logger.exception("LLM extraction failed", extra={"chunk_offset": chunk.source_offset})
    
    def _extract_heuristic_single(self, chunk: ChunkData):
        """Apply heuristic extraction to single chunk."""
        text_lower = chunk.full_text.lower()
        
        # Pedagogy role
        if any(p in text_lower for p in ["is defined as", "we define", "definition:"]):
            chunk.pedagogy_role = "definition"
        elif any(p in text_lower for p in ["for example", "for instance"]):
            chunk.pedagogy_role = "example"
        elif any(p in text_lower for p in ["proof:", "to prove", "q.e.d."]):
            chunk.pedagogy_role = "proof"
        elif any(p in text_lower for p in ["deriving", "derivation"]):
            chunk.pedagogy_role = "derivation"
        elif any(p in text_lower for p in ["problem:", "exercise:"]):
            chunk.pedagogy_role = "problem"
        else:
            chunk.pedagogy_role = "explanation"
        
        # Extract concepts using improved heuristics
        chunk.concepts = self._extract_concepts_heuristic(chunk.full_text)
        
        # Difficulty heuristic
        if any(w in text_lower for w in ["differential", "partial derivative", "laplacian", "fourier", "bessel"]):
            chunk.difficulty = "advanced"
        elif any(w in text_lower for w in ["integral", "derivative", "calculus", "equation", "coefficient"]):
            chunk.difficulty = "intermediate"
        else:
            chunk.difficulty = "introductory"
    
    def _extract_concepts_heuristic(self, text: str) -> List[str]:
        """Extract concepts from text using improved heuristics."""
        concepts = set()
        
        # Stop words to filter out
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "this", "that", "these",
            "those", "it", "its", "we", "our", "you", "your", "they", "their", "he",
            "she", "his", "her", "which", "who", "whom", "what", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "also", "now", "here", "there", "then",
            "chapter", "section", "figure", "table", "equation", "example", "problem",
            "solution", "note", "see", "given", "find", "determine", "calculate",
            "using", "applying", "substituting", "from", "into", "where", "therefore",
        }
        
        # Pattern 1: Multi-word capitalized phrases (e.g., "Heat Transfer")
        pattern1 = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
        for match in pattern1.finditer(text):
            phrase = match.group(1)
            words = phrase.lower().split()
            # Filter out phrases with stop words or too short
            if len(phrase) > 5 and not any(w in stop_words for w in words):
                concepts.add(phrase)
        
        # Pattern 2: Technical terms with common suffixes
        pattern2 = re.compile(r'\b([a-z]+(?:tion|sion|ance|ence|ity|ment|ness|ing|ive|ous))\b', re.IGNORECASE)
        technical_terms = {
            "conduction", "convection", "radiation", "diffusion", "absorption",
            "transmission", "reflection", "emission", "conductivity", "diffusivity",
            "capacitance", "resistance", "temperature", "gradient", "coefficient",
            "equilibrium", "steady-state", "transient", "boundary", "interface",
        }
        for match in pattern2.finditer(text.lower()):
            term = match.group(1).lower()
            if term in technical_terms:
                concepts.add(term.title())
        
        # Pattern 3: Named methods/laws (e.g., "Fourier's Law", "Newton's Law")
        pattern3 = re.compile(r"([A-Z][a-z]+)'s\s+(Law|Method|Equation|Number|Theorem|Principle)", re.IGNORECASE)
        for match in pattern3.finditer(text):
            concepts.add(match.group(0))
        
        # Pattern 4: Domain-specific terms from heat transfer
        heat_transfer_terms = [
            "heat transfer", "thermal conductivity", "thermal diffusivity",
            "convection coefficient", "heat flux", "temperature distribution",
            "lumped capacitance", "biot number", "fourier number", "nusselt number",
            "thermal resistance", "thermal capacity", "steady state", "transient",
            "boundary condition", "initial condition", "heat generation",
            "fin efficiency", "heat exchanger", "reynolds number", "prandtl number",
        ]
        for term in heat_transfer_terms:
            if term in text.lower():
                concepts.add(term.title())
        
        # Limit and sort
        return sorted(list(concepts))[:10]
    
    def _get_unified_prompt(self) -> str:
        """Get unified extraction prompt template from prompts file or fallback."""
        try:
            from prompts import get as prompt_get
            template = prompt_get("ingest.unified_extract")
            if template:
                return template
        except Exception:
            pass
        
        # Fallback template
        return """Analyze this educational content and extract structured metadata.

Text:
{{text}}

Section context: {{section_title}}

Extract ALL of the following in a single JSON response:
{
  "concepts": ["concept1", "concept2"],
  "prerequisites": ["prereq1"],
  "pedagogy_role": "explanation|definition|example|derivation|proof|problem|summary",
  "domain": "Physics|Mathematics|etc",
  "topic": "specific topic",
  "subtopic": "specific subtopic",
  "difficulty": "introductory|intermediate|advanced",
  "cognitive_level": "remember|understand|apply|analyze|evaluate|create",
  "formulas": [
    {
      "latex": "F = ma",
      "variables": [{"symbol": "F", "meaning": "Force", "units": "N"}],
      "type": "equation"
    }
  ]
}

Return ONLY valid JSON, no other text."""
    
    def _extract_with_llm(self, chunk: ChunkData, template: str) -> Dict[str, Any]:
        """Extract metadata using LLM."""
        from prompts import render as prompt_render
        from llm import call_llm_json
        
        prompt = template.replace("{{text}}", chunk.full_text[:2000])
        default = {
            "concepts": [],
            "prerequisites": [],
            "pedagogy_role": "explanation",
            "difficulty": "intermediate",
            "formulas": []
        }
        
        return call_llm_json(prompt, default) or default
    
    def _apply_llm_result(self, chunk: ChunkData, result: Dict[str, Any]):
        """Apply LLM extraction result to chunk."""
        if result.get("concepts"):
            chunk.concepts = result["concepts"][:10]
        if result.get("prerequisites"):
            chunk.prerequisites = result["prerequisites"][:5]
        if result.get("pedagogy_role"):
            chunk.pedagogy_role = result["pedagogy_role"]
        if result.get("domain"):
            chunk.domain = result["domain"]
        if result.get("topic"):
            chunk.topic = result["topic"]
        if result.get("subtopic"):
            chunk.subtopic = result["subtopic"]
        if result.get("difficulty"):
            chunk.difficulty = result["difficulty"]
        if result.get("cognitive_level"):
            chunk.cognitive_level = result["cognitive_level"]
        if result.get("formulas"):
            chunk.formulas = result["formulas"]
    
    # =========================================================================
    # Phase 3: Embed
    # =========================================================================
    
    def _phase_embed(self):
        """Compute embeddings for all chunks (batched)."""
        from ingestion import embed as embed_service
        
        texts = [c.full_text for c in self.chunks]
        
        # Batch embed
        try:
            embeddings = embed_service.embed_texts(texts)
            for chunk, embedding in zip(self.chunks, embeddings):
                chunk.embedding = embedding
        except Exception:
            logger.exception("Batch embedding failed")
            # Fallback to individual embedding
            for chunk in self.chunks:
                try:
                    chunk.embedding = embed_service.embed_text(chunk.full_text)
                except Exception:
                    logger.warning("Embedding failed for chunk", extra={"offset": chunk.source_offset})
    
    # =========================================================================
    # Phase 4: Store
    # =========================================================================
    
    def _phase_store(self):
        """Store chunks to databases."""
        # Store to Postgres first
        self._store_postgres()
        
        # Then store to Neo4j (if not FAST mode)
        if self.mode != IngestMode.FAST:
            self._store_neo4j()
    
    def _store_postgres(self):
        """Store chunks to Postgres with batched inserts."""
        with DatabasePool.postgres() as conn:
            if conn is None:
                self.result.errors.append("Postgres connection failed")
                return
            
            try:
                from psycopg2.extras import execute_values, Json
            except ImportError:
                self.result.errors.append("psycopg2 not available")
                return
            
            embed_version = os.getenv("EMBED_VERSION", "all-MiniLM-L6-v2-2025-09")
            
            try:
                with conn.cursor() as cur:
                    # Delete existing chunks for this resource
                    cur.execute(
                        "DELETE FROM chunk WHERE resource_id = %s::uuid",
                        (self.resource_id,)
                    )
                    
                    # Batch insert
                    for i in range(0, len(self.chunks), self.batch_size):
                        batch = self.chunks[i:i + self.batch_size]
                        
                        values = []
                        for chunk in batch:
                            # Format embedding as vector literal
                            vec_lit = None
                            if chunk.embedding:
                                vec_lit = "[" + ",".join(f"{float(x):.6f}" for x in chunk.embedding) + "]"
                            
                            # Build search text
                            heading = " ".join(filter(None, [chunk.section_number, chunk.section_title]))
                            
                            # Format arrays as PostgreSQL array literals
                            def to_pg_array(lst):
                                if not lst:
                                    return "{}"
                                escaped = [str(x).replace("\\", "\\\\").replace('"', '\\"') for x in lst]
                                return "{" + ",".join(f'"{x}"' for x in escaped) + "}"
                            
                            values.append((
                                self.resource_id,
                                chunk.page_number,
                                chunk.source_offset,
                                chunk.full_text,
                                chunk.chunk_type or chunk.pedagogy_role,
                                to_pg_array(chunk.concepts),
                                chunk.text_snippet,
                                vec_lit,
                                embed_version,
                                Json(chunk.to_tags_json()),
                                chunk.section_title,
                                chunk.section_number,
                                to_pg_array(chunk.section_path),
                                chunk.section_level,
                                chunk.page_start,
                                chunk.page_end,
                                chunk.token_count,
                                chunk.has_figure,
                                chunk.has_equation,
                                to_pg_array(chunk.figure_labels),
                                to_pg_array(chunk.equation_labels),
                                heading,
                            ))
                        
                        # Execute batch insert using execute_values
                        execute_values(
                            cur,
                            """
                            INSERT INTO chunk (
                                resource_id, page_number, source_offset, full_text,
                                chunk_type, concepts, text_snippet, embedding, embedding_version,
                                tags, section_title, section_number, section_path, section_level,
                                page_start, page_end, token_count, has_figure, has_equation,
                                figure_labels, equation_labels,
                                search_tsv, created_at, updated_at
                            )
                            SELECT 
                                v.resource_id::uuid, v.page_number::int, v.source_offset::int, v.full_text,
                                v.chunk_type, v.concepts::text[], v.text_snippet, 
                                v.embedding::vector, v.embedding_version,
                                v.tags::jsonb, v.section_title, v.section_number, v.section_path::text[], v.section_level::int,
                                v.page_start::int, v.page_end::int, v.token_count::int, v.has_figure::bool, v.has_equation::bool,
                                v.figure_labels::text[], v.equation_labels::text[],
                                setweight(to_tsvector('english', coalesce(v.heading, '')), 'A')
                                    || setweight(to_tsvector('english', v.full_text), 'B'),
                                now(), now()
                            FROM (VALUES %s) AS v(
                                resource_id, page_number, source_offset, full_text,
                                chunk_type, concepts, text_snippet, embedding, embedding_version,
                                tags, section_title, section_number, section_path, section_level,
                                page_start, page_end, token_count, has_figure, has_equation,
                                figure_labels, equation_labels, heading
                            )
                            """,
                            values
                        )
                        
                        self.result.chunks_created += len(batch)
                    
                    conn.commit()
                    
            except Exception as e:
                logger.exception("Postgres store failed")
                self.result.errors.append(f"Postgres error: {str(e)}")
                try:
                    conn.rollback()
                except Exception:
                    pass
    
    def _store_neo4j(self):
        """Store pedagogical knowledge graph to Neo4j.
        
        Creates:
        - Concept nodes with difficulty, frequency, first_page
        - Chunk nodes with pedagogy_role
        - OCCURS_IN: Concept → Chunk (with pedagogy_role)
        - PREREQUISITE_OF: Concept → Concept (LLM-inferred)
        - RELATED_TO: Concept → Concept (from strong co-occurrence)
        - TEACHES: Chunk → Concept (for definition/explanation chunks)
        """
        with DatabasePool.neo4j_session() as session:
            if session is None:
                logger.warning("Neo4j not available, skipping KG storage")
                return
            
            try:
                # Build pedagogical graph data
                graph_data = self._build_pedagogical_graph()
                
                # Always use LLM to build prerequisites (use lite model for FAST mode)
                if graph_data["concepts"]:
                    use_lite = self.mode in (IngestMode.FAST, IngestMode.STANDARD)
                    llm_prereqs = self._build_prerequisites_llm(graph_data["concepts"], use_lite=use_lite)
                    graph_data["prerequisites"].extend(llm_prereqs)
                
                # Batch create concepts with metadata
                self._batch_create_concepts_pedagogical(session, graph_data["concepts"])
                
                # Batch create chunk nodes
                self._batch_create_chunk_nodes(session, graph_data["chunks"])
                
                # Batch create OCCURS_IN relationships
                self._batch_create_occurrences_pedagogical(session, graph_data["occurrences"])
                
                # Batch create PREREQUISITE_OF relationships
                self._batch_create_prerequisites(session, graph_data["prerequisites"])
                
                # Batch create RELATED_TO relationships
                self._batch_create_related(session, graph_data["related"])
                
                # Batch create TEACHES relationships
                self._batch_create_teaches(session, graph_data["teaches"])
                
                self.result.concepts_created = len(graph_data["concepts"])
                self.result.relationships_created = (
                    len(graph_data["prerequisites"]) + 
                    len(graph_data["related"]) + 
                    len(graph_data["teaches"])
                )
                
            except Exception as e:
                logger.exception("Neo4j store failed")
                self.result.errors.append(f"Neo4j error: {str(e)}")
    
    def _build_prerequisites_llm(self, concepts: Dict[str, Dict], use_lite: bool = False) -> List[Tuple]:
        """Use LLM to build high-quality prerequisite relationships.
        
        Args:
            concepts: Dict of concept data
            use_lite: If True, use lightweight model (gemini-2.0-flash-lite) for speed
        """
        try:
            from prompts import get as prompt_get, render as prompt_render
            from llm.common import call_json_chat
        except ImportError:
            logger.warning("LLM not available for prerequisite building")
            return []
        
        # Get the prompt template
        template = prompt_get("ingest.build_prerequisites")
        if not template:
            logger.warning("build_prerequisites prompt not found")
            return []
        
        # Format concept list with metadata
        concept_list = []
        for canonical, data in concepts.items():
            concept_list.append(f"- {data['display']} (difficulty: {data.get('difficulty', 'unknown')}, " +
                              f"first_page: {data.get('first_page', '?')})")
        
        # Infer domain from concepts
        domain = "General"
        domain_keywords = {
            "Physics": ["heat", "thermal", "energy", "force", "motion", "wave"],
            "Mathematics": ["equation", "derivative", "integral", "function"],
            "Chemistry": ["reaction", "bond", "molecule", "element"],
            "Biology": ["cell", "organism", "gene", "protein"],
        }
        concept_text = " ".join(c["display"].lower() for c in concepts.values())
        for d, keywords in domain_keywords.items():
            if any(k in concept_text for k in keywords):
                domain = d
                break
        
        # Build prompt
        prompt = prompt_render(template, {
            "concepts_list": "\n".join(concept_list),
            "domain": domain,
        })
        
        # Select model
        if use_lite:
            model = "google/gemini-2.0-flash-lite-001"
        else:
            model = None  # Use default model
        
        # Call LLM
        try:
            result = call_json_chat(
                prompt,
                default={"prerequisites": [], "learning_order": []},
                model_hint=model,
                max_tokens=2000,
            )
        except Exception as e:
            logger.warning(f"LLM prerequisite call failed: {e}")
            return []
        
        # Parse results
        prerequisites = []
        for item in result.get("prerequisites", []):
            concept = item.get("concept", "")
            requires = item.get("requires", [])
            confidence = item.get("confidence", 0.5)
            
            # Only include high-confidence relationships
            if confidence < 0.7:
                continue
            
            concept_canonical = self._canonicalize(concept)
            if not concept_canonical or concept_canonical not in concepts:
                continue
            
            for req in requires:
                req_canonical = self._canonicalize(req)
                if req_canonical and req_canonical in concepts and req_canonical != concept_canonical:
                    prerequisites.append((req_canonical, concept_canonical, confidence, "llm"))
        
        model_used = "gemini-flash-lite" if use_lite else "default"
        logger.info(f"LLM ({model_used}) extracted {len(prerequisites)} prerequisite relationships")
        return prerequisites
    
    def _build_pedagogical_graph(self) -> Dict[str, Any]:
        """Build pedagogical graph data from chunks.
        
        Returns dict with:
        - concepts: {canonical: {display, difficulty, frequency, first_page, pedagogy_roles}}
        - chunks: [(chunk_id, resource_id, page, pedagogy_role, difficulty, snippet)]
        - occurrences: [(concept_canonical, chunk_id, pedagogy_role)]
        - prerequisites: [(from_concept, to_concept, confidence, evidence)]
        - related: [(concept_a, concept_b, weight)]
        - teaches: [(chunk_id, concept_canonical)] for definition/explanation chunks
        """
        from collections import defaultdict
        
        concepts: Dict[str, Dict[str, Any]] = {}
        chunks: List[Tuple] = []
        occurrences: List[Tuple] = []
        prerequisites: List[Tuple] = []
        related: List[Tuple] = []
        teaches: List[Tuple] = []
        
        # Track concept co-occurrences for RELATED_TO
        concept_cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Track concept first appearance order for prerequisite inference
        concept_first_page: Dict[str, int] = {}
        concept_difficulties: Dict[str, List[str]] = defaultdict(list)
        
        # Difficulty ordering for prerequisite inference
        difficulty_order = {"introductory": 1, "intermediate": 2, "advanced": 3}
        
        for chunk in self.chunks:
            chunk_id = f"{self.resource_id}:{chunk.page_number}:{chunk.source_offset}"
            
            # Add chunk data
            chunks.append((
                chunk_id,
                self.resource_id,
                chunk.page_number,
                chunk.pedagogy_role,
                chunk.difficulty,
                chunk.text_snippet[:200] if chunk.text_snippet else chunk.full_text[:200]
            ))
            
            chunk_concepts_canonical = []
            
            for concept in chunk.concepts:
                canonical = self._canonicalize(concept)
                if not canonical:
                    continue
                
                chunk_concepts_canonical.append(canonical)
                
                # Track concept metadata
                if canonical not in concepts:
                    concepts[canonical] = {
                        "display": concept,
                        "frequency": 0,
                        "first_page": chunk.page_number,
                        "pedagogy_roles": defaultdict(int),
                        "difficulties": [],
                    }
                
                concepts[canonical]["frequency"] += 1
                concepts[canonical]["pedagogy_roles"][chunk.pedagogy_role] += 1
                concepts[canonical]["difficulties"].append(chunk.difficulty)
                
                # Track first appearance
                if canonical not in concept_first_page:
                    concept_first_page[canonical] = chunk.page_number
                concept_difficulties[canonical].append(chunk.difficulty)
                
                # Add occurrence
                occurrences.append((canonical, chunk_id, chunk.pedagogy_role))
                
                # If this chunk defines/explains, create TEACHES relationship
                if chunk.pedagogy_role in ("definition", "explanation"):
                    teaches.append((chunk_id, canonical))
            
            # Track co-occurrences for RELATED_TO
            for i, c1 in enumerate(chunk_concepts_canonical):
                for c2 in chunk_concepts_canonical[i+1:]:
                    if c1 < c2:
                        concept_cooccurrence[(c1, c2)] += 1
                    else:
                        concept_cooccurrence[(c2, c1)] += 1
            
            # Add explicit prerequisites from chunk
            for prereq in chunk.prerequisites:
                prereq_canonical = self._canonicalize(prereq)
                if not prereq_canonical:
                    continue
                
                # Add prereq as concept if not seen
                if prereq_canonical not in concepts:
                    concepts[prereq_canonical] = {
                        "display": prereq,
                        "frequency": 0,
                        "first_page": 0,
                        "pedagogy_roles": defaultdict(int),
                        "difficulties": [],
                    }
                
                # Create prerequisite edges to all concepts in this chunk
                for concept in chunk.concepts[:3]:
                    concept_canonical = self._canonicalize(concept)
                    if concept_canonical and concept_canonical != prereq_canonical:
                        prerequisites.append((
                            prereq_canonical,
                            concept_canonical,
                            0.8,  # High confidence for explicit
                            chunk_id
                        ))
        
        # In FULL mode, use LLM to build prerequisites after collecting all concepts
        # This is done separately in _build_prerequisites_llm() called after graph building
        # For now, only explicit prerequisites from chunks are included
        
        # Build RELATED_TO from co-occurrences
        # Higher threshold to reduce noise - concepts must appear together frequently
        min_cooccurrence = max(3, len(self.chunks) // 6)  # At least 3, scales with doc size
        
        for (c1, c2), count in concept_cooccurrence.items():
            if count >= min_cooccurrence:
                weight = min(1.0, count / 10.0)  # Normalize weight
                related.append((c1, c2, weight))
        
        # Compute dominant difficulty for each concept
        for canonical, data in concepts.items():
            if data["difficulties"]:
                # Most common difficulty
                diff_counts = defaultdict(int)
                for d in data["difficulties"]:
                    diff_counts[d] += 1
                data["difficulty"] = max(diff_counts, key=diff_counts.get)
            else:
                data["difficulty"] = "intermediate"
            
            # Dominant pedagogy role
            if data["pedagogy_roles"]:
                data["primary_role"] = max(data["pedagogy_roles"], key=data["pedagogy_roles"].get)
            else:
                data["primary_role"] = "explanation"
        
        return {
            "concepts": concepts,
            "chunks": chunks,
            "occurrences": occurrences,
            "prerequisites": prerequisites,
            "related": related,
            "teaches": teaches,
        }
    
    def _infer_prerequisites(
        self,
        concepts: Dict[str, Dict],
        first_page: Dict[str, int],
        difficulties: Dict[str, List[str]],
        difficulty_order: Dict[str, int],
    ) -> List[Tuple]:
        """Infer prerequisite relationships from document structure.
        
        CONSERVATIVE heuristic - only infer prerequisites when:
        1. Concepts co-occur (appear in same chunk) AND
        2. One is significantly easier AND appears earlier
        
        This prevents noisy edges between unrelated concepts.
        """
        # Don't use heuristic inference - too noisy
        # Prerequisites should come from:
        # 1. Explicit extraction (FULL mode with LLM)
        # 2. Co-occurrence patterns (related concepts only)
        # 3. Domain knowledge (hardcoded for common domains)
        return []
    
    def _canonicalize(self, name: str) -> str:
        """Canonicalize concept name."""
        import unicodedata
        
        raw = (name or "").strip()
        if not raw:
            return ""
        
        # Normalize unicode
        normalized = unicodedata.normalize("NFKD", raw.lower())
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        
        # Clean
        normalized = normalized.replace("-", " ")
        normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        
        return normalized
    
    def _batch_create_concepts(self, session, concepts: Dict[str, str]):
        """Batch create concept nodes."""
        if not concepts:
            return
        
        def _tx(tx):
            tx.run("""
                UNWIND $concepts AS c
                MERGE (concept:Concept {canonical_name: c.canonical})
                ON CREATE SET 
                    concept.display_name = c.display,
                    concept.name_lower = c.canonical,
                    concept.created_at = datetime()
                SET concept.last_seen = datetime()
            """, concepts=[
                {"canonical": k, "display": v}
                for k, v in concepts.items()
            ])
        
        try:
            session.execute_write(_tx)
        except Exception:
            logger.exception("Batch concept creation failed")
    
    def _batch_create_occurrences(self, session, chunk_concepts: List[Tuple[str, str, str, str]]):
        """Batch create OCCURS_IN relationships."""
        if not chunk_concepts:
            return
        
        def _tx(tx):
            tx.run("""
                UNWIND $rels AS r
                MERGE (ch:Chunk {id: r.chunk_id})
                SET ch.resource_id = $resource_id, ch.snippet = r.snippet
                WITH ch, r
                MATCH (c:Concept {canonical_name: r.canonical})
                MERGE (c)-[rel:OCCURS_IN]->(ch)
                SET rel.last_seen = datetime()
            """, 
                rels=[
                    {"chunk_id": c[0], "canonical": c[1], "display": c[2], "snippet": c[3][:200]}
                    for c in chunk_concepts
                ],
                resource_id=self.resource_id
            )
        
        try:
            session.execute_write(_tx)
        except Exception:
            logger.exception("Batch occurrence creation failed")
    
    def _batch_create_prerequisites(self, session, relationships: List[Tuple]):
        """Batch create PREREQUISITE_OF relationships."""
        if not relationships:
            return
        
        def _tx(tx):
            tx.run("""
                UNWIND $rels AS r
                MERGE (prereq:Concept {canonical_name: r.prereq})
                ON CREATE SET prereq.created_at = datetime()
                MERGE (target:Concept {canonical_name: r.target})
                ON CREATE SET target.created_at = datetime()
                MERGE (prereq)-[rel:PREREQUISITE_OF]->(target)
                SET rel.confidence = r.confidence,
                    rel.evidence = r.evidence,
                    rel.method = 'ingestion_pipeline',
                    rel.last_seen = datetime()
            """, rels=[
                {"prereq": r[0], "target": r[1], "confidence": r[2], "evidence": r[3]}
                for r in relationships
            ])
        
        try:
            session.execute_write(_tx)
        except Exception:
            logger.exception("Batch prerequisite creation failed")
    
    def _batch_create_concepts_pedagogical(self, session, concepts: Dict[str, Dict]):
        """Batch create concept nodes with pedagogical metadata."""
        if not concepts:
            return
        
        def _tx(tx):
            tx.run("""
                UNWIND $concepts AS c
                MERGE (concept:Concept {canonical_name: c.canonical})
                ON CREATE SET 
                    concept.created_at = datetime()
                SET concept.display_name = c.display,
                    concept.name_lower = c.canonical,
                    concept.difficulty = c.difficulty,
                    concept.frequency = c.frequency,
                    concept.first_page = c.first_page,
                    concept.primary_role = c.primary_role,
                    concept.last_seen = datetime()
            """, concepts=[
                {
                    "canonical": k,
                    "display": v["display"],
                    "difficulty": v.get("difficulty", "intermediate"),
                    "frequency": v.get("frequency", 1),
                    "first_page": v.get("first_page", 1),
                    "primary_role": v.get("primary_role", "explanation"),
                }
                for k, v in concepts.items()
            ])
        
        try:
            session.execute_write(_tx)
        except Exception:
            logger.exception("Batch pedagogical concept creation failed")
    
    def _batch_create_chunk_nodes(self, session, chunks: List[Tuple]):
        """Batch create Chunk nodes with pedagogy metadata."""
        if not chunks:
            return
        
        def _tx(tx):
            tx.run("""
                UNWIND $chunks AS c
                MERGE (ch:Chunk {id: c.id})
                SET ch.resource_id = c.resource_id,
                    ch.page_number = c.page,
                    ch.pedagogy_role = c.role,
                    ch.difficulty = c.difficulty,
                    ch.snippet = c.snippet,
                    ch.last_seen = datetime()
            """, chunks=[
                {
                    "id": c[0],
                    "resource_id": c[1],
                    "page": c[2],
                    "role": c[3],
                    "difficulty": c[4],
                    "snippet": c[5],
                }
                for c in chunks
            ])
        
        try:
            session.execute_write(_tx)
        except Exception:
            logger.exception("Batch chunk node creation failed")
    
    def _batch_create_occurrences_pedagogical(self, session, occurrences: List[Tuple]):
        """Batch create OCCURS_IN relationships with pedagogy role."""
        if not occurrences:
            return
        
        def _tx(tx):
            tx.run("""
                UNWIND $rels AS r
                MATCH (c:Concept {canonical_name: r.canonical})
                MATCH (ch:Chunk {id: r.chunk_id})
                MERGE (c)-[rel:OCCURS_IN]->(ch)
                SET rel.pedagogy_role = r.role,
                    rel.last_seen = datetime()
            """, rels=[
                {"canonical": r[0], "chunk_id": r[1], "role": r[2]}
                for r in occurrences
            ])
        
        try:
            session.execute_write(_tx)
        except Exception:
            logger.exception("Batch occurrence creation failed")
    
    def _batch_create_related(self, session, related: List[Tuple]):
        """Batch create RELATED_TO relationships from co-occurrence."""
        if not related:
            return
        
        def _tx(tx):
            tx.run("""
                UNWIND $rels AS r
                MATCH (c1:Concept {canonical_name: r.c1})
                MATCH (c2:Concept {canonical_name: r.c2})
                MERGE (c1)-[rel:RELATED_TO]-(c2)
                SET rel.weight = r.weight,
                    rel.method = 'cooccurrence',
                    rel.last_seen = datetime()
            """, rels=[
                {"c1": r[0], "c2": r[1], "weight": r[2]}
                for r in related
            ])
        
        try:
            session.execute_write(_tx)
        except Exception:
            logger.exception("Batch related creation failed")
    
    def _batch_create_teaches(self, session, teaches: List[Tuple]):
        """Batch create TEACHES relationships (Chunk teaches Concept)."""
        if not teaches:
            return
        
        def _tx(tx):
            tx.run("""
                UNWIND $rels AS r
                MATCH (ch:Chunk {id: r.chunk_id})
                MATCH (c:Concept {canonical_name: r.canonical})
                MERGE (ch)-[rel:TEACHES]->(c)
                SET rel.last_seen = datetime()
            """, rels=[
                {"chunk_id": r[0], "canonical": r[1]}
                for r in teaches
            ])
        
        try:
            session.execute_write(_tx)
        except Exception:
            logger.exception("Batch teaches creation failed")


def ingest_resource(resource_id: str, file_path: str, mode: str = "standard") -> IngestResult:
    """Convenience function to run ingestion pipeline.
    
    Args:
        resource_id: UUID of the resource
        file_path: Path to the document file
        mode: Ingestion mode ("fast", "standard", "full")
    
    Returns:
        IngestResult with statistics and errors
    """
    try:
        ingest_mode = IngestMode(mode.lower())
    except ValueError:
        ingest_mode = IngestMode.STANDARD
    
    pipeline = IngestPipeline(
        resource_id=resource_id,
        file_path=file_path,
        mode=ingest_mode,
    )
    
    return pipeline.run()
