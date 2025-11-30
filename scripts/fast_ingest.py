#!/usr/bin/env python3
"""Fast ingestion script for educational documents.

Usage:
    python scripts/fast_ingest.py path/to/document.pdf
    python scripts/fast_ingest.py path/to/document.pdf --title "My Document"
    
Uses google/gemini-2.5-flash-lite for LLM calls (knowledge graph building).
"""
import argparse
import os
import sys
import uuid
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import re
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Model for LLM calls
FAST_INGEST_MODEL = "google/gemini-2.5-flash-lite"


@dataclass
class FastIngestResult:
    """Result of fast ingestion."""
    resource_id: str
    title: str
    chunks_created: int = 0
    concepts_extracted: int = 0
    prerequisites_inferred: int = 0
    related_concepts: int = 0
    elapsed_ms: int = 0
    errors: List[str] = field(default_factory=list)


def get_db_connection():
    """Get Postgres connection with Docker/host fallback."""
    import psycopg2
    import socket
    import re
    
    # Check if we can reach Docker's postgres hostname
    def is_docker_context():
        try:
            socket.gethostbyname("postgres")
            return True
        except socket.error:
            return False
    
    in_docker = is_docker_context()
    
    # Use explicit host/port for non-Docker context
    if not in_docker:
        host = "localhost"
        port = "5433"
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        db = os.getenv("POSTGRES_DB", "app")
        logger.info(f"Host context detected, connecting to: {host}:{port}/{db}")
        return psycopg2.connect(host=host, port=port, user=user, password=password, dbname=db)
    
    # Docker context - use DATABASE_URL or env vars
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        logger.info("Docker context, using DATABASE_URL")
        return psycopg2.connect(dsn)
    
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "app")
    
    logger.info(f"Docker context, connecting to: {host}:{port}/{db}")
    return psycopg2.connect(host=host, port=port, user=user, password=password, dbname=db)


def create_resource(conn, resource_id: str, title: str, file_path: str) -> bool:
    """Create resource entry in database."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO resource (id, title, filename, storage_path, created_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title
            """, (resource_id, title, Path(file_path).name, file_path))
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to create resource: {e}")
        conn.rollback()
        return False


def parse_pdf(file_path: str) -> List[str]:
    """Parse PDF into pages of text."""
    # Try PyMuPDF first (fastest)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        pages = []
        for page in doc:
            text = page.get_text()
            pages.append(text)
        doc.close()
        logger.info(f"Parsed with PyMuPDF: {len(pages)} pages")
        return pages
    except ImportError:
        pass
    
    # Try pdfplumber
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
        logger.info(f"Parsed with pdfplumber: {len(pages)} pages")
        return pages
    except ImportError:
        pass
    
    # Try pdfminer (available in Docker)
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        from io import StringIO
        
        # Extract all text (pdfminer doesn't easily give per-page)
        output = StringIO()
        with open(file_path, 'rb') as f:
            extract_text_to_fp(f, output, laparams=LAParams())
        
        full_text = output.getvalue()
        
        # Split by form feed (page break) if present, otherwise by large gaps
        if '\f' in full_text:
            pages = [p.strip() for p in full_text.split('\f') if p.strip()]
        else:
            # Approximate page splits (~3000 chars per page)
            pages = []
            chunk_size = 3000
            for i in range(0, len(full_text), chunk_size):
                page_text = full_text[i:i + chunk_size].strip()
                if page_text:
                    pages.append(page_text)
        
        logger.info(f"Parsed with pdfminer: {len(pages)} pages")
        return pages
    except ImportError:
        pass
    
    logger.error("No PDF library available (install pymupdf, pdfplumber, or pdfminer.six)")
    return []


def chunk_text(pages: List[str], target_tokens: int = 200) -> List[Dict]:
    """Chunk pages into smaller pieces."""
    chunks = []
    
    for page_idx, page_text in enumerate(pages, start=1):
        if not page_text or not page_text.strip():
            continue
        
        # Simple sentence-based splitting
        sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', page_text) if s.strip()]
        
        current_chunk = []
        current_tokens = 0
        source_offset = 0
        
        for sentence in sentences:
            tokens = len(sentence.split())
            
            if current_tokens + tokens > target_tokens and current_chunk:
                text = " ".join(current_chunk)
                chunks.append({
                    "page_number": page_idx,
                    "source_offset": source_offset,
                    "full_text": text,
                    "text_snippet": text[:200] if len(text) > 200 else text,
                    "token_count": current_tokens,
                })
                source_offset += len(text)
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += tokens
        
        # Emit remaining
        if current_chunk:
            text = " ".join(current_chunk)
            chunks.append({
                "page_number": page_idx,
                "source_offset": source_offset,
                "full_text": text,
                "text_snippet": text[:200] if len(text) > 200 else text,
                "token_count": current_tokens,
            })
    
    return chunks


def extract_concepts_heuristic(text: str) -> Tuple[List[str], str, str]:
    """Extract concepts, pedagogy role, and difficulty using heuristics."""
    
    # Stopwords and noise phrases to filter out
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those',
        'it', 'its', 'we', 'they', 'he', 'she', 'you', 'i', 'me', 'my', 'your',
        'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too', 'very',
        'just', 'also', 'now', 'here', 'there', 'then', 'once', 'if', 'and', 'or',
        'but', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
        'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    }
    
    # Common noise phrases to exclude
    NOISE_PHRASES = {
        'for example', 'we have', 'it is', 'we get', 'in general', 'let us',
        'if and only if', 'such that', 'given that', 'note that', 'we can',
        'this is', 'that is', 'there is', 'there are', 'we see', 'we know',
        'in this', 'in the', 'on the', 'at the', 'to the', 'from the',
        'as shown', 'is given', 'are given', 'is called', 'are called',
        'concepts of physics', 'hc verma', 'chapter', 'figure', 'example',
    }
    
    # Physics-specific concept patterns (high precision)
    physics_patterns = [
        # Named laws/principles: "Newton's Law", "Ohm's Law"
        r"\b([A-Z][a-z]+(?:'s)?\s+(?:law|laws|theorem|principle|principles|rule|rules|equation|equations|hypothesis|postulate))\b",
        # Named effects/phenomena: "Doppler Effect", "Photoelectric Effect"  
        r"\b([A-Z][a-z]+(?:'s)?\s+(?:effect|phenomenon|paradox|experiment))\b",
        # Named constants/numbers: "Avogadro Number", "Planck Constant"
        r"\b([A-Z][a-z]+(?:'s)?\s+(?:number|constant|coefficient|factor|ratio))\b",
        # Force types: "Gravitational Force", "Frictional Force"
        r"\b([A-Z][a-z]+(?:al|ic|ive)?\s+(?:force|forces|field|fields|energy|motion|wave|waves))\b",
        # Physics concepts: "Angular Momentum", "Kinetic Energy"
        r"\b((?:angular|linear|kinetic|potential|mechanical|thermal|electric|magnetic|gravitational|centripetal|centrifugal|nuclear|atomic)\s+(?:momentum|energy|force|velocity|acceleration|displacement|field|potential|power|work))\b",
        # Specific physics terms
        r"\b((?:Newton|Joule|Watt|Pascal|Hertz|Coulomb|Ampere|Volt|Ohm|Farad|Tesla|Weber|Henry|Kelvin|Celsius|Fahrenheit)(?:'s)?(?:\s+(?:law|unit|force|constant|equation))?)\b",
    ]
    
    concepts = set()
    text_lower = text.lower()
    
    for pattern in physics_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            term = match.group(1).strip()
            
            # Clean up
            words = term.split()
            if len(words) < 2 or len(words) > 4:
                continue
            
            # Skip if starts with stopword
            if words[0].lower() in STOPWORDS:
                continue
            
            # Skip noise phrases
            if term.lower() in NOISE_PHRASES:
                continue
            
            # Skip if all words are stopwords
            non_stop = [w for w in words if w.lower() not in STOPWORDS]
            if len(non_stop) < 1:
                continue
            
            # Title case and add
            clean_term = " ".join(w.capitalize() for w in words)
            concepts.add(clean_term)
    
    # Pedagogy role detection
    role = "explanation"
    if any(kw in text_lower for kw in ["example", "consider", "suppose", "given that", "calculate"]):
        role = "example"
    elif any(kw in text_lower for kw in ["define", "is defined as", "refers to", "means"]):
        role = "definition"
    elif any(kw in text_lower for kw in ["derive", "derivation", "proof", "therefore", "hence"]):
        role = "derivation"
    elif any(kw in text_lower for kw in ["problem", "exercise", "find", "determine", "solve"]):
        role = "problem"
    
    # Difficulty detection
    difficulty = "intermediate"
    if any(kw in text_lower for kw in ["basic", "introduction", "fundamental", "simple"]):
        difficulty = "introductory"
    elif any(kw in text_lower for kw in ["advanced", "complex", "rigorous", "comprehensive"]):
        difficulty = "advanced"
    
    return list(concepts)[:8], role, difficulty


def compute_embeddings(chunks: List[Dict], batch_size: int = 32) -> List[List[float]]:
    """Compute embeddings for chunks."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        texts = [c["full_text"] for c in chunks]
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs = model.encode(batch, show_progress_bar=False)
            embeddings.extend([v.tolist() for v in vecs])
        
        return embeddings
    except ImportError:
        logger.warning("sentence-transformers not available, using pseudo-embeddings")
        return [[0.0] * 384 for _ in chunks]


def build_prerequisites_llm(concepts: Dict[str, Dict], domain: str = "Physics", batch_size: int = 12) -> List[Tuple]:
    """Use LLM to infer prerequisite relationships with batching.
    
    Args:
        concepts: Dict of concept data
        domain: Subject domain
        batch_size: Max concepts per LLM call (default 12)
    """
    try:
        from llm.common import call_json_chat
    except ImportError:
        logger.warning("LLM module not available")
        return []
    
    if not concepts:
        return []
    
    # Sort concepts by frequency (most important first)
    sorted_concepts = sorted(
        concepts.items(),
        key=lambda x: x[1].get('frequency', 0),
        reverse=True
    )
    
    # Take top concepts (limit to avoid too many batches)
    max_concepts = min(60, len(sorted_concepts))
    top_concepts = dict(sorted_concepts[:max_concepts])
    
    all_prerequisites = []
    
    # Process in batches
    concept_items = list(top_concepts.items())
    num_batches = (len(concept_items) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(concept_items)} concepts in {num_batches} batches of {batch_size}")
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(concept_items))
        batch = concept_items[start:end]
        
        # Build prompt for this batch
        concept_list = []
        for canonical, data in batch:
            concept_list.append(f"- {data['display']} (difficulty: {data.get('difficulty', 'unknown')})")
        
        prompt = f"""Identify prerequisite relationships between these {len(batch)} physics/science concepts.

Concepts:
{chr(10).join(concept_list)}

Domain: {domain}

For each concept, list which OTHER concepts from this list are prerequisites (must understand BEFORE).

Return JSON:
{{
  "prerequisites": [
    {{"concept": "concept_name", "requires": ["prereq1", "prereq2"], "confidence": 0.9}}
  ]
}}

Rules:
- Only high confidence (>0.7) relationships
- Be conservative - quality over quantity
- Max 2-3 prerequisites per concept

Return ONLY valid JSON."""

        try:
            result = call_json_chat(
                prompt,
                default={"prerequisites": []},
                model_hint=FAST_INGEST_MODEL,
                max_tokens=1000,
            )
        except Exception as e:
            logger.warning(f"Batch {batch_idx + 1} LLM call failed: {e}")
            continue
        
        # Parse results
        batch_concepts = {c.lower().strip(): c for c, _ in batch}
        
        for item in result.get("prerequisites", []):
            concept = item.get("concept", "")
            requires = item.get("requires", [])
            confidence = item.get("confidence", 0.5)
            
            if confidence < 0.7:
                continue
            
            concept_canonical = concept.lower().strip()
            if concept_canonical not in batch_concepts:
                continue
            
            for req in requires:
                req_canonical = req.lower().strip()
                if req_canonical in batch_concepts and req_canonical != concept_canonical:
                    all_prerequisites.append((req_canonical, concept_canonical, confidence))
        
        logger.info(f"  Batch {batch_idx + 1}/{num_batches}: {len(result.get('prerequisites', []))} relationships")
    
    # Deduplicate
    unique_prereqs = list(set(all_prerequisites))
    logger.info(f"LLM ({FAST_INGEST_MODEL}) extracted {len(unique_prereqs)} prerequisites total")
    return unique_prereqs


def store_chunks(conn, resource_id: str, chunks: List[Dict], embeddings: List[List[float]]) -> int:
    """Store chunks to Postgres."""
    from psycopg2.extras import execute_values, Json
    
    embed_version = os.getenv("EMBED_VERSION", "all-MiniLM-L6-v2-2025-09")
    
    try:
        with conn.cursor() as cur:
            # Delete existing chunks
            cur.execute("DELETE FROM chunk WHERE resource_id = %s::uuid", (resource_id,))
            
            # Prepare values
            values = []
            for i, chunk in enumerate(chunks):
                embedding = embeddings[i] if i < len(embeddings) else None
                vec_lit = None
                if embedding:
                    vec_lit = "[" + ",".join(f"{float(x):.6f}" for x in embedding) + "]"
                
                concepts = chunk.get("concepts", [])
                role = chunk.get("pedagogy_role", "explanation")
                difficulty = chunk.get("difficulty", "intermediate")
                
                # Format arrays
                def to_pg_array(lst):
                    if not lst:
                        return "{}"
                    escaped = [str(x).replace("\\", "\\\\").replace('"', '\\"') for x in lst]
                    return "{" + ",".join(f'"{x}"' for x in escaped) + "}"
                
                tags = Json({
                    "pedagogy_role": role,
                    "difficulty": difficulty,
                    "concepts": concepts,
                })
                
                values.append((
                    resource_id,
                    chunk["page_number"],
                    chunk.get("source_offset", 0),
                    chunk["full_text"],
                    role,
                    to_pg_array(concepts),
                    chunk.get("text_snippet", chunk["full_text"][:200]),
                    vec_lit,
                    embed_version,
                    tags,
                    None,  # section_title
                    None,  # section_number
                    "{}",  # section_path
                    0,     # section_level
                    chunk["page_number"],  # page_start
                    chunk["page_number"],  # page_end
                    chunk.get("token_count", len(chunk["full_text"].split())),
                    False,  # has_figure
                    False,  # has_equation
                    "{}",   # figure_labels
                    "{}",   # equation_labels
                    "",     # heading
                ))
            
            # Batch insert
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
            
            conn.commit()
            return len(values)
    except Exception as e:
        logger.error(f"Failed to store chunks: {e}")
        conn.rollback()
        return 0


def get_neo4j_driver():
    """Get Neo4j driver with fallback for host context."""
    from neo4j import GraphDatabase
    import socket
    
    uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
    password = os.getenv('NEO4J_PASSWORD', 'password')
    
    # Check if neo4j hostname is resolvable
    try:
        socket.gethostbyname('neo4j')
    except socket.error:
        uri = 'bolt://localhost:7687'
        logger.info("Neo4j hostname not resolvable, using localhost:7687")
    
    return GraphDatabase.driver(uri, auth=('neo4j', password))


def store_neo4j(resource_id: str, title: str, chunks: List[Dict], concepts: Dict[str, Dict], prerequisites: List[Tuple]) -> int:
    """Store concepts and relationships in Neo4j.
    
    Creates:
    - Resource node
    - Concept nodes
    - Chunk nodes
    - OCCURS_IN relationships (Concept -> Chunk)
    - PREREQUISITE_OF relationships (Concept -> Concept)
    """
    try:
        driver = get_neo4j_driver()
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        return 0
    
    created = 0
    
    try:
        with driver.session() as session:
            # Create Resource node
            session.run("""
                MERGE (r:Resource {id: $rid})
                SET r.title = $title, r.updated_at = datetime()
            """, rid=resource_id, title=title)
            
            # Create Concept nodes
            for canonical, data in concepts.items():
                session.run("""
                    MERGE (c:Concept {canonical_name: $canonical})
                    SET c.display_name = $display,
                        c.name_lower = $lower,
                        c.difficulty = $difficulty,
                        c.updated_at = datetime()
                """, 
                    canonical=canonical,
                    display=data['display'],
                    lower=canonical,
                    difficulty=data.get('difficulty', 'intermediate')
                )
                created += 1
            
            # Create Chunk nodes and OCCURS_IN relationships
            for i, chunk in enumerate(chunks):
                chunk_id = f"{resource_id}_{i}"
                
                session.run("""
                    MERGE (ch:Chunk {id: $chunk_id})
                    SET ch.resource_id = $rid,
                        ch.page_number = $page,
                        ch.pedagogy_role = $role,
                        ch.text_preview = $preview,
                        ch.updated_at = datetime()
                """,
                    chunk_id=chunk_id,
                    rid=resource_id,
                    page=chunk.get('page_number', 0),
                    role=chunk.get('pedagogy_role', 'explanation'),
                    preview=chunk.get('text_snippet', '')[:200]
                )
                
                # Link chunk to concepts (Chunk -> Concept direction for RAG queries)
                for concept in chunk.get('concepts', []):
                    canonical = concept.lower().strip()
                    session.run("""
                        MATCH (c:Concept {canonical_name: $canonical})
                        MATCH (ch:Chunk {id: $chunk_id})
                        MERGE (ch)-[r:MENTIONS]->(c)
                        SET r.resource_id = $rid, r.pedagogy_role = $role
                    """,
                        canonical=canonical,
                        chunk_id=chunk_id,
                        rid=resource_id,
                        role=chunk.get('pedagogy_role', 'explanation')
                    )
            
            # Create PREREQUISITE_OF relationships
            for prereq, concept, confidence in prerequisites:
                session.run("""
                    MATCH (pre:Concept {canonical_name: $prereq})
                    MATCH (con:Concept {canonical_name: $concept})
                    MERGE (pre)-[r:PREREQUISITE_OF]->(con)
                    SET r.confidence = $conf, r.source = 'llm'
                """,
                    prereq=prereq,
                    concept=concept,
                    conf=confidence
                )
            
            # Link Resource to Concepts
            session.run("""
                MATCH (r:Resource {id: $rid})
                MATCH (c:Concept)-[:OCCURS_IN]->(:Chunk {resource_id: $rid})
                MERGE (r)-[:CONTAINS]->(c)
            """, rid=resource_id)
            
            logger.info(f"Neo4j: Created {created} concepts, {len(chunks)} chunks, {len(prerequisites)} prerequisites")
            
    except Exception as e:
        logger.error(f"Neo4j store failed: {e}")
        return 0
    finally:
        driver.close()
    
    return created


def fast_ingest(file_path: str, title: Optional[str] = None) -> FastIngestResult:
    """Run fast ingestion on a document.
    
    Args:
        file_path: Path to the document (PDF)
        title: Optional title (defaults to filename)
    
    Returns:
        FastIngestResult with resource_id and stats
    """
    # Load .env first
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    t0 = time.time()
    
    # Generate resource ID
    resource_id = str(uuid.uuid4())
    title = title or Path(file_path).stem
    
    result = FastIngestResult(resource_id=resource_id, title=title)
    
    # Validate file
    if not os.path.exists(file_path):
        result.errors.append(f"File not found: {file_path}")
        return result
    
    logger.info(f"Starting fast ingestion: {file_path}")
    logger.info(f"Resource ID: {resource_id}")
    logger.info(f"Using model: {FAST_INGEST_MODEL}")
    
    # Connect to database
    try:
        conn = get_db_connection()
        logger.info("Database connected")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        result.errors.append(f"Database connection failed: {e}")
        return result
    
    try:
        # Step 1: Create resource
        logger.info("Step 1: Creating resource entry...")
        try:
            if not create_resource(conn, resource_id, title, file_path):
                result.errors.append("Failed to create resource")
                return result
        except Exception as e:
            logger.exception(f"Create resource failed: {e}")
            result.errors.append(f"Failed to create resource: {e}")
            return result
        
        # Step 2: Parse PDF
        logger.info("Step 2: Parsing document...")
        pages = parse_pdf(file_path)
        if not pages:
            result.errors.append("Failed to parse document")
            return result
        logger.info(f"  Parsed {len(pages)} pages")
        
        # Step 3: Chunk text
        logger.info("Step 3: Chunking text...")
        chunks = chunk_text(pages)
        logger.info(f"  Created {len(chunks)} chunks")
        
        # Step 4: Extract metadata (heuristic)
        logger.info("Step 4: Extracting metadata...")
        all_concepts = {}
        
        # First pass: extract all concepts
        for chunk in chunks:
            concepts, role, difficulty = extract_concepts_heuristic(chunk["full_text"])
            chunk["raw_concepts"] = concepts
            chunk["pedagogy_role"] = role
            chunk["difficulty"] = difficulty
            
            # Track concepts
            for c in concepts:
                canonical = c.lower().strip()
                if canonical not in all_concepts:
                    all_concepts[canonical] = {
                        "display": c,
                        "difficulty": difficulty,
                        "first_page": chunk["page_number"],
                        "frequency": 0,
                    }
                all_concepts[canonical]["frequency"] += 1
        
        # Filter: keep only concepts that appear 2+ times (reduces noise)
        MIN_FREQUENCY = 2
        filtered_concepts = {
            k: v for k, v in all_concepts.items() 
            if v["frequency"] >= MIN_FREQUENCY
        }
        
        # Second pass: update chunks with filtered concepts only
        for chunk in chunks:
            chunk["concepts"] = [
                c for c in chunk.get("raw_concepts", [])
                if c.lower().strip() in filtered_concepts
            ]
            del chunk["raw_concepts"]
        
        logger.info(f"  Raw concepts: {len(all_concepts)}, After filtering (freq≥{MIN_FREQUENCY}): {len(filtered_concepts)}")
        all_concepts = filtered_concepts
        result.concepts_extracted = len(all_concepts)
        
        # Step 5: Compute embeddings
        logger.info("Step 5: Computing embeddings...")
        embeddings = compute_embeddings(chunks)
        logger.info(f"  Computed {len(embeddings)} embeddings")
        
        # Step 6: Store to Postgres
        logger.info("Step 6: Storing chunks to database...")
        stored = store_chunks(conn, resource_id, chunks, embeddings)
        result.chunks_created = stored
        logger.info(f"  Stored {stored} chunks")
        
        # Step 7: Build prerequisite graph with LLM
        logger.info("Step 7: Building knowledge graph with LLM...")
        prerequisites = build_prerequisites_llm(all_concepts)
        result.prerequisites_inferred = len(prerequisites)
        
        # Compute related concepts from co-occurrence
        from collections import defaultdict
        cooccurrence = defaultdict(int)
        for chunk in chunks:
            concepts = [c.lower().strip() for c in chunk.get("concepts", [])]
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i+1:]:
                    if c1 < c2:
                        cooccurrence[(c1, c2)] += 1
                    else:
                        cooccurrence[(c2, c1)] += 1
        
        min_cooccur = max(3, len(chunks) // 6)
        related = [(c1, c2) for (c1, c2), count in cooccurrence.items() if count >= min_cooccur]
        result.related_concepts = len(related)
        
        logger.info(f"  Prerequisites: {len(prerequisites)}, Related: {len(related)}")
        
        # Step 8: Store to Neo4j (knowledge graph)
        logger.info("Step 8: Storing to Neo4j...")
        neo4j_concepts = store_neo4j(resource_id, title, chunks, all_concepts, prerequisites)
        logger.info(f"  Neo4j: {neo4j_concepts} concepts stored")
        
    finally:
        conn.close()
    
    result.elapsed_ms = int((time.time() - t0) * 1000)
    
    logger.info(f"\n{'='*60}")
    logger.info("INGESTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Resource ID: {resource_id}")
    logger.info(f"Title: {title}")
    logger.info(f"Chunks: {result.chunks_created}")
    logger.info(f"Concepts: {result.concepts_extracted}")
    logger.info(f"Prerequisites: {result.prerequisites_inferred}")
    logger.info(f"Related: {result.related_concepts}")
    logger.info(f"Elapsed: {result.elapsed_ms}ms")
    
    if result.errors:
        logger.error(f"Errors: {result.errors}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Fast ingest a document")
    parser.add_argument("file_path", help="Path to the document (PDF)")
    parser.add_argument("--title", "-t", help="Document title (defaults to filename)")
    
    args = parser.parse_args()
    
    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    try:
        result = fast_ingest(args.file_path, args.title)
    except Exception as e:
        logger.exception(f"Ingestion failed with exception: {e}")
        print(f"\n{json.dumps({'error': str(e)}, indent=2)}")
        return 1
    
    # Print result as JSON for easy parsing
    print(f"\n{json.dumps({'resource_id': result.resource_id, 'chunks': result.chunks_created, 'concepts': result.concepts_extracted}, indent=2)}")
    
    return 0 if not result.errors else 1


if __name__ == "__main__":
    sys.exit(main())
