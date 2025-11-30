from typing import List, Dict, Any, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


def _get_db_dsn() -> str:
    """Get PostgreSQL connection string with local fallback."""
    import socket
    
    # Check if 'postgres' hostname is resolvable (Docker vs local)
    postgres_resolvable = True
    try:
        socket.gethostbyname("postgres")
    except socket.error:
        postgres_resolvable = False
    
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        # If DATABASE_URL uses 'postgres' host but it's not resolvable, fix it
        if not postgres_resolvable and "@postgres:" in dsn:
            dsn = dsn.replace("@postgres:", "@localhost:").replace(":5432/", ":5433/")
        return dsn
    
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "app")
    
    # Dev fallback: if host is 'postgres' and not resolvable, use localhost:5433
    if host == "postgres" and not postgres_resolvable:
        host = "localhost"
        port = "5433"
    
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


class RAGTools:
    """
    RAG (Retrieval-Augmented Generation) tools for knowledge retrieval.
    
    Implements standalone vector search without depending on agents.retrieval.
    """
    
    # Class-level state for lazy initialization
    _embed_fn = None
    _embed_error = None
    _embed_checked = False
    
    _managed_driver = None
    _managed_driver_checked = False
    
    _db_available = None
    _db_error = None
    
    def __init__(self):
        self._init_embed()
        self._init_managed_driver()
        self._init_db()
    
    def _init_embed(self):
        """Lazy init for embedding function."""
        if RAGTools._embed_checked:
            return
        RAGTools._embed_checked = True
        
        try:
            from ingestion.embed import embed_text
            RAGTools._embed_fn = embed_text
        except ImportError as e:
            RAGTools._embed_error = f"embed import: {e}"
            logger.debug(f"Embedding unavailable: {e}")
        except Exception as e:
            RAGTools._embed_error = f"embed init: {e}"
            logger.debug(f"Embedding init failed: {e}")
    
    def _init_managed_driver(self):
        """Lazy init for Neo4j driver."""
        if RAGTools._managed_driver_checked:
            return
        RAGTools._managed_driver_checked = True
        
        try:
            from kg_pipeline.base import managed_driver
            RAGTools._managed_driver = managed_driver
        except ImportError as e:
            logger.debug(f"Graph driver unavailable: {e}")
        except Exception as e:
            logger.debug(f"Graph driver init failed: {e}")
    
    def _init_db(self):
        """Check if PostgreSQL is available (re-checks each time)."""
        # Always re-check DB availability since connection status can change
        try:
            import psycopg2
            conn = psycopg2.connect(_get_db_dsn())
            conn.close()
            RAGTools._db_available = True
            RAGTools._db_error = None
        except Exception as e:
            RAGTools._db_available = False
            RAGTools._db_error = str(e)
            logger.debug(f"Database unavailable: {e}")
    
    @property
    def _vector_available(self) -> bool:
        return RAGTools._embed_fn is not None and RAGTools._db_available
    
    @property
    def _graph_available(self) -> bool:
        return RAGTools._managed_driver is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get RAG availability status."""
        if not self._vector_available:
            error = RAGTools._embed_error or RAGTools._db_error or "Unknown"
        else:
            error = None
        return {
            "vector_search": self._vector_available,
            "vector_error": error,
            "graph_search": self._graph_available,
        }
    
    def is_available(self) -> bool:
        """Check if any RAG capability is available."""
        return self._vector_available or self._graph_available
    
    def _simple_vector_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Standalone vector search using pgvector.
        
        Falls back to text search if embedding fails.
        """
        if not RAGTools._db_available:
            return []
        
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError:
            return []
        
        try:
            conn = psycopg2.connect(_get_db_dsn())
            
            # Try vector search first
            if RAGTools._embed_fn:
                try:
                    qvec = RAGTools._embed_fn(query)
                    qvec_lit = "[" + ",".join(f"{float(x):.6f}" for x in qvec) + "]"
                    
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute("""
                            SELECT id::text, resource_id::text, page_number,
                                   LEFT(full_text, 800) AS snippet,
                                   1 - (embedding <=> %s::vector) AS score
                            FROM chunk
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s
                        """, (qvec_lit, qvec_lit, limit))
                        results = [dict(r) for r in cur.fetchall()]
                        conn.close()
                        return results
                except Exception as e:
                    logger.debug(f"Vector search failed, trying text: {e}")
            
            # Fallback to text search
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id::text, resource_id::text, page_number,
                           LEFT(full_text, 800) AS snippet,
                           ts_rank_cd(search_tsv, plainto_tsquery('english', %s)) AS score
                    FROM chunk
                    WHERE search_tsv @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                """, (query, query, limit))
                results = [dict(r) for r in cur.fetchall()]
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_concepts_for_resources(self, resource_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch all concepts linked to the given resource IDs via Chunks.
        Returns list of dicts with concept 'id' (canonical) and 'name' (display).
        """
        if not RAGTools._managed_driver:
            return []
            
        try:
            with RAGTools._managed_driver() as driver:
                if not driver:
                    return []
                    
                query = """
                MATCH (c:Chunk)-[r]->(con:Concept)
                WHERE r.resource_id IN $resource_ids
                RETURN con.canonical_name as id, con.display_name as name, min(c.id) as first_chunk_id
                """
                
                with driver.session() as session:
                    result = session.run(query, resource_ids=resource_ids)
                    return [record.data() for record in result]
        except Exception as e:
            logger.error(f"get_concepts_for_resources failed: {e}")
            return []

    def get_concept_dependencies(self, concept_ids: List[str]) -> List[Dict[str, str]]:
        """
        Fetch PREREQUISITE_OF relationships where both start and end concepts are in the provided list.
        Returns list of {'from': concept_id, 'to': concept_id}.
        """
        if not RAGTools._managed_driver:
            return []
            
        try:
            with RAGTools._managed_driver() as driver:
                if not driver:
                    return []
                    
                query = """
                MATCH (c1:Concept)-[r:PREREQUISITE_OF]->(c2:Concept)
                WHERE c1.canonical_name IN $concept_ids AND c2.canonical_name IN $concept_ids
                RETURN c1.canonical_name as from, c2.canonical_name as to
                """
                
                with driver.session() as session:
                    result = session.run(query, concept_ids=concept_ids)
                    return [record.data() for record in result]
        except Exception as e:
            logger.error(f"get_concept_dependencies failed: {e}")
            return []

    def get_concept_details(self, concept_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch details (name, etc.) for a specific list of concept IDs.
        """
        if not RAGTools._managed_driver:
            return []
            
        try:
            with RAGTools._managed_driver() as driver:
                if not driver:
                    return []
                    
                query = """
                MATCH (con:Concept)
                WHERE con.canonical_name IN $concept_ids
                RETURN con.canonical_name as id, con.display_name as name
                """
                
                with driver.session() as session:
                    result = session.run(query, concept_ids=concept_ids)
                    return [record.data() for record in result]
        except Exception as e:
            logger.error(f"get_concept_details failed: {e}")
            return []

    def search_graph_data(self, concept: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Raw graph search returning list of dicts."""
        if not RAGTools._managed_driver:
            return []

        try:
            logger.debug(f"Searching graph for concept: {concept}")
            with RAGTools._managed_driver() as driver:
                if not driver:
                    logger.warning("Graph driver unavailable")
                    return []
                
                query = """
                MATCH (c:Concept)-[r]-(related:Concept)
                WHERE c.display_name =~ '(?i)' + $concept
                RETURN related.display_name AS name, type(r) AS relationship, r.weight AS weight
                ORDER BY weight DESC
                LIMIT $limit
                """
                
                with driver.session() as session:
                    result = session.run(query, concept=concept, limit=limit)
                    data = [record.data() for record in result]
                    logger.info(f"Graph search found {len(data)} records for concept: {concept}")
                    return data
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    def search_graph(self, concept: str, limit: int = 5) -> str:
        """Formatted graph search string."""
        records = self.search_graph_data(concept, limit)
        if not records:
            return ""
        lines = []
        for r in records:
            lines.append(f"- {r.get('name')} ({r.get('relationship')})")
        return "\n".join(lines)

    def search_vector_chunks(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Raw vector search returning list of chunks."""
        if not self._vector_available:
            return []
        try:
            logger.debug(f"Searching vector chunks for query: {query}")
            results = self._simple_vector_search(query, limit) or []
            logger.debug(f"Vector search found {len(results)} chunks for query: {query}")
            return results
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []

    def search_context(self, query: str, limit: int = 3) -> str:
        """Formatted vector search string."""
        chunks = self.search_vector_chunks(query, limit)
        if not chunks:
            return ""
        
        context_lines = []
        for i, chunk in enumerate(chunks):
            snippet = chunk.get("snippet", "") or chunk.get("text", "")
            context_lines.append(f"[{i+1}] {snippet}")
        
        return "\n\n".join(context_lines)
    
    def search_multi_query(self, queries: List[str], limit_per_query: int = 2) -> str:
        """
        Search with multiple short queries and deduplicate results.
        
        Each query should be 2-3 words max for optimal retrieval.
        
        Args:
            queries: List of short search queries (2-3 words each)
            limit_per_query: Max results per query
            
        Returns:
            Formatted context string with deduplicated results
        """
        if not queries:
            return ""
        
        all_chunks = []
        seen_texts = set()
        
        for query in queries[:3]:  # Max 3 queries
            # Clean and validate query
            query = query.strip()
            if not query or len(query.split()) > 4:
                logger.warning(f"Skipping invalid RAG query: '{query}' (should be 2-3 words)")
                continue
            
            logger.info(f"RAG multi-query search: '{query}'")
            chunks = self.search_vector_chunks(query, limit_per_query)
            
            for chunk in chunks:
                snippet = chunk.get("snippet", "") or chunk.get("text", "")
                # Deduplicate by text prefix
                text_key = snippet[:100] if snippet else ""
                if text_key and text_key not in seen_texts:
                    seen_texts.add(text_key)
                    all_chunks.append(chunk)
        
        if not all_chunks:
            return ""
        
        # Format results
        context_lines = []
        for i, chunk in enumerate(all_chunks[:5]):  # Max 5 total results
            snippet = chunk.get("snippet", "") or chunk.get("text", "")
            context_lines.append(f"[{i+1}] {snippet}")
        
        logger.info(f"RAG multi-query returned {len(context_lines)} unique chunks")
        return "\n\n".join(context_lines)

    def get_planning_context(self, concept: str) -> Tuple[str, Dict[str, Any]]:
        """
        Get combined context from Vector DB and Knowledge Graph for planning.
        Returns (formatted_string, raw_data_dict).
        
        When RAG is unavailable, provides a minimal concept-based context.
        """
        vector_chunks = self.search_vector_chunks(concept, limit=3)
        graph_records = self.search_graph_data(concept, limit=5)
        
        # Format Vector Context
        vector_str = ""
        if vector_chunks:
            lines = []
            for i, chunk in enumerate(vector_chunks):
                snippet = chunk.get("snippet", "") or chunk.get("text", "")
                lines.append(f"[{i+1}] {snippet}")
            vector_str = "\n\n".join(lines)
        
        # If no RAG available, provide minimal context based on concept name
        if not vector_chunks and not graph_records:
            fallback_context = f"Topic: {concept}\n\nNote: Knowledge base unavailable. Use general knowledge about {concept}."
            return fallback_context, {"fallback": True, "concept": concept}

        # Format Graph Context
        graph_str = ""
        if graph_records:
            lines = []
            for r in graph_records:
                lines.append(f"- {r.get('name')} ({r.get('relationship')})")
            graph_str = "\n".join(lines)
        
        sections = []
        if vector_str:
            sections.append(f"### Relevant Content:\n{vector_str}")
        if graph_str:
            sections.append(f"### Related Concepts (Knowledge Graph):\n{graph_str}")
            
        combined_context = "\n\n".join(sections)
        
        raw_data = {
            "vector_chunks": vector_chunks,
            "graph_records": graph_records
        }
        
        return combined_context, raw_data
