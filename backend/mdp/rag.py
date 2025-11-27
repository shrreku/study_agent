from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from agents.retrieval import hybrid_search
except ImportError:
    hybrid_search = None

try:
    from kg_pipeline.base import managed_driver
except ImportError:
    managed_driver = None

class RAGTools:
    def __init__(self):
        pass

    def get_concepts_for_resources(self, resource_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch all concepts linked to the given resource IDs via Chunks.
        Returns list of dicts with concept 'id' (canonical) and 'name' (display).
        """
        if not managed_driver:
            logger.warning("RAG tools unavailable: managed_driver import failed")
            return []
            
        try:
            with managed_driver() as driver:
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
        if not managed_driver:
            return []
            
        try:
            with managed_driver() as driver:
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
        if not managed_driver:
            return []
            
        try:
            with managed_driver() as driver:
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
        if not managed_driver:
            logger.warning("RAG tools unavailable: managed_driver import failed")
            return []

        try:
            logger.info(f"Searching graph for concept: {concept}")
            with managed_driver() as driver:
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
        if not hybrid_search:
            logger.warning("RAG tools unavailable: hybrid_search import failed")
            return []
        try:
            logger.info(f"Searching vector chunks for query: {query}")
            results = hybrid_search(query, limit) or []
            logger.info(f"Vector search found {len(results)} chunks for query: {query}")
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

    def get_planning_context(self, concept: str) -> Tuple[str, Dict[str, Any]]:
        """
        Get combined context from Vector DB and Knowledge Graph for planning.
        Returns (formatted_string, raw_data_dict).
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
