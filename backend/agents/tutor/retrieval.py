"""
Retrieval helper functions for the tutor agent.

Provides functions to retrieve relevant chunks for concepts.
"""

from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from core.db import get_db_conn


def retrieve_chunks(
    concept: str,
    resource_id: str,
    pedagogy_roles: Optional[List[str]] = None,
    k: int = 6
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks for a concept from a specific resource.
    
    Args:
        concept: Concept name to search for
        resource_id: Resource ID to search within
        pedagogy_roles: Optional list of pedagogy roles to filter by
        k: Maximum number of chunks to return
        
    Returns:
        List of chunk dictionaries with id, snippet, score, etc.
    """
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Simple implementation: fetch chunks that mention the concept
            if pedagogy_roles:
                cur.execute(
                    """
                    SELECT 
                        id::text,
                        full_text as snippet,
                        page_number,
                        resource_id::text,
                        tags
                    FROM chunk
                    WHERE resource_id = %s::uuid
                      AND %s = ANY(concepts)
                      AND (tags->>'pedagogy_role') = ANY(%s)
                    LIMIT %s
                    """,
                    (resource_id, concept, pedagogy_roles, k)
                )
            else:
                cur.execute(
                    """
                    SELECT 
                        id::text,
                        full_text as snippet,
                        page_number,
                        resource_id::text,
                        tags
                    FROM chunk
                    WHERE resource_id = %s::uuid
                      AND %s = ANY(concepts)
                    LIMIT %s
                    """,
                    (resource_id, concept, k)
                )
            
            rows = cur.fetchall() or []
            
            chunks = []
            for idx, row in enumerate(rows):
                chunks.append({
                    "id": row.get("id"),
                    "snippet": row.get("snippet", ""),
                    "page_number": row.get("page_number"),
                    "resource_id": row.get("resource_id"),
                    "score": 1.0 - (idx * 0.1),  # Simple relevance scoring
                    "tags": row.get("tags", {})
                })
            
            return chunks
    finally:
        conn.close()
