"""
Knowledge helper functions for the tutor agent.

Provides functions to fetch mastery maps and concept prerequisite chains.
"""

from typing import Dict, Any, List, Optional


def fetch_mastery_map(cursor, user_id: str) -> Dict[str, Any]:
    """
    Fetch user's concept mastery map from the database.
    
    Args:
        cursor: Database cursor
        user_id: User ID
        
    Returns:
        Dict mapping concept names to mastery info dicts
    """
    cursor.execute(
        """
        SELECT concept AS concept_id, mastery, last_seen AS last_updated
        FROM user_concept_mastery
        WHERE user_id = %s::uuid
        """,
        (user_id,)
    )
    rows = cursor.fetchall()
    
    mastery_map = {}
    for row in rows:
        concept_id = row[0] if isinstance(row, tuple) else row.get("concept_id")
        mastery = row[1] if isinstance(row, tuple) else row.get("mastery")
        
        if concept_id:
            mastery_map[concept_id] = {
                "mastery": mastery,
                "last_updated": row[2] if isinstance(row, tuple) else row.get("last_updated")
            }
    
    return mastery_map


def fetch_prereq_chain(concepts: List[str]) -> List[str]:
    """
    Fetch prerequisite chain for a list of concepts from Neo4j.
    
    This performs a topological sort based on prerequisite relationships.
    For now, returns the input concepts as-is (stub implementation).
    
    Args:
        concepts: List of concept names
        
    Returns:
        Ordered list of concepts based on prerequisite dependencies
    """
    # TODO: Implement Neo4j query to fetch prerequisite chain
    # For now, return concepts as-is
    return concepts or []
