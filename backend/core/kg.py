"""Knowledge Graph utilities (Neo4j merge helpers)."""
from __future__ import annotations
import os
import logging
from typing import List


def merge_concepts_in_neo4j(concepts: List[str], chunk_id: str, snippet: str, resource_id: str) -> None:
    # Lazy import to avoid import error when dependency not yet installed in image
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception:
        logging.exception("neo4j_driver_import_failed")
        return
    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))

        def _tx(tx):
            for cname in concepts:
                tx.run(
                    """
                    MERGE (c:Concept {name: $cname})
                    MERGE (ch:Chunk {id: $chunk_id})
                    SET ch.snippet = $snippet
                    MERGE (r:Resource {id: $resid})
                    MERGE (c)-[:EXPLAINED_BY]->(ch)
                    MERGE (ch)-[:PART_OF]->(r)
                    """,
                    cname=cname,
                    chunk_id=chunk_id,
                    snippet=(snippet or "")[:300],
                    resid=resource_id,
                )

        with driver.session() as session:
            session.execute_write(_tx)
    except Exception:
        logging.exception("neo4j_merge_failed")
