from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from psycopg2.extras import Json

logger = logging.getLogger(__name__)


def fetch_mastery_map(cursor, user_id: str) -> Dict[str, Dict[str, Any]]:
    cursor.execute(
        """
        SELECT concept, mastery, attempts, correct
        FROM user_concept_mastery
        WHERE user_id = %s::uuid
        """,
        (user_id,),
    )
    rows = cursor.fetchall()
    mastery_map: Dict[str, Dict[str, Any]] = {}
    for r in rows or []:
        concept = r[0]
        mastery_map[concept] = {
            "mastery": float(r[1]) if r[1] is not None else None,
            "attempts": int(r[2]) if r[2] is not None else 0,
            "correct": int(r[3]) if r[3] is not None else 0,
        }
    return mastery_map


def fetch_prereq_chain(concepts: List[str], max_depth: int = 4) -> List[str]:
    if not concepts:
        return []
    try:
        from neo4j import GraphDatabase
    except Exception:
        return list(dict.fromkeys(concepts))

    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
    order: List[str] = []
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
    except Exception:
        logger.exception("tutor_neo4j_driver_failed")
        return list(dict.fromkeys(concepts))

    def _fetch(tx, name: str) -> List[str]:
        try:
            q = (
                "MATCH (c:Concept {name:$name})\n"
                "CALL apoc.path.subgraphNodes(c, {relationshipFilter:'<PREREQUISITE_OF', maxLevel:$depth}) YIELD node\n"
                "RETURN DISTINCT node.name AS name"
            )
            res = tx.run(q, name=name, depth=max_depth)
        except Exception:
            q2 = (
                "MATCH (p:Concept)-[:PREREQUISITE_OF*0..$depth]->(c:Concept {name:$name})\n"
                "RETURN DISTINCT p.name AS name"
            )
            res = tx.run(q2, name=name, depth=max_depth)
        return [row["name"] for row in res]

    try:
        with driver.session() as session:
            seen: List[str] = []
            for concept in concepts:
                try:
                    names = session.execute_read(_fetch, concept)
                except Exception:
                    names = []
                if concept not in names:
                    names.append(concept)
                for name in names:
                    if name not in seen:
                        seen.append(name)
            order = seen
    except Exception:
        logger.exception("tutor_prereq_fetch_failed")
        order = list(dict.fromkeys(concepts))
    finally:
        try:
            driver.close()
        except Exception:
            pass
    return order


def record_cold_start(cursor, session_id: str, concept: str) -> None:
    cursor.execute(
        """
        INSERT INTO tutor_event (session_id, event_type, payload)
        VALUES (%s::uuid, %s, %s)
        """,
        (
            session_id,
            "cold_start_triggered",
            Json({"concept": concept}),
        ),
    )
