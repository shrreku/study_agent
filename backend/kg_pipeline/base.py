"""Shared helpers for Neo4j knowledge graph utilities."""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from contextlib import contextmanager
from typing import Any, Iterator, Tuple


_CONSTRAINTS_ENSURED = False


def _strip_diacritics(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _canonicalize(name: str) -> Tuple[str, str]:
    raw = (name or "").strip()
    if not raw:
        return "", ""
    display = raw
    lowered = _strip_diacritics(raw.lower())
    lowered = lowered.replace("-", " ")
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered, display


def canonicalize_concept(name: str) -> Tuple[str, str]:
    """Public wrapper for canonicalization."""
    return _canonicalize(name)


def count_occurrences(text: str, term: str) -> int:
    if not text or not term:
        return 0
    pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
    return len(pattern.findall(text))


def _ensure_constraints(driver: Any) -> None:
    global _CONSTRAINTS_ENSURED
    if _CONSTRAINTS_ENSURED:
        return

    try:
        def _tx(tx):
            tx.run(
                """
                CREATE CONSTRAINT concept_canonical_unique IF NOT EXISTS
                FOR (c:Concept) REQUIRE c.canonical_name IS UNIQUE
                """
            )
            tx.run(
                """
                CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
                FOR (ch:Chunk) REQUIRE ch.id IS UNIQUE
                """
            )
            tx.run(
                """
                CREATE CONSTRAINT resource_id_unique IF NOT EXISTS
                FOR (r:Resource) REQUIRE r.id IS UNIQUE
                """
            )
            tx.run(
                """
                CREATE INDEX concept_name_lower_idx IF NOT EXISTS
                FOR (c:Concept) ON (c.name_lower)
                """
            )

        with driver.session() as session:
            session.execute_write(_tx)
        _CONSTRAINTS_ENSURED = True
    except Exception:
        logging.exception("neo4j_constraint_creation_failed")


@contextmanager
def managed_driver() -> Iterator[Any]:
    """Yield a Neo4j driver with constraints ensured, closing it afterward."""
    driver = None
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception:
        logging.exception("neo4j_driver_import_failed")
        yield None
        return

    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        _ensure_constraints(driver)
    except Exception:
        logging.exception("neo4j_driver_init_failed")
        if driver is not None:
            try:
                driver.close()
            except Exception:
                logging.exception("neo4j_driver_close_failed")
        yield None
        return

    try:
        yield driver
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                logging.exception("neo4j_driver_close_failed")


def ensure_neo4j_constraints() -> None:
    """Public helper to ensure required constraints exist."""
    with managed_driver():
        # managed_driver ensures constraints upon acquisition
        pass
