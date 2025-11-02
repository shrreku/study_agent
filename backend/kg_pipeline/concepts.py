"""Concept merge helpers for the knowledge graph."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

from .base import canonicalize_concept, count_occurrences, managed_driver


def merge_concepts_in_neo4j(
    concepts: Iterable[str],
    chunk_id: str,
    snippet: str,
    resource_id: str,
    chunk_meta: Dict[str, Any] | None = None,
) -> None:
    chunk_meta = chunk_meta or {}
    concepts = concepts or []

    cleaned: List[Tuple[str, str]] = []
    counts: List[int] = []
    full_text = chunk_meta.get("full_text") or ""
    for name in concepts:
        canonical, display = canonicalize_concept(name)
        if not canonical:
            continue
        cleaned.append((canonical, display))
        counts.append(max(1, count_occurrences(full_text, display) or count_occurrences(full_text, canonical)))

    if not cleaned:
        return

    total = sum(counts) or max(1, len(cleaned))
    concept_stats: List[Tuple[str, str, int, float]] = []
    for (canonical, display), count in zip(cleaned, counts):
        salience = min(1.0, count / total) if total else 0.0
        concept_stats.append((canonical, display, count, salience))

    section_path = chunk_meta.get("section_path") or []
    if isinstance(section_path, str):
        section_path_value = section_path
    else:
        section_path_value = [str(s) for s in section_path]

    page_number = chunk_meta.get("page_number")
    section_title = chunk_meta.get("section_title")
    section_number = chunk_meta.get("section_number")
    section_level = chunk_meta.get("section_level")
    chunk_type = chunk_meta.get("chunk_type")

    snippet_value = (snippet or "")[:300]

    with managed_driver() as driver:
        if driver is None:
            return

        def _tx(tx):
            for canonical, display, freq, salience in concept_stats:
                tx.run(
                    """
                    MERGE (c:Concept {canonical_name: $canonical})
                    ON CREATE SET c.display_name = $display, c.name_lower = $canonical, c.created_at = datetime()
                    SET c.display_name = coalesce(c.display_name, $display),
                        c.last_seen = datetime(),
                        c.name_lower = $canonical
                    MERGE (r:Resource {id: $resid})
                    MERGE (ch:Chunk {id: $chunk_id})
                    SET ch.snippet = $snippet,
                        ch.page_number = $page_number,
                        ch.section_path = $section_path,
                        ch.section_title = $section_title,
                        ch.section_number = $section_number,
                        ch.section_level = $section_level,
                        ch.chunk_type = $chunk_type,
                        ch.last_seen = datetime(),
                        ch.resource_id = $resid
                    MERGE (c)-[rel:OCCURS_IN]->(ch)
                    SET rel.frequency = $frequency,
                        rel.salience = $salience,
                        rel.page_number = $page_number,
                        rel.section_path = $section_path,
                        rel.section_title = $section_title,
                        rel.section_number = $section_number,
                        rel.section_level = $section_level,
                        rel.chunk_type = $chunk_type,
                        rel.snippet = $snippet,
                        rel.last_seen = datetime(),
                        rel.resource_id = $resid
                    MERGE (ch)-[:PART_OF]->(r)
                    """,
                    canonical=canonical,
                    display=display,
                    chunk_id=chunk_id,
                    snippet=snippet_value,
                    resid=resource_id,
                    frequency=freq,
                    salience=float(salience),
                    page_number=page_number,
                    section_path=section_path_value,
                    section_title=section_title,
                    section_number=section_number,
                    section_level=section_level,
                    chunk_type=chunk_type,
                )

        try:
            with driver.session() as session:
                session.execute_write(_tx)
        except Exception:
            logging.exception("neo4j_merge_failed")
