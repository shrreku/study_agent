"""Relationship helpers for the knowledge graph."""
from __future__ import annotations

import logging
from typing import Iterable, Tuple

from .base import canonicalize_concept, managed_driver


def merge_related_concepts(
    pairs: Iterable[Tuple[str, str] | Tuple[str, str, float]],
    default_weight: float = 1.0,
    method: str = "co_occurrence",
    evidence_chunk_id: str | None = None,
) -> None:
    cleaned_pairs: list[tuple[str, str, str, str, float]] = []
    for entry in pairs or []:
        if len(entry) >= 3:
            a, b, pair_weight = entry[:3]
        else:
            a, b = entry[:2]  # type: ignore[index]
            pair_weight = default_weight
        ca, da = canonicalize_concept(a)
        cb, db = canonicalize_concept(b)
        if not ca or not cb or ca == cb:
            continue
        cleaned_pairs.append((ca, cb, da, db, float(pair_weight)))

    if not cleaned_pairs:
        return

    with managed_driver() as driver:
        if driver is None:
            return

        def _tx(tx):
            for ca, cb, da, db, pair_weight in cleaned_pairs:
                tx.run(
                    """
                    MERGE (a:Concept {canonical_name: $ca})
                    ON CREATE SET a.display_name = $da, a.name_lower = $ca, a.created_at = datetime()
                    SET a.display_name = coalesce(a.display_name, $da), a.last_seen = datetime(), a.name_lower = $ca
                    MERGE (b:Concept {canonical_name: $cb})
                    ON CREATE SET b.display_name = $db, b.name_lower = $cb, b.created_at = datetime()
                    SET b.display_name = coalesce(b.display_name, $db), b.last_seen = datetime(), b.name_lower = $cb
                    MERGE (a)-[rel:RELATED_TO]-(b)
                    SET rel.method = $method,
                        rel.last_seen = datetime(),
                        rel.evidence_chunk_id = coalesce(rel.evidence_chunk_id, $evidence_chunk_id),
                        rel.evidence_chunk_ids = CASE
                            WHEN $evidence_chunk_id IS NULL THEN rel.evidence_chunk_ids
                            WHEN rel.evidence_chunk_ids IS NULL THEN [$evidence_chunk_id]
                            WHEN $evidence_chunk_id IN coalesce(rel.evidence_chunk_ids, []) THEN rel.evidence_chunk_ids
                            ELSE rel.evidence_chunk_ids + $evidence_chunk_id
                        END,
                        rel.weight = CASE
                            WHEN $evidence_chunk_id IS NULL THEN coalesce(rel.weight, 0) + $weight
                            WHEN rel.evidence_chunk_ids IS NULL THEN coalesce(rel.weight, 0) + $weight
                            WHEN $evidence_chunk_id IN coalesce(rel.evidence_chunk_ids, []) THEN rel.weight
                            ELSE coalesce(rel.weight, 0) + $weight
                        END
                    """,
                    ca=ca,
                    cb=cb,
                    da=da,
                    db=db,
                    weight=pair_weight,
                    method=method,
                    evidence_chunk_id=evidence_chunk_id,
                )

        try:
            with driver.session() as session:
                session.execute_write(_tx)
        except Exception:
            logging.exception("neo4j_merge_related_failed")


def merge_prerequisite_edge(
    prereq: str,
    target: str,
    confidence: float,
    evidence_chunk_id: str | None = None,
    method: str | None = None,
    evidence_sentences: Iterable[str] | None = None,
    evidence_confidence: float | None = None,
) -> None:
    cp, dp = canonicalize_concept(prereq)
    ct, dt = canonicalize_concept(target)
    if not cp or not ct or cp == ct:
        return

    with managed_driver() as driver:
        if driver is None:
            return

        def _tx(tx):
            tx.run(
                """
                MERGE (p:Concept {canonical_name: $cp})
                ON CREATE SET p.display_name = $dp, p.name_lower = $cp, p.created_at = datetime()
                SET p.display_name = coalesce(p.display_name, $dp), p.last_seen = datetime(), p.name_lower = $cp
                MERGE (t:Concept {canonical_name: $ct})
                ON CREATE SET t.display_name = $dt, t.name_lower = $ct, t.created_at = datetime()
                SET t.display_name = coalesce(t.display_name, $dt), t.last_seen = datetime(), t.name_lower = $ct
                MERGE (p)-[rel:PREREQUISITE_OF]->(t)
                SET rel.confidence = $confidence,
                    rel.method = $method,
                    rel.evidence_chunk_id = coalesce(rel.evidence_chunk_id, $evidence_chunk_id),
                    rel.evidence_sentences = CASE
                        WHEN $evidence_sentences IS NULL OR size($evidence_sentences) = 0 THEN rel.evidence_sentences
                        ELSE $evidence_sentences
                    END,
                    rel.evidence_confidence = coalesce($evidence_confidence, rel.evidence_confidence),
                    rel.last_seen = datetime()
                """,
                cp=cp,
                ct=ct,
                dp=dp,
                dt=dt,
                confidence=float(confidence),
                method=method or "heuristic",
                evidence_chunk_id=evidence_chunk_id,
                evidence_sentences=list(evidence_sentences) if evidence_sentences else None,
                evidence_confidence=evidence_confidence,
            )

        try:
            with driver.session() as session:
                session.execute_write(_tx)
        except Exception:
            logging.exception("neo4j_merge_prereq_failed")


def merge_alias(
    alias_name: str,
    canonical_name: str,
    method: str = "definition_alias",
    evidence_chunk_id: str | None = None,
) -> None:
    ca, da = canonicalize_concept(canonical_name)
    cb, db = canonicalize_concept(alias_name)
    if not ca or not cb or ca == cb:
        return

    with managed_driver() as driver:
        if driver is None:
            return

        def _tx(tx):
            tx.run(
                """
                MERGE (c:Concept {canonical_name: $canonical})
                ON CREATE SET c.display_name = $canonical_display, c.name_lower = $canonical, c.created_at = datetime()
                SET c.display_name = coalesce(c.display_name, $canonical_display), c.last_seen = datetime(), c.name_lower = $canonical
                MERGE (a:Concept {canonical_name: $alias})
                ON CREATE SET a.display_name = $alias_display, a.name_lower = $alias, a.created_at = datetime(), a.is_alias = true
                SET a.display_name = coalesce(a.display_name, $alias_display),
                    a.last_seen = datetime(),
                    a.name_lower = $alias,
                    a.is_alias = true
                MERGE (a)-[rel:ALIAS_OF]->(c)
                SET rel.method = $method,
                    rel.evidence_chunk_id = coalesce(rel.evidence_chunk_id, $evidence_chunk_id),
                    rel.last_seen = datetime()
                """,
                canonical=ca,
                canonical_display=da,
                alias=cb,
                alias_display=db,
                method=method,
                evidence_chunk_id=evidence_chunk_id,
            )

        try:
            with driver.session() as session:
                session.execute_write(_tx)
        except Exception:
            logging.exception("neo4j_merge_alias_failed")
