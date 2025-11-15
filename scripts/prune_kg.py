#!/usr/bin/env python3
"""
Prune low-quality nodes and relationships from the Neo4j Knowledge Graph.

Usage examples:
  python scripts/prune_kg.py --mode moderate --dry-run
  python scripts/prune_kg.py --mode aggressive

Modes:
  - conservative: keep most edges/nodes, minimal pruning
  - moderate: reasonable cleanup (default)
  - aggressive: heavy cleanup for noisy graphs
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is importable for `backend` package
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
APP_DIR = Path("/app")
if APP_DIR.exists() and str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

# Import managed_driver from proper package based on runtime context
try:
    if (APP_DIR / "core").exists():
        # Running inside backend container where /app is the backend root
        from kg_pipeline.base import managed_driver  # type: ignore
    else:
        from backend.kg_pipeline.base import managed_driver  # type: ignore
except ModuleNotFoundError:
    # Last-resort fallback: try direct import without prefix
    from kg_pipeline.base import managed_driver  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kg_prune")


def prune_singleton_nodes(driver, dry_run: bool = False) -> int:
    query = """
    MATCH (n:Concept)
    WHERE NOT (n)--()
    RETURN n.canonical_name AS id
    """
    with driver.session() as session:
        ids = [rec["id"] for rec in session.run(query)]
        logger.info("Singleton Concept nodes: %d", len(ids))
        if not dry_run and ids:
            session.run(
                """
                MATCH (n:Concept)
                WHERE n.canonical_name IN $ids
                DELETE n
                """,
                ids=ids,
            )
            logger.info("Deleted %d singleton concepts", len(ids))
        return len(ids)


def prune_high_degree_generic_nodes(driver, max_degree: int = 50, dry_run: bool = False) -> int:
    query = """
    MATCH (n:Concept)-[r]-()
    WITH n, count(r) AS degree
    WHERE degree > $max_degree
    RETURN n.canonical_name AS id, degree
    ORDER BY degree DESC
    """
    with driver.session() as session:
        rows = session.run(query, max_degree=max_degree)
        high = [(rec["id"], rec["degree"]) for rec in rows]
        logger.info("High-degree concepts > %d: %d", max_degree, len(high))
        for node_id, deg in high[:10]:
            logger.info("  %s: %d", node_id, deg)
        if not dry_run and high:
            session.run(
                """
                MATCH (n:Concept)
                WHERE n.canonical_name IN $ids
                DETACH DELETE n
                """,
                ids=[x[0] for x in high],
            )
            logger.info("Deleted %d high-degree nodes", len(high))
        return len(high)


def prune_low_confidence_relationships(driver, min_confidence: float = 0.5, dry_run: bool = False) -> int:
    count_q = """
    MATCH ()-[r]->()
    WHERE coalesce(r.confidence, 1.0) < $min
    RETURN count(r) AS cnt
    """
    delete_q = """
    MATCH ()-[r]->()
    WHERE coalesce(r.confidence, 1.0) < $min
    DELETE r
    """
    with driver.session() as session:
        cnt = session.run(count_q, min=min_confidence).single()["cnt"]
        logger.info("Low-confidence relationships (< %.2f): %d", min_confidence, cnt)
        if not dry_run and cnt > 0:
            session.run(delete_q, min=min_confidence)
            logger.info("Deleted %d low-confidence relationships", cnt)
        return int(cnt)


def detect_circular_prerequisites(driver) -> List[List[str]]:
    query = """
    MATCH path = (start:Concept)-[:PREREQUISITE_OF*2..10]->(start)
    RETURN [node IN nodes(path) | node.canonical_name] AS cycle
    LIMIT 100
    """
    with driver.session() as session:
        cycles = [rec["cycle"] for rec in session.run(query)]
        if cycles:
            logger.warning("Found %d circular prerequisite chains", len(cycles))
            for cycle in cycles[:5]:
                logger.warning("  %s", " → ".join(cycle))
        return cycles


def break_circular_prerequisites(driver, cycles: List[List[str]], dry_run: bool = False) -> int:
    weakest_q = """
    MATCH (a:Concept {canonical_name: $source})-[r:PREREQUISITE_OF]->(b:Concept {canonical_name: $target})
    RETURN r.confidence AS conf
    """
    with driver.session() as session:
        broken = 0
        for cycle in cycles:
            weakest = None
            min_conf = 1.1
            for i in range(len(cycle) - 1):
                rec = session.run(weakest_q, source=cycle[i], target=cycle[i + 1]).single()
                if rec and rec["conf"] is not None and rec["conf"] < min_conf:
                    min_conf = rec["conf"]
                    weakest = (cycle[i], cycle[i + 1])
            if weakest and not dry_run:
                session.run(
                    """
                    MATCH (a:Concept {canonical_name: $source})-[r:PREREQUISITE_OF]->(b:Concept {canonical_name: $target})
                    DELETE r
                    """,
                    source=weakest[0],
                    target=weakest[1],
                )
                logger.info("Broke cycle by removing %s → %s", weakest[0], weakest[1])
                broken += 1
        return broken


def main():
    parser = argparse.ArgumentParser(description="Prune Knowledge Graph")
    parser.add_argument("--mode", choices=["conservative", "moderate", "aggressive"], default="moderate")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    thresholds = {
        "conservative": {"max_degree": 100, "min_confidence": 0.3},
        "moderate": {"max_degree": 50, "min_confidence": 0.5},
        "aggressive": {"max_degree": 30, "min_confidence": 0.65},
    }
    cfg = thresholds[args.mode]

    logger.info("Starting KG pruning (mode=%s, dry_run=%s)", args.mode, args.dry_run)

    with managed_driver() as driver:
        if driver is None:
            logger.error("Neo4j driver unavailable. Check NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD.")
            return
        singletons = prune_singleton_nodes(driver, args.dry_run)
        high_degree = prune_high_degree_generic_nodes(driver, cfg["max_degree"], args.dry_run)
        low_conf = prune_low_confidence_relationships(driver, cfg["min_confidence"], args.dry_run)
        cycles = detect_circular_prerequisites(driver)
        broken = break_circular_prerequisites(driver, cycles, args.dry_run) if cycles else 0

    logger.info("Pruning summary:\n  Singletons: %d\n  High-degree removed: %d\n  Low-confidence removed: %d\n  Cycles broken: %d", singletons, high_degree, low_conf, broken)


if __name__ == "__main__":
    main()
