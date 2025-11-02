#!/usr/bin/env python3
"""Utility script to wipe the Neo4j knowledge graph (use with caution)."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dotenv import load_dotenv, find_dotenv  # type: ignore  # noqa: E402
from backend.core.kg_base import managed_driver  # noqa: E402


def wipe_neo4j(dry_run: bool) -> None:
    with managed_driver() as driver:
        if driver is None:
            raise RuntimeError("Failed to initialize Neo4j driver")

        def _tx(tx):
            if dry_run:
                count_nodes = tx.run("MATCH (n) RETURN count(n) AS nodes").single()["nodes"]
                count_rels = tx.run("MATCH ()-[r]-() RETURN count(r) AS rels").single()["rels"]
                logging.info("[dry-run] nodes=%s relationships=%s would be deleted", count_nodes, count_rels)
                return
            tx.run("MATCH (n) DETACH DELETE n")

        with driver.session() as session:
            session.execute_write(_tx)


def main() -> None:
    load_dotenv(find_dotenv(), override=False)

    parser = argparse.ArgumentParser(description="Wipe all nodes and relationships from Neo4j.")
    parser.add_argument("--dry-run", action="store_true", help="Show counts only, do not delete")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    logging.warning("Preparing to wipe Neo4j (dry_run=%s)", args.dry_run)
    wipe_neo4j(args.dry_run)
    logging.warning("Neo4j wipe completed (dry_run=%s)", args.dry_run)


if __name__ == "__main__":
    main()
