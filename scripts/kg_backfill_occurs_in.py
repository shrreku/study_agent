#!/usr/bin/env python3
"""Backfill enriched OCCURS_IN relationships in Neo4j for existing chunks."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, Iterable

from pathlib import Path

from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dotenv import load_dotenv, find_dotenv  # type: ignore  # noqa: E402
from backend.core.db import get_db_conn  # noqa: E402
from backend.core.kg import merge_concepts_in_neo4j  # noqa: E402


def _chunk_meta_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    section_level = row.get("section_level")
    if section_level is not None:
        try:
            section_level = int(section_level)
        except Exception:
            section_level = None

    return {
        "full_text": row.get("full_text") or "",
        "page_number": row.get("page_number"),
        "section_path": row.get("section_path") or [],
        "section_title": row.get("section_title"),
        "section_number": row.get("section_number"),
        "section_level": section_level,
        "chunk_type": row.get("chunk_type"),
    }


def _fetch_chunks(conn, resource_id: str | None, limit: int | None) -> Iterable[Dict[str, Any]]:
    base_query = (
        "SELECT c.id::text AS id, c.resource_id::text AS resource_id, c.full_text, c.text_snippet, "
        "c.concepts, c.page_number, c.section_title, c.section_number, c.section_path, "
        "c.section_level, c.chunk_type, c.updated_at "
        "FROM chunk c "
        "WHERE c.concepts IS NOT NULL AND array_length(c.concepts, 1) > 0"
    )
    params: tuple[Any, ...] = ()
    if resource_id:
        base_query += " AND c.resource_id = %s::uuid"
        params = (resource_id,)
    base_query += " ORDER BY c.updated_at DESC"
    if limit:
        base_query += " LIMIT %s"
        params = params + (limit,) if params else (limit,)

    cur = conn.cursor(name="kg_occurs_in_backfill", cursor_factory=RealDictCursor)
    cur.itersize = 500
    cur.execute(base_query, params)
    for row in cur:
        yield row
    cur.close()


def process_chunks(resource_id: str | None, limit: int | None, dry_run: bool) -> Dict[str, int]:
    stats = {"processed": 0, "skipped": 0, "merged": 0}

    conn = get_db_conn()
    try:
        for row in _fetch_chunks(conn, resource_id, limit):
            stats["processed"] += 1
            concepts = row.get("concepts") or []
            if not concepts:
                stats["skipped"] += 1
                continue

            chunk_meta = _chunk_meta_from_row(row)
            snippet = (row.get("text_snippet") or row.get("full_text") or "")[:160]

            if dry_run:
                logging.info(
                    "[dry-run] would merge concepts for chunk=%s resource=%s concepts=%d",
                    row.get("id"),
                    row.get("resource_id"),
                    len(concepts),
                )
                stats["merged"] += 1
                continue

            try:
                merge_concepts_in_neo4j(
                    concepts,
                    row.get("id"),
                    snippet,
                    row.get("resource_id"),
                    chunk_meta,
                )
                stats["merged"] += 1
            except Exception:
                logging.exception(
                    "merge_failed chunk=%s resource=%s",
                    row.get("id"),
                    row.get("resource_id"),
                )
    finally:
        conn.close()

    return stats


def main() -> None:
    load_dotenv(find_dotenv(), override=False)

    ap = argparse.ArgumentParser(description="Backfill OCCURS_IN edges with enriched metadata.")
    ap.add_argument("--resource-id", help="Limit backfill to a single resource UUID", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N chunks")
    ap.add_argument("--dry-run", action="store_true", help="Log actions without writing to Neo4j")
    ap.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging level (default INFO)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    logging.info(
        "starting KG OCCURS_IN backfill resource=%s limit=%s dry_run=%s",
        args.resource_id,
        args.limit,
        args.dry_run,
    )

    stats = process_chunks(args.resource_id, args.limit, args.dry_run)

    logging.info("backfill complete", extra={"stats": stats})
    print({"stats": stats})


if __name__ == "__main__":
    main()
