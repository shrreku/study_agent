#!/usr/bin/env python3
"""Backfill LLM-derived pedagogy relations for existing chunks."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# Also add backend/ so modules imported as top-level (e.g., `metrics`) resolve when backend code expects it
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))
APP_DIR = Path("/app")
if APP_DIR.exists() and str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from dotenv import load_dotenv, find_dotenv  # type: ignore  # noqa: E402
if (APP_DIR / "core").exists():
    from core.db import get_db_conn  # type: ignore  # noqa: E402
    from kg_pipeline import merge_chunk_pedagogy_relations  # type: ignore  # noqa: E402
    from llm import extract_pedagogy_relations  # type: ignore  # noqa: E402
else:
    from backend.core.db import get_db_conn  # type: ignore  # noqa: E402
    from backend.kg_pipeline import merge_chunk_pedagogy_relations  # type: ignore  # noqa: E402
    from backend.llm import extract_pedagogy_relations  # type: ignore  # noqa: E402


ENABLE_VALUES = {"1", "true", "yes", "on"}


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


def _fetch_chunks(conn, resource_id: str | None, limit: int | None):
    base_query = (
        "SELECT c.id::text AS id, c.resource_id::text AS resource_id, c.full_text, c.text_snippet, "
        "c.concepts, c.page_number, c.section_title, c.section_number, c.section_path, "
        "c.section_level, c.chunk_type, c.updated_at "
        "FROM chunk c "
        "WHERE c.full_text IS NOT NULL AND length(c.full_text) > 0"
    )
    params: tuple[Any, ...] = ()
    if resource_id:
        base_query += " AND c.resource_id = %s::uuid"
        params = (resource_id,)
    base_query += " ORDER BY c.updated_at DESC"
    if limit:
        base_query += " LIMIT %s"
        params = params + (limit,) if params else (limit,)

    cur = conn.cursor(name="kg_pedagogy_backfill", cursor_factory=RealDictCursor)
    cur.itersize = 200
    cur.execute(base_query, params)
    for row in cur:
        yield row
    cur.close()


def _section_key_from_row(row: Dict[str, Any]) -> str:
    path = row.get("section_path") or []
    parts: List[str] = []
    for seg in path:
        s = str(seg or "").strip()
        if s:
            parts.append(s)
    if not parts:
        num = str(row.get("section_number") or "").strip()
        title = str(row.get("section_title") or "").strip()
        if num:
            parts = [num]
        elif title:
            parts = [title]
        else:
            parts = [f"page-{int(row.get('page_number') or 0)}"]
    return "|".join(parts)


def _fetch_sections(conn, resource_id: str | None, limit: int | None):
    base_query = (
        "SELECT c.id::text AS id, c.resource_id::text AS resource_id, c.full_text, c.text_snippet, "
        "c.concepts, c.page_number, c.section_title, c.section_number, c.section_path, "
        "c.section_level, c.chunk_type, c.updated_at "
        "FROM chunk c "
        "WHERE c.full_text IS NOT NULL AND length(c.full_text) > 0"
    )
    params: tuple[Any, ...] = ()
    if resource_id:
        base_query += " AND c.resource_id = %s::uuid"
        params = (resource_id,)
    base_query += " ORDER BY c.resource_id, c.section_path, c.section_number, c.page_number"

    cur = conn.cursor(name="kg_pedagogy_backfill_sections", cursor_factory=RealDictCursor)
    cur.itersize = 500
    cur.execute(base_query, params)

    current_key: Optional[Tuple[str, str]] = None  # (resource_id, section_key)
    acc_rows: List[Dict[str, Any]] = []
    yielded = 0

    def _yield_group(rows: List[Dict[str, Any]]):
        if not rows:
            return None
        resid = rows[0].get("resource_id")
        s_title = rows[0].get("section_title")
        s_number = rows[0].get("section_number")
        s_level = rows[0].get("section_level")
        s_path = rows[0].get("section_path") or []
        # concatenate texts; allow larger cap for sections
        try:
            max_chars = int(os.getenv("PEDAGOGY_SECTION_MAX_CHARS", os.getenv("PEDAGOGY_LLM_MAX_CHARS", "6000")))
        except Exception:
            max_chars = 6000
        texts = []
        total = 0
        for r in rows:
            t = (r.get("full_text") or "").strip()
            if not t:
                continue
            if total + len(t) + 2 > max_chars:
                remaining = max_chars - total
                if remaining > 0:
                    texts.append(t[:remaining])
                    total += remaining
                break
            texts.append(t)
            total += len(t)
        combined = "\n\n".join(texts)
        chunk_ids = [r.get("id") for r in rows if r.get("id")]
        chunk_types = [str(r.get("chunk_type") or "").strip() for r in rows if r.get("chunk_type")]
        dominant_chunk_type = chunk_types[0] if chunk_types else None
        return {
            "resource_id": resid,
            "section_title": s_title,
            "section_number": s_number,
            "section_level": s_level,
            "section_path": s_path,
            "chunk_ids": chunk_ids,
            "full_text": combined,
            "chunk_type": dominant_chunk_type,
        }

    for row in cur:
        resid = row.get("resource_id")
        skey = _section_key_from_row(row)
        key = (resid, skey)
        if current_key is None:
            current_key = key
        if key != current_key:
            group = _yield_group(acc_rows)
            if group is not None:
                yield group
                yielded += 1
                if limit and yielded >= limit:
                    acc_rows = []
                    break
            acc_rows = []
            current_key = key
        acc_rows.append(row)

    if acc_rows and (not limit or yielded < limit):
        group = _yield_group(acc_rows)
        if group is not None:
            yield group
    cur.close()


def _should_enable_llm(force: bool) -> bool:
    if force:
        return True
    env_value = os.getenv("PEDAGOGY_LLM_ENABLE", "1").lower()
    return env_value in ENABLE_VALUES


def process_chunks(resource_id: str | None, limit: int | None, dry_run: bool, force_llm: bool, by_section: bool = False) -> Dict[str, int]:
    stats = {
        "processed": 0,
        "skipped_empty_text": 0,
        "skipped_disabled": 0,
        "llm_calls": 0,
        "payload_nonempty": 0,
        "merged": 0,
        "errors": 0,
    }

    if not _should_enable_llm(force_llm):
        logging.warning("PEDAGOGY_LLM_ENABLE disabled; use --force to override or set env var.")
        stats["skipped_disabled"] = -1  # sentinel to indicate no work attempted
        return stats

    conn = get_db_conn()
    try:
        iterator = _fetch_sections(conn, resource_id, limit) if by_section else _fetch_chunks(conn, resource_id, limit)
        for row in iterator:
            stats["processed"] += 1
            # for by_section mode, row carries aggregated section text and list of chunk_ids
            chunk_ids = row.get("chunk_ids") or []
            chunk_id = (chunk_ids[0] if chunk_ids else row.get("id"))
            resource = row.get("resource_id")
            full_text = row.get("full_text") or ""
            if not full_text.strip():
                stats["skipped_empty_text"] += 1
                continue

            chunk_meta = _chunk_meta_from_row(row)
            snippet = (row.get("text_snippet") or full_text)[:160]

            try:
                stats["llm_calls"] += 1
                pedagogy = extract_pedagogy_relations(
                    full_text,
                    {
                        "chunk_type": chunk_meta.get("chunk_type"),
                        "title": chunk_meta.get("section_title"),
                        "resource_id": resource,
                    },
                )
            except Exception:
                stats["errors"] += 1
                logging.exception("pedagogy_llm_failed chunk=%s resource=%s", chunk_id, resource)
                continue

            if any(pedagogy.get(key) for key in ("defines", "explains", "exemplifies", "proves", "derives", "figure_links", "prereqs")):
                stats["payload_nonempty"] += 1
            else:
                logging.debug("pedagogy payload empty chunk=%s", chunk_id)

            if dry_run:
                logging.info(
                    "[dry-run] would merge pedagogy relations chunk=%s resource=%s nonempty=%s",
                    chunk_id,
                    resource,
                    stats["payload_nonempty"],
                )
                stats["merged"] += 1
                continue

            try:
                merge_chunk_pedagogy_relations(
                    chunk_id,
                    resource,
                    pedagogy,
                    chunk_type=chunk_meta.get("chunk_type"),
                    method=("backfill_llm_pedagogy_section" if by_section else "backfill_llm_pedagogy"),
                )
                stats["merged"] += 1
            except Exception:
                stats["errors"] += 1
                logging.exception("pedagogy_merge_failed chunk=%s resource=%s", chunk_id, resource)
    finally:
        conn.close()

    return stats


def main() -> None:
    load_dotenv(find_dotenv(), override=False)

    ap = argparse.ArgumentParser(description="Backfill KG pedagogy relations via LLM extraction.")
    ap.add_argument("--resource-id", help="Limit backfill to a single resource UUID", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N chunks")
    ap.add_argument("--dry-run", action="store_true", help="Log actions without writing to Neo4j")
    ap.add_argument("--force", action="store_true", help="Ignore PEDAGOGY_LLM_ENABLE and force LLM calls")
    ap.add_argument("--by-section", action="store_true", help="Aggregate chunks per section and extract pedagogy once per section")
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
        "starting KG pedagogy backfill resource=%s limit=%s dry_run=%s force=%s",
        args.resource_id,
        args.limit,
        args.dry_run,
        args.force,
    )

    stats = process_chunks(args.resource_id, args.limit, args.dry_run, args.force, by_section=args.by_section)

    logging.info("pedagogy backfill complete", extra={"stats": stats})
    print({"stats": stats})


if __name__ == "__main__":
    main()
