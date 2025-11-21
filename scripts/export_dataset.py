#!/usr/bin/env python3
"""Export tutor sessions and turns to JSONL for analysis/RL datasets.

Usage:
  python scripts/export_dataset.py --out dataset_exports/tutor_sessions.jsonl \
      [--start 2024-01-01] [--end 2025-01-01] [--mode personal_notes|general]

The script connects to the same Postgres instance as the backend using
DATABASE_URL (or POSTGRES_* env vars via backend.core.db.get_db_conn).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
  sys.path.insert(0, str(BACKEND_DIR))

from backend.core.db import get_db_conn  # type: ignore  # noqa: E402


def _parse_date(value: Optional[str], default: Optional[datetime]) -> Optional[datetime]:
  if not value:
    return default
  try:
    return datetime.fromisoformat(value)
  except Exception:
    raise SystemExit(f"Invalid date format (expected ISO 8601): {value}")


def _iter_rows(start: Optional[datetime], end: Optional[datetime], mode: Optional[str]) -> Iterable[Dict[str, Any]]:
  conn = get_db_conn()
  try:
    with conn.cursor() as cur:
      params: List[Any] = []
      where_clauses: List[str] = []

      if start is not None:
        where_clauses.append("s.created_at >= %s")
        params.append(start)
      if end is not None:
        where_clauses.append("s.created_at <= %s")
        params.append(end)
      if mode:
        where_clauses.append("s.mode = %s")
        params.append(mode)

      where_sql = ""
      if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

      cur.execute(
        f"""
        SELECT
          s.id::text AS session_id,
          s.user_id::text AS user_id,
          s.mode,
          s.agent_action_mode,
          s.resource_ids,
          s.config,
          s.created_at AS session_created_at,
          s.updated_at AS session_updated_at,
          t.id::text AS turn_id,
          t.turn_index,
          t.user_text,
          t.response_text,
          t.intent,
          t.affect,
          t.concept,
          t.action_type,
          t.proposed_action,
          t.user_choice,
          t.model_id,
          t.model_name,
          t.tool_calls,
          t.retrieval_metadata,
          t.policy_trace,
          t.candidates,
          t.preference_rating,
          (
            SELECT json_agg(
              json_build_object(
                'id', a.id::text,
                'labeler_id', a.labeler_id::text,
                'correctness', a.correctness,
                'helpfulness', a.helpfulness,
                'coverage', a.coverage,
                'notes', a.notes,
                'created_at', a.created_at
              )
              ORDER BY a.created_at ASC
            )
            FROM tutor_annotation a
            WHERE a.turn_id = t.id
          ) AS annotations,
          t.source_chunk_ids,
          t.confidence,
          t.mastery_delta,
          t.created_at AS turn_created_at
        FROM tutor_session s
        JOIN tutor_turn t ON t.session_id = s.id
        {where_sql}
        ORDER BY s.id, t.turn_index
        """,
        params,
      )

      columns = [c[0] for c in cur.description]
      for row in cur:
        yield {col: row[idx] for idx, col in enumerate(columns)}
  finally:
    conn.close()


def _build_records(rows: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
  current_session_id: Optional[str] = None
  current: Optional[Dict[str, Any]] = None

  for row in rows:
    sid = row["session_id"]
    if current_session_id is None or sid != current_session_id:
      if current is not None:
        yield current
      current_session_id = sid
      current = {
        "session_id": sid,
        "user_id": row.get("user_id"),
        "mode": row.get("mode"),
        "agent_action_mode": row.get("agent_action_mode"),
        "resource_ids": row.get("resource_ids") or [],
        "config": row.get("config") or {},
        "created_at": row.get("session_created_at").isoformat() if row.get("session_created_at") else None,
        "updated_at": row.get("session_updated_at").isoformat() if row.get("session_updated_at") else None,
        "turns": [],
      }

    if current is None:
      continue

    turn = {
      "turn_id": row.get("turn_id"),
      "turn_index": row.get("turn_index"),
      "user_input": row.get("user_text"),
      "agent_response": row.get("response_text"),
      "intent": row.get("intent"),
      "affect": row.get("affect"),
      "concept": row.get("concept"),
      "action_type": row.get("action_type"),
      "proposed_action": row.get("proposed_action"),
      "user_choice": row.get("user_choice"),
      "model_id": row.get("model_id"),
      "model_name": row.get("model_name"),
      "tool_calls": row.get("tool_calls") or {},
      "retrieval_metadata": row.get("retrieval_metadata") or {},
      "policy_trace": row.get("policy_trace") or {},
      "candidates": row.get("candidates") or [],
      "preference_rating": row.get("preference_rating") or {},
      "annotations": row.get("annotations") or [],
      "source_chunk_ids": [str(cid) for cid in (row.get("source_chunk_ids") or [])],
      "confidence": float(row["confidence"]) if row.get("confidence") is not None else None,
      "mastery_delta": float(row["mastery_delta"]) if row.get("mastery_delta") is not None else None,
      "created_at": row.get("turn_created_at").isoformat() if row.get("turn_created_at") else None,
    }
    current["turns"].append(turn)

  if current is not None:
    yield current


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Export tutor sessions/turns to JSONL")
  parser.add_argument("--out", type=Path, required=True, help="Output JSONL file path")
  parser.add_argument("--start", type=str, default=None, help="Start datetime (ISO 8601)")
  parser.add_argument("--end", type=str, default=None, help="End datetime (ISO 8601)")
  parser.add_argument("--mode", type=str, default=None, help="Filter by session mode (personal_notes or general)")
  parser.add_argument("--days", type=int, default=30, help="Default lookback window in days if start/end not set")
  return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
  args = parse_args(argv)

  now = datetime.utcnow()
  default_start = now - timedelta(days=args.days)

  start = _parse_date(args.start, default_start if args.end is None else None)
  end = _parse_date(args.end, None)

  if args.mode:
    mode = args.mode.strip().lower()
    if mode not in {"personal_notes", "general"}:
      raise SystemExit("--mode must be 'personal_notes' or 'general'")
  else:
    mode = None

  args.out.parent.mkdir(parents=True, exist_ok=True)

  rows = _iter_rows(start=start, end=end, mode=mode)
  records = _build_records(rows)

  count = 0
  with args.out.open("w", encoding="utf-8") as handle:
    for record in records:
      handle.write(json.dumps(record, ensure_ascii=False) + "\n")
      count += 1

  print(f"Exported {count} sessions to {args.out}")
  return 0


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
