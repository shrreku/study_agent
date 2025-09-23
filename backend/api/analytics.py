from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import io as _io
import csv as _csv

from core.auth import require_auth
from core.db import get_db_conn

router = APIRouter()


@router.get("/api/analytics/mastery")
async def export_mastery_csv(user_id: str, token: str = Depends(require_auth)):
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id required")
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT concept, mastery, last_seen, attempts, correct FROM user_concept_mastery WHERE user_id=%s::uuid",
                    (user_id,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception:
        raise HTTPException(status_code=500, detail="export_failed")

    out = _io.StringIO()
    w = _csv.writer(out)
    w.writerow(["concept_name", "mastery_score", "last_seen", "attempts", "correct_rate"])
    for r in rows:
        concept = r[0]
        mastery = float(r[1]) if r[1] is not None else 0.0
        last_seen = r[2].isoformat() if r[2] else ""
        attempts = int(r[3] or 0)
        correct = int(r[4] or 0)
        rate = round((correct / attempts), 4) if attempts > 0 else 0.0
        w.writerow([concept, mastery, last_seen, attempts, rate])
    return out.getvalue()
