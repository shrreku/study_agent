from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
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


class TutorAnnotationRequest(BaseModel):
    correctness: Optional[int] = None
    helpfulness: Optional[int] = None
    coverage: Optional[int] = None
    notes: Optional[str] = None


@router.post("/api/analytics/tutor/turn/{turn_id}/annotate")
async def annotate_tutor_turn(turn_id: str, body: TutorAnnotationRequest, labeler_id: str = Depends(require_auth)):
    if not turn_id or not turn_id.strip():
        raise HTTPException(status_code=400, detail="turn_id required")

    if not any([body.correctness, body.helpfulness, body.coverage, body.notes]):
        raise HTTPException(status_code=400, detail="at least one field must be provided")

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tutor_annotation (
                  turn_id,
                  labeler_id,
                  correctness,
                  helpfulness,
                  coverage,
                  notes
                )
                VALUES (
                  %s::uuid,
                  %s::uuid,
                  %s,
                  %s,
                  %s,
                  %s
                )
                RETURNING id::text
                """,
                (
                    turn_id,
                    labeler_id,
                    body.correctness,
                    body.helpfulness,
                    body.coverage,
                    body.notes,
                ),
            )
            row = cur.fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise HTTPException(status_code=500, detail="annotation_insert_failed")
    finally:
        conn.close()

    return {"ok": True, "annotation_id": row[0] if row else None}


@router.get("/api/analytics/tutor/turn/{turn_id}/annotations")
async def list_tutor_annotations(turn_id: str, token: str = Depends(require_auth)) -> Dict[str, Any]:
    if not turn_id or not turn_id.strip():
        raise HTTPException(status_code=400, detail="turn_id required")

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, labeler_id::text, correctness, helpfulness, coverage, notes, created_at
                FROM tutor_annotation
                WHERE turn_id = %s::uuid
                ORDER BY created_at ASC
                """,
                (turn_id,),
            )
            rows = cur.fetchall() or []
    except Exception:
        raise HTTPException(status_code=500, detail="annotation_query_failed")
    finally:
        conn.close()

    annotations: List[Dict[str, Any]] = []
    for r in rows:
        annotations.append(
            {
                "id": r[0],
                "labeler_id": r[1],
                "correctness": r[2],
                "helpfulness": r[3],
                "coverage": r[4],
                "notes": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
            }
        )

    return {"turn_id": turn_id, "annotations": annotations}
