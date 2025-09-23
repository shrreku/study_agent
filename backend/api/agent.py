from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import logging
import time

from core.auth import require_auth
from core.db import get_db_conn
from agents import orchestrator_dispatch
from metrics import MetricsCollector

router = APIRouter()


class AgentRequest(BaseModel):
    target_concepts: Optional[List[str]] = None
    concepts: Optional[List[str]] = None
    count: Optional[int] = None
    question: Optional[str] = None
    question_text: Optional[str] = None
    context_chunk_ids: Optional[List[str]] = None


@router.post("/api/agent/{agent_name}")
async def agent_endpoint(agent_name: str, body: AgentRequest, token: str = Depends(require_auth)):
    mc = MetricsCollector.get_global()
    t0 = time.time()
    try:
        payload = {k: v for k, v in body.dict().items() if v is not None}
        if not payload.get("question"):
            alias_q = payload.get("question_text") or payload.get("q")
            if alias_q:
                payload["question"] = alias_q
        result = orchestrator_dispatch(agent_name, payload)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception:
        logging.exception("agent_error")
        raise HTTPException(status_code=500, detail="agent_error")
    finally:
        try:
            elapsed_ms = int((time.time() - t0) * 1000)
            mc.increment("agent_calls_total")
            mc.increment(f"agent_{agent_name}_calls")
            mc.timing(f"agent_{agent_name}_elapsed_ms", elapsed_ms)
        except Exception:
            logging.exception("agent_metrics_failed")


class QuizAnswerRequest(BaseModel):
    quiz_id: str
    answers: List[Dict[str, Any]]


@router.post("/api/agent/quiz/answer")
async def submit_quiz_answer(req: QuizAnswerRequest, token: str = Depends(require_auth)):
    user_id = os.getenv("TEST_USER_ID") or None
    if not user_id:
        raise HTTPException(status_code=400, detail="TEST_USER_ID env var required for grading in dev")

    updated = 0
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            for ans in req.answers:
                concept = ans.get("concept")
                chosen = int(ans.get("chosen", -1))
                correct = int(ans.get("correct_index", -1))
                is_correct = 1 if chosen == correct else 0
                cur.execute(
                    """
                    INSERT INTO user_concept_mastery (user_id, concept, mastery, last_seen, attempts, correct)
                    VALUES (%s::uuid, %s, %s, now(), %s, %s)
                    ON CONFLICT (user_id, concept) DO UPDATE
                      SET attempts = user_concept_mastery.attempts + 1,
                          correct = user_concept_mastery.correct + EXCLUDED.correct,
                          last_seen = now(),
                          mastery = LEAST(1.0, GREATEST(0.0, user_concept_mastery.mastery + (CASE WHEN EXCLUDED.correct=1 THEN 0.1 ELSE -0.05 END)))
                    """,
                    (user_id, concept, 0.0 + (0.1 if is_correct else 0.0), 1, is_correct),
                )
                updated += 1
        conn.commit()
    finally:
        conn.close()
    return {"graded": updated}
