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
from prompts import active_set as prompts_active_set

router = APIRouter()


class AgentRequest(BaseModel):
    target_concepts: Optional[List[str]] = None
    concepts: Optional[List[str]] = None
    count: Optional[int] = None
    question: Optional[str] = None
    question_text: Optional[str] = None
    context_chunk_ids: Optional[List[str]] = None
    user_id: Optional[str] = None
    resource_id: Optional[str] = None
    top_n: Optional[int] = None
    message: Optional[str] = None
    session_id: Optional[str] = None
    session_policy: Optional[Dict[str, Any]] = None
    intent: Optional[str] = None
    affect: Optional[str] = None
    mastery_delta: Optional[float] = None
    # New: allow clients to hint a model and force an action type
    model_hint: Optional[str] = None
    action_override: Optional[Dict[str, Any]] = None


@router.post("/api/agent/{agent_name}")
async def agent_endpoint(agent_name: str, body: AgentRequest, token: str = Depends(require_auth)):
    mc = MetricsCollector.get_global()
    t0 = time.time()
    # Observability: track doubt calls early and set prompt_set context
    prompt_set = os.getenv("PROMPT_SET", "default").strip() or "default"
    if agent_name in {"doubt", "tutor"}:
        try:
            mc.increment(f"{agent_name}_calls_total")
            mc.increment(f"{agent_name}_calls_total_ps_{prompt_set}")
        except Exception:
            logging.exception("doubt_calls_metric_failed")
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
            # Prompt set tagged counters for experiment analysis
            mc.increment(f"agent_calls_total_ps_{prompt_set}")
            mc.increment(f"agent_{agent_name}_calls_ps_{prompt_set}")
        except Exception:
            logging.exception("agent_metrics_failed")
        try:
            ps = prompts_active_set()
            mc.increment(f"prompt_set_{ps}_agent_calls")
            mc.increment(f"agent_{agent_name}_promptset_{ps}")
        except Exception:
            pass


class QuizAnswerRequest(BaseModel):
    quiz_id: str
    answers: List[Dict[str, Any]]
    user_id: Optional[str] = None


@router.post("/api/agent/quiz/answer")
async def submit_quiz_answer(req: QuizAnswerRequest, token: str = Depends(require_auth)):
    user_id = req.user_id or os.getenv("TEST_USER_ID") or None
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required (set TEST_USER_ID env var or pass user_id in body)")

    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default

    step_correct = _env_float("MASTERY_STEP_CORRECT", 0.1)
    step_wrong = _env_float("MASTERY_STEP_WRONG", -0.05)
    if step_wrong > 0:
        step_wrong = -abs(step_wrong)

    updated = 0
    total = 0
    correct_total = 0
    mastery_updates: List[Dict[str, Any]] = []
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            for ans in req.answers:
                concept = ans.get("concept")
                if not concept:
                    continue
                chosen = int(ans.get("chosen", -1))
                correct = int(ans.get("correct_index", -1))
                is_correct = 1 if chosen == correct else 0
                delta = step_correct if is_correct else step_wrong
                initial_mastery = step_correct if is_correct else 0.0
                cur.execute(
                    """
                    INSERT INTO user_concept_mastery (user_id, concept, mastery, last_seen, attempts, correct)
                    VALUES (%s::uuid, %s, %s, now(), %s, %s)
                    ON CONFLICT (user_id, concept) DO UPDATE
                      SET attempts = user_concept_mastery.attempts + 1,
                          correct = user_concept_mastery.correct + EXCLUDED.correct,
                          last_seen = now(),
                          mastery = LEAST(1.0, GREATEST(0.0, user_concept_mastery.mastery + %s))
                    RETURNING user_concept_mastery.mastery, user_concept_mastery.attempts, user_concept_mastery.correct
                    """,
                    (user_id, concept, initial_mastery, 1, is_correct, delta),
                )
                row = cur.fetchone()
                updated += 1
                total += 1
                correct_total += is_correct
                mastery_updates.append(
                    {
                        "concept": concept,
                        "correct": bool(is_correct),
                        "delta": delta,
                        "mastery": float(row[0]) if row else None,
                        "attempts": int(row[1]) if row else None,
                        "correct_attempts": int(row[2]) if row else None,
                    }
                )
        conn.commit()
    finally:
        conn.close()
    # Metrics roll-up for quiz grading
    try:
        mc = MetricsCollector.get_global()
        incorrect = max(0, total - correct_total)
        mc.increment("quiz_answers_total", total)
        mc.increment("quiz_answers_correct", correct_total)
        mc.increment("quiz_answers_incorrect", incorrect)
        if correct_total:
            mc.increment("quiz_strength_signals_total", correct_total)
        if incorrect:
            mc.increment("quiz_weak_signals_total", incorrect)
    except Exception:
        logging.exception("quiz_metrics_failed")
    return {"graded": updated, "updates": mastery_updates}
