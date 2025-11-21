from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import logging
import time

from core.auth import require_auth
from core.db import get_db_conn
from psycopg2.extras import Json
from agents import orchestrator_dispatch
from agents.tutor.runtime.context import TurnContext
from agents.tutor.persistence import next_turn_index
from agents.tutor.runtime_v2.orchestrator_pedagogical_mdp import run_pedagogical_mdp_turn
from agents.tutor.environment import run_environment_turn
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


class TutorModelStrategy(BaseModel):
    type: str  # "single" | "multi_rank" (future)
    model_id: Optional[str] = None
    model_ids: Optional[List[str]] = None


class TutorSessionStartRequest(BaseModel):
    mode: str  # "personal_notes" | "general"
    agent_action_mode: str  # "auto" | "step_by_step"
    resource_ids: Optional[List[str]] = None
    target_concepts: Optional[List[str]] = None
    model_strategy: Optional[TutorModelStrategy] = None


class TutorSessionMessageRequest(BaseModel):
    message: str
    # Step-by-step control surface
    #
    # In agent_action_mode="step_by_step" the client may:
    # - Send a normal conversational turn with free-text `message`.
    # - Send a pure control turn (no free-text) using one or more of:
    #   * confirmed_action: e.g. "continue" to accept the proposed step.
    #   * step_control: structured control such as {"type": "skip_to_quiz"}.
    #   * mcq_answer: structured quiz answer payload.
    #
    # The tutor runtime responds with canonical `step` and `plan` fields on
    # the main payload for step-by-step sessions, backed by the StepEngine
    # StudyStep / StudyPlan projections, while keeping legacy `srl_plan` and
    # `srl_next_step` fields for a migration period.
    confirmed_action: Optional[str] = None
    action_override: Optional[Dict[str, Any]] = None
    mcq_answer: Optional[Dict[str, Any]] = None
    step_control: Optional[Dict[str, Any]] = None


class TutorCandidate(BaseModel):
    id: int
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    response: str
    action_type: Optional[str] = None
    intent: Optional[str] = None
    affect: Optional[str] = None
    concept: Optional[str] = None
    confidence: Optional[float] = None
    source_chunk_ids: Optional[List[str]] = None


class TutorPreferenceRating(BaseModel):
    correctness: Optional[int] = None
    helpfulness: Optional[int] = None
    coverage: Optional[int] = None
    notes: Optional[str] = None


class TutorChooseCandidateRequest(BaseModel):
    message: str
    chosen_id: int
    candidates: List[TutorCandidate]
    rating: Optional[TutorPreferenceRating] = None


@router.post("/api/tutor/session/start")
async def tutor_session_start(body: TutorSessionStartRequest, user_id: str = Depends(require_auth)):
    mode = (body.mode or "").strip().lower()
    if mode not in {"personal_notes", "general"}:
        raise HTTPException(status_code=400, detail="invalid mode")

    agent_action_mode = (body.agent_action_mode or "").strip().lower()
    if agent_action_mode not in {"auto", "step_by_step"}:
        raise HTTPException(status_code=400, detail="invalid agent_action_mode")

    resource_ids = body.resource_ids or []
    primary_resource_id = resource_ids[0] if resource_ids else None

    # Derive model strategy
    strategy_type = "single"
    model_ids: Optional[List[str]] = None

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            ms = body.model_strategy
            if ms is not None:
                strategy_type = (ms.type or "single").strip().lower()
                if strategy_type not in {"single", "multi_rank"}:
                    raise HTTPException(status_code=400, detail="invalid model_strategy.type")
                if strategy_type == "single":
                    chosen_id = (ms.model_id or "").strip() or None
                    if chosen_id is None:
                        cur.execute(
                            """
                            SELECT id
                            FROM llm_model
                            WHERE enabled = TRUE
                            ORDER BY cost_hint NULLS FIRST, latency_hint NULLS FIRST, id
                            LIMIT 1
                            """,
                        )
                        row_model = cur.fetchone()
                        chosen_id = row_model[0] if row_model else None
                    model_ids = [chosen_id] if chosen_id else None
                else:
                    ids = ms.model_ids or []
                    ids = [i.strip() for i in ids if i and i.strip()]
                    if len(ids) < 2:
                        raise HTTPException(status_code=400, detail="multi_rank requires at least two model_ids")
                    model_ids = ids
            else:
                # Default to first enabled model if present
                cur.execute(
                    """
                    SELECT id
                    FROM llm_model
                    WHERE enabled = TRUE
                    ORDER BY cost_hint NULLS FIRST, latency_hint NULLS FIRST, id
                    LIMIT 1
                    """,
                )
                row_model = cur.fetchone()
                default_id = row_model[0] if row_model else None
                model_ids = [default_id] if default_id else None

            cur.execute(
                """
                INSERT INTO tutor_session (
                    user_id,
                    resource_id,
                    target_concepts,
                    status,
                    policy,
                    last_concept,
                    last_action,
                    mode,
                    agent_action_mode,
                    resource_ids,
                    model_strategy_type,
                    model_ids
                )
                VALUES (
                    %s::uuid,
                    NULLIF(%s, '')::uuid,
                    %s::text[],
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::uuid[],
                    %s,
                    %s
                )
                RETURNING id::text
                """,
                (
                    user_id,
                    primary_resource_id,
                    body.target_concepts or [],
                    "active",
                    None,
                    None,
                    None,
                    mode,
                    agent_action_mode,
                    resource_ids,
                    strategy_type,
                    model_ids,
                ),
            )
            row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()

    session_id = row[0] if row and row[0] else None  # type: ignore[index]
    if session_id:
        try:
            logging.info(
                "tutor_session_start_api session_id=%s user_id=%s mode=%s agent_action_mode=%s strategy_type=%s resource_count=%s",
                session_id,
                user_id,
                mode,
                agent_action_mode,
                strategy_type,
                len(resource_ids),
            )
        except Exception:
            logging.exception("tutor_session_start_log_failed")
    if not session_id:
        raise HTTPException(status_code=500, detail="failed_to_create_session")

    return {
        "session_id": session_id,
        "mode": mode,
        "agent_action_mode": agent_action_mode,
        "resource_ids": resource_ids,
    }


@router.get("/api/tutor/models")
async def list_tutor_models(token: str = Depends(require_auth)) -> List[Dict[str, Any]]:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, provider, model_name, display_name, cost_hint, latency_hint
                FROM llm_model
                WHERE enabled = TRUE
                ORDER BY cost_hint NULLS FIRST, latency_hint NULLS FIRST, id
                """,
            )
            rows = cur.fetchall() or []
    finally:
        conn.close()

    models: List[Dict[str, Any]] = []
    for r in rows:
        models.append(
            {
                "id": r[0],
                "provider": r[1],
                "model_name": r[2],
                "display_name": r[3],
                "cost_hint": float(r[4]) if r[4] is not None else None,
                "latency_hint": float(r[5]) if r[5] is not None else None,
            }
        )

    return models


@router.post("/api/tutor/session/{session_id}/message")
async def tutor_session_message(session_id: str, body: TutorSessionMessageRequest, user_id: str = Depends(require_auth)):
    message = (body.message or "").strip()
    allows_empty = False
    # In step-by-step and quiz flows the client may send a pure control turn
    # (step_control, mcq_answer, and/or confirmed_action like "continue")
    # without free-text. Permit such turns while still requiring a message
    # for normal conversational usage.
    if body.step_control is not None or body.mcq_answer is not None or body.confirmed_action is not None:
        allows_empty = True
    if not message and not allows_empty:
        raise HTTPException(status_code=400, detail="message required")

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, user_id::text, mode, agent_action_mode, resource_ids, model_strategy_type, model_ids
                FROM tutor_session
                WHERE id = %s::uuid
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="session_not_found")
            session_user_id = row[1]
            if str(session_user_id) != str(user_id):
                raise HTTPException(status_code=403, detail="forbidden")
            mode = (row[2] or "general").strip().lower()
            agent_action_mode = (row[3] or "auto").strip().lower()
            resource_ids = row[4] or []
            model_strategy_type = (row[5] or "single").strip().lower()
            model_ids = row[6] or []

            # Resolve model_id and model_hint for this turn (single-model strategy only for now)
            model_id: Optional[str] = None
            model_hint: Optional[str] = None
            if model_strategy_type == "single" and model_ids:
                try:
                    model_id = str(model_ids[0])
                except Exception:
                    model_id = None
                if model_id:
                    try:
                        cur.execute(
                            """
                            SELECT model_name
                            FROM llm_model
                            WHERE id = %s AND enabled = TRUE
                            LIMIT 1
                            """,
                            (model_id,),
                        )
                        row_model = cur.fetchone()
                        if row_model and row_model[0]:
                            model_hint = row_model[0]
                    except Exception:
                        logging.exception("tutor_model_lookup_failed")
            if model_id and not model_hint:
                # Fallback: treat model_id as direct model hint when not present in llm_model
                model_hint = model_id
            try:
                logging.info(
                    "tutor_session_message_api session_id=%s mode=%s agent_action_mode=%s strategy_type=%s message_len=%s has_step_control=%s has_mcq=%s has_confirmed_action=%s",
                    session_id,
                    mode,
                    agent_action_mode,
                    model_strategy_type,
                    len(message),
                    body.step_control is not None,
                    body.mcq_answer is not None,
                    body.confirmed_action is not None,
                )
            except Exception:
                logging.exception("tutor_session_message_log_failed")
    finally:
        conn.close()

    primary_resource_id: Optional[str] = None
    if mode == "personal_notes" and resource_ids:
        try:
            primary_resource_id = str(resource_ids[0])
        except Exception:
            primary_resource_id = None

    # Multi-model comparison path: generate candidates with dry_run
    if model_strategy_type == "multi_rank":
        # Limit to at most 3 models
        model_ids_limited: List[str] = []
        for mid in model_ids:
            if len(model_ids_limited) >= 3:
                break
            if mid:
                model_ids_limited.append(str(mid))

        # Resolve model_id -> model_name
        model_name_map: Dict[str, str] = {}
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                for mid in model_ids_limited:
                    try:
                        cur.execute(
                            """
                            SELECT model_name
                            FROM llm_model
                            WHERE id = %s AND enabled = TRUE
                            LIMIT 1
                            """,
                            (mid,),
                        )
                        row_model = cur.fetchone()
                        if row_model and row_model[0]:
                            model_name_map[mid] = row_model[0]
                    except Exception:
                        logging.exception("tutor_multirank_model_lookup_failed")
        finally:
            conn.close()

        candidates: List[Dict[str, Any]] = []
        for idx, mid in enumerate(model_ids_limited):
            # Fall back to using model_id as the direct model name when not present in llm_model
            mname = model_name_map.get(mid) or mid
            if not mname:
                continue
            payload: Dict[str, Any] = {
                "message": message,
                "user_id": user_id,
                "session_id": session_id,
                "dry_run": True,
                "agent_action_mode": agent_action_mode,
            }
            if primary_resource_id:
                payload["resource_id"] = primary_resource_id
            payload["model_hint"] = mname
            payload["model_id"] = mid

            t0 = time.time()
            try:
                result = orchestrator_dispatch("tutor", payload)
                elapsed_ms = int((time.time() - t0) * 1000)
            except ValueError as ve:
                logging.exception("tutor_multirank_candidate_failed: %s", ve)
                continue
            except Exception:
                logging.exception("tutor_multirank_candidate_failed")
                continue

            candidates.append(
                {
                    "id": idx,
                    "model_id": mid,
                    "model_name": mname,
                    "response": result.get("response"),
                    "action_type": result.get("action_type"),
                    "intent": result.get("intent"),
                    "affect": result.get("affect"),
                    "concept": result.get("concept"),
                    "confidence": result.get("confidence"),
                    "source_chunk_ids": result.get("source_chunk_ids") or [],
                    "latency_ms": elapsed_ms,
                }
            )

        return {
            "session_id": session_id,
            "mode": mode,
            "agent_action_mode": agent_action_mode,
            "resource_ids": resource_ids,
            "model_strategy_type": model_strategy_type,
            "model_ids": model_ids_limited,
            "candidates": candidates,
        }

    # Single-model path (existing behaviour)
    payload: Dict[str, Any] = {
        "message": message,
        "user_id": user_id,
        "session_id": session_id,
        "agent_action_mode": agent_action_mode,
    }
    if primary_resource_id:
        payload["resource_id"] = primary_resource_id
    if model_hint:
        payload["model_hint"] = model_hint
    if model_id:
        payload["model_id"] = model_id
    if body.confirmed_action is not None:
        payload["confirmed_action"] = body.confirmed_action
    if body.mcq_answer is not None:
        payload["mcq_answer"] = body.mcq_answer
    if body.step_control is not None:
        payload["step_control"] = body.step_control

    # Optional explicit action override from client (auto/explain/ask/etc.)
    if body.action_override is not None:
        try:
            override = body.action_override
            if isinstance(override, dict):
                raw_type = (override.get("type") or "").strip()
                # Treat "auto" as no explicit override
                if raw_type and raw_type != "auto":
                    payload["action_override"] = override
        except Exception:
            logging.exception("tutor_session_message_action_override_error")

    try:
        result = orchestrator_dispatch("tutor", payload)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception:
        logging.exception("tutor_session_message_failed")
        raise HTTPException(status_code=500, detail="agent_error")

    turn_id = result.get("turn_id")
    action_type = result.get("action_type")

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    UPDATE tutor_session
                    SET mode = %s,
                        agent_action_mode = %s,
                        resource_ids = %s
                    WHERE id = %s::uuid
                    """,
                    (mode, agent_action_mode, resource_ids or None, session_id),
                )
            except Exception:
                logging.exception("tutor_session_update_failed")

            if turn_id and action_type:
                proposed_action = str(action_type)
                # Derive a simple user_choice label for logging
                if agent_action_mode == "step_by_step":
                    if body.confirmed_action and body.confirmed_action not in {"accept", "accepted"}:
                        user_choice = "overridden"
                    else:
                        user_choice = "accepted_step"
                else:
                    user_choice = "auto"
                try:
                    cur.execute(
                        """
                        UPDATE tutor_turn
                        SET proposed_action = %s,
                            user_choice = %s
                        WHERE id = %s::uuid
                        """,
                        (proposed_action, user_choice, str(turn_id)),
                    )
                except Exception:
                    logging.exception("tutor_turn_update_failed")

        conn.commit()
    finally:
        conn.close()

    result["mode"] = mode
    result["agent_action_mode"] = agent_action_mode
    result["resource_ids"] = resource_ids
    return result


@router.post("/api/tutor/pedagogical/session/{session_id}/message")
async def tutor_pedagogical_session_message(
    session_id: str,
    body: TutorSessionMessageRequest,
    user_id: str = Depends(require_auth),
):
    """Pedagogical MDP-specific tutor endpoint.

    This calls the new pedagogical MDP orchestrator directly instead of the
    legacy orchestrator_dispatch path.
    """

    message = (body.message or "").strip()
    allows_empty = False
    if body.step_control is not None or body.mcq_answer is not None or body.confirmed_action is not None:
        allows_empty = True
    if not message and not allows_empty:
        raise HTTPException(status_code=400, detail="message required")

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # Validate session and ownership; fetch target concepts for context.
            cur.execute(
                """
                SELECT id::text, user_id::text, target_concepts
                FROM tutor_session
                WHERE id = %s::uuid
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="session_not_found")
            session_user_id = row[1]
            if str(session_user_id) != str(user_id):
                raise HTTPException(status_code=403, detail="forbidden")
            target_concepts = row[2] or []

            turn_index = next_turn_index(cur, session_id)

            try:
                logging.info(
                    "tutor_pedagogical_session_message_api session_id=%s user_id=%s turn_index=%s message_len=%s has_step_control=%s has_mcq=%s has_confirmed_action=%s",
                    session_id,
                    user_id,
                    turn_index,
                    len(message),
                    body.step_control is not None,
                    body.mcq_answer is not None,
                    body.confirmed_action is not None,
                )
            except Exception:
                logging.exception("tutor_pedagogical_session_message_log_failed")

            ctx = TurnContext(
                session_id=session_id,
                user_id=user_id,
                turn_index=turn_index,
                message=message,
                target_concepts=target_concepts,
                resource_id=None,
                dry_run=False,
                emit_state_requested=False,
                payload={
                    "message": message,
                    "confirmed_action": body.confirmed_action,
                    "action_override": body.action_override,
                    "mcq_answer": body.mcq_answer,
                    "step_control": body.step_control,
                    "agent_action_mode": "step_by_step",
                },
            )

            result = run_pedagogical_mdp_turn(ctx, cur)
        conn.commit()
    finally:
        conn.close()

    return result


@router.post("/api/tutor/pedagogy/session/{session_id}/message")
async def tutor_environment_session_message(
    session_id: str,
    body: TutorSessionMessageRequest,
    user_id: str = Depends(require_auth),
):
    """Environment-based MDP endpoint using the new 3-layer architecture.
    
    This endpoint uses the clean environment orchestrator that coordinates:
    - Session layer (concept sequencing)
    - Concept layer (plan execution)
    - Tutor layer (pedagogical actions)
    """
    
    message = (body.message or "").strip()
    allows_empty = False
    if body.step_control is not None or body.mcq_answer is not None or body.confirmed_action is not None:
        allows_empty = True
    if not message and not allows_empty:
        raise HTTPException(status_code=400, detail="message required")
    
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # Validate session and ownership
            cur.execute(
                """
                SELECT id::text, user_id::text, target_concepts
                FROM tutor_session
                WHERE id = %s::uuid
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="session_not_found")
            session_user_id = row[1]
            if str(session_user_id) != str(user_id):
                raise HTTPException(status_code=403, detail="forbidden")
            target_concepts = row[2] or []
            
            turn_index = next_turn_index(cur, session_id)
            
            # Check if button was clicked
            button_clicked = False
            if body.confirmed_action in {"continue", "next", "yes"}:
                button_clicked = True
            elif body.step_control and body.step_control.get("type") in {"continue", "next"}:
                button_clicked = True
            
            try:
                logging.info(
                    "tutor_environment_session_message_api session_id=%s user_id=%s turn_index=%s message_len=%s button_clicked=%s",
                    session_id,
                    user_id,
                    turn_index,
                    len(message),
                    button_clicked,
                )
            except Exception:
                logging.exception("tutor_environment_session_message_log_failed")
            
            # Build turn context for environment
            ctx = TurnContext(
                session_id=session_id,
                user_id=user_id,
                turn_index=turn_index,
                message=message,
                target_concepts=target_concepts,
                resource_id=None,
                dry_run=False,
                emit_state_requested=False,
                payload={
                    "message": message,
                    "confirmed_action": body.confirmed_action,
                    "action_override": body.action_override,
                    "mcq_answer": body.mcq_answer,
                    "step_control": body.step_control,
                    "button_clicked": button_clicked,
                    "agent_action_mode": "environment_v1",
                },
            )
            
            # Run through environment orchestrator
            result = run_environment_turn(ctx, cur)
        conn.commit()
    finally:
        conn.close()
    
    return result


@router.post("/api/tutor/session/{session_id}/choose-candidate")
async def tutor_choose_candidate(session_id: str, body: TutorChooseCandidateRequest, user_id: str = Depends(require_auth)):
    message = (body.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message required")

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, user_id::text, mode, agent_action_mode, resource_ids
                FROM tutor_session
                WHERE id = %s::uuid
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="session_not_found")
            session_user_id = row[1]
            if str(session_user_id) != str(user_id):
                raise HTTPException(status_code=403, detail="forbidden")
            mode = (row[2] or "general").strip().lower()
            agent_action_mode = (row[3] or "auto").strip().lower()
            resource_ids = row[4] or []

            # Compute next turn index
            cur.execute(
                """
                SELECT COALESCE(MAX(turn_index), -1) + 1
                FROM tutor_turn
                WHERE session_id = %s::uuid
                """,
                (session_id,),
            )
            row_idx = cur.fetchone()
            turn_index = int(row_idx[0]) if row_idx and row_idx[0] is not None else 0

            # Find chosen candidate
            chosen = None
            for cand in body.candidates:
                if cand.id == body.chosen_id:
                    chosen = cand
                    break
            if chosen is None:
                raise HTTPException(status_code=400, detail="chosen_id not found in candidates")

            source_chunk_ids = chosen.source_chunk_ids or []

            rating_payload: Dict[str, Any] = {}
            if body.rating is not None:
                rating_payload = {
                    "correctness": body.rating.correctness,
                    "helpfulness": body.rating.helpfulness,
                    "coverage": body.rating.coverage,
                    "notes": body.rating.notes,
                    "chosen_id": body.chosen_id,
                    "chosen_model_id": chosen.model_id,
                    "chosen_model_name": chosen.model_name,
                }

            cur.execute(
                """
                INSERT INTO tutor_turn (
                  session_id,
                  turn_index,
                  user_text,
                  intent,
                  affect,
                  concept,
                  action_type,
                  response_text,
                  source_chunk_ids,
                  confidence,
                  mastery_delta,
                  model_id,
                  model_name,
                  candidates,
                  preference_rating,
                  user_choice
                )
                VALUES (
                  %s::uuid,
                  %s,
                  %s,
                  %s,
                  %s,
                  %s,
                  %s,
                  %s,
                  %s::uuid[],
                  %s,
                  %s,
                  %s,
                  %s,
                  %s,
                  %s,
                  %s
                )
                RETURNING id::text
                """,
                (
                    session_id,
                    turn_index,
                    message,
                    chosen.intent,
                    chosen.affect,
                    chosen.concept,
                    chosen.action_type,
                    chosen.response,
                    source_chunk_ids,
                    chosen.confidence,
                    None,
                    chosen.model_id,
                    chosen.model_name,
                    Json([c.dict() for c in body.candidates]),
                    Json(rating_payload) if rating_payload else None,
                    "multi_rank_chosen",
                ),
            )
            row_turn = cur.fetchone()
            turn_id = row_turn[0] if row_turn and row_turn[0] else None

            # Update session last concept/action
            try:
                cur.execute(
                    """
                    UPDATE tutor_session
                    SET last_concept = COALESCE(%s, last_concept),
                        last_action = %s,
                        updated_at = now()
                    WHERE id = %s::uuid
                    """,
                    (chosen.concept, chosen.action_type, session_id),
                )
            except Exception:
                logging.exception("tutor_choose_candidate_update_session_failed")

        conn.commit()
    finally:
        conn.close()

    if not turn_id:
        raise HTTPException(status_code=500, detail="failed_to_create_turn")

    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "turn_index": turn_index,
        "response": chosen.response,
        "action_type": chosen.action_type,
        "intent": chosen.intent,
        "affect": chosen.affect,
        "concept": chosen.concept,
        "confidence": chosen.confidence,
        "model_id": chosen.model_id,
        "model_name": chosen.model_name,
        "mode": mode,
        "agent_action_mode": agent_action_mode,
        "resource_ids": resource_ids,
        "candidates": [c.dict() for c in body.candidates],
        "preference_rating": rating_payload,
    }
