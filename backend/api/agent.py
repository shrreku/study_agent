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

from metrics import MetricsCollector
from prompts import active_set as prompts_active_set
# from llm.common import model_override_context

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


# @router.post("/api/tutor/session/{session_id}/message")
# async def tutor_session_message(session_id: str, body: TutorSessionMessageRequest, user_id: str = Depends(require_auth)):
#     message = (body.message or "").strip()
#     allows_empty = False
#     # In step-by-step and quiz flows the client may send a pure control turn
#     # (step_control, mcq_answer, and/or confirmed_action like "continue")
#     # without free-text. Permit such turns while still requiring a message
#     # for normal conversational usage.
#     if body.step_control is not None or body.mcq_answer is not None or body.confirmed_action is not None:
#         allows_empty = True
#     if not message and not allows_empty:
#         raise HTTPException(status_code=400, detail="message required")

#     conn = get_db_conn()
#     try:
#         with conn.cursor() as cur:
#             cur.execute(
#                 """
#                 SELECT id::text, user_id::text, mode, agent_action_mode, resource_ids, model_strategy_type, model_ids
#                 FROM tutor_session
#                 WHERE id = %s::uuid
#                 LIMIT 1
#                 """,
#                 (session_id,),
#             )
#             row = cur.fetchone()
#             if not row:
#                 raise HTTPException(status_code=404, detail="session_not_found")
#             session_user_id = row[1]
#             if str(session_user_id) != str(user_id):
#                 raise HTTPException(status_code=403, detail="forbidden")
#             mode = (row[2] or "general").strip().lower()
#             agent_action_mode = (row[3] or "auto").strip().lower()
#             resource_ids = row[4] or []
#             model_strategy_type = (row[5] or "single").strip().lower()
#             model_ids = row[6] or []

#             # Resolve model_id and model_hint for this turn (single-model strategy only for now)
#             model_id: Optional[str] = None
#             model_hint: Optional[str] = None
#             if model_strategy_type == "single" and model_ids:
#                 try:
#                     model_id = str(model_ids[0])
#                 except Exception:
#                     model_id = None
#                 if model_id:
#                     try:
#                         cur.execute(
#                             """
#                             SELECT model_name
#                             FROM llm_model
#                             WHERE id = %s AND enabled = TRUE
#                             LIMIT 1
#                             """,
#                             (model_id,),
#                         )
#                         row_model = cur.fetchone()
#                         if row_model and row_model[0]:
#                             model_hint = row_model[0]
#                     except Exception:
#                         logging.exception("tutor_model_lookup_failed")
#             if model_id and not model_hint:
#                 # Fallback: treat model_id as direct model hint when not present in llm_model
#                 model_hint = model_id
            
#             # Allow override from body if present
#             if body.model_hint:
#                 model_hint = body.model_hint

#             try:
#                 logging.info(
#                     "tutor_session_message_api session_id=%s mode=%s agent_action_mode=%s strategy_type=%s message_len=%s has_step_control=%s has_mcq=%s has_confirmed_action=%s model_hint=%s",
#                     session_id,
#                     mode,
#                     agent_action_mode,
#                     model_strategy_type,
#                     len(message),
#                     body.step_control is not None,
#                     body.mcq_answer is not None,
#                     body.confirmed_action is not None,
#                     model_hint
#                 )
#             except Exception:
#                 logging.exception("tutor_session_message_log_failed")
#     finally:
#         conn.close()

#     primary_resource_id: Optional[str] = None
#     if mode == "personal_notes" and resource_ids:
#         try:
#             primary_resource_id = str(resource_ids[0])
#         except Exception:
#             primary_resource_id = None

#     # Multi-model comparison path: generate candidates with dry_run
#     if model_strategy_type == "multi_rank":
#         # Limit to at most 3 models
#         model_ids_limited: List[str] = []
#         for mid in model_ids:
#             if len(model_ids_limited) >= 3:
#                 break
#             if mid:
#                 model_ids_limited.append(str(mid))

#         # Resolve model_id -> model_name
#         model_name_map: Dict[str, str] = {}
#         conn = get_db_conn()
#         try:
#             with conn.cursor() as cur:
#                 for mid in model_ids_limited:
#                     try:
#                         cur.execute(
#                             """
#                             SELECT model_name
#                             FROM llm_model
#                             WHERE id = %s AND enabled = TRUE
#                             LIMIT 1
#                             """,
#                             (mid,),
#                         )
#                         row_model = cur.fetchone()
#                         if row_model and row_model[0]:
#                             model_name_map[mid] = row_model[0]
#                     except Exception:
#                         logging.exception("tutor_multirank_model_lookup_failed")
#         finally:
#             conn.close()

#         candidates: List[Dict[str, Any]] = []
#         for idx, mid in enumerate(model_ids_limited):
#             # Fall back to using model_id as the direct model name when not present in llm_model
#             mname = model_name_map.get(mid) or mid
#             if not mname:
#                 continue
#             payload: Dict[str, Any] = {
#                 "message": message,
#                 "user_id": user_id,
#                 "session_id": session_id,
#                 "dry_run": True,
#                 "agent_action_mode": agent_action_mode,
#             }
#             if primary_resource_id:
#                 payload["resource_id"] = primary_resource_id
#             payload["model_hint"] = mname
#             payload["model_id"] = mid

#             t0 = time.time()
#             try:
#                 result = orchestrator_dispatch("tutor", payload)
#                 elapsed_ms = int((time.time() - t0) * 1000)
#             except ValueError as ve:
#                 logging.exception("tutor_multirank_candidate_failed: %s", ve)
#                 continue
#             except Exception:
#                 logging.exception("tutor_multirank_candidate_failed")
#                 continue

#             candidates.append(
#                 {
#                     "id": idx,
#                     "model_id": mid,
#                     "model_name": mname,
#                     "response": result.get("response"),
#                     "action_type": result.get("action_type"),
#                     "intent": result.get("intent"),
#                     "affect": result.get("affect"),
#                     "concept": result.get("concept"),
#                     "confidence": result.get("confidence"),
#                     "source_chunk_ids": result.get("source_chunk_ids") or [],
#                     "latency_ms": elapsed_ms,
#                 }
#             )

#         return {
#             "session_id": session_id,
#             "mode": mode,
#             "agent_action_mode": agent_action_mode,
#             "resource_ids": resource_ids,
#             "model_strategy_type": model_strategy_type,
#             "model_ids": model_ids_limited,
#             "candidates": candidates,
#         }

#     # Single-model path (existing behaviour)
#     payload: Dict[str, Any] = {
#         "message": message,
#         "user_id": user_id,
#         "session_id": session_id,
#         "agent_action_mode": agent_action_mode,
#     }
#     if primary_resource_id:
#         payload["resource_id"] = primary_resource_id
#     if model_hint:
#         payload["model_hint"] = model_hint
#     if model_id:
#         payload["model_id"] = model_id
#     if body.confirmed_action is not None:
#         payload["confirmed_action"] = body.confirmed_action
#     if body.mcq_answer is not None:
#         payload["mcq_answer"] = body.mcq_answer
#     if body.step_control is not None:
#         payload["step_control"] = body.step_control

#     # Optional explicit action override from client (auto/explain/ask/etc.)
#     if body.action_override is not None:
#         try:
#             override = body.action_override
#             if isinstance(override, dict):
#                 raw_type = (override.get("type") or "").strip()
#                 # Treat "auto" as no explicit override
#                 if raw_type and raw_type != "auto":
#                     payload["action_override"] = override
#         except Exception:
#             logging.exception("tutor_session_message_action_override_error")

#     try:
#         result = orchestrator_dispatch("tutor", payload)
#     except ValueError as ve:
#         raise HTTPException(status_code=404, detail=str(ve))
#     except Exception:
#         logging.exception("tutor_session_message_failed")
#         raise HTTPException(status_code=500, detail="agent_error")

#     turn_id = result.get("turn_id")
#     action_type = result.get("action_type")

#     conn = get_db_conn()
#     try:
#         with conn.cursor() as cur:
#             try:
#                 cur.execute(
#                     """
#                     UPDATE tutor_session
#                     SET mode = %s,
#                         agent_action_mode = %s,
#                         resource_ids = %s
#                     WHERE id = %s::uuid
#                     """,
#                     (mode, agent_action_mode, resource_ids or None, session_id),
#                 )
#             except Exception:
#                 logging.exception("tutor_session_update_failed")

#             if turn_id and action_type:
#                 proposed_action = str(action_type)
#                 # Derive a simple user_choice label for logging
#                 if agent_action_mode == "step_by_step":
#                     if body.confirmed_action and body.confirmed_action not in {"accept", "accepted"}:
#                         user_choice = "overridden"
#                     else:
#                         user_choice = "accepted_step"
#                 else:
#                     user_choice = "auto"
#                 try:
#                     cur.execute(
#                         """
#                         UPDATE tutor_turn
#                         SET proposed_action = %s,
#                             user_choice = %s
#                         WHERE id = %s::uuid
#                         """,
#                         (proposed_action, user_choice, str(turn_id)),
#                     )
#                 except Exception:
#                     logging.exception("tutor_turn_update_failed")

#         conn.commit()
#     finally:
#         conn.close()

#     result["mode"] = mode
#     result["agent_action_mode"] = agent_action_mode
#     result["resource_ids"] = resource_ids
#     return result

