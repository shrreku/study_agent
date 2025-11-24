from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from psycopg2.extras import Json

from core.auth import require_auth
from core.db import get_db_conn
from agents.tutor.runtime.context import TurnContext
from agents.tutor.persistence import next_turn_index
from agents.tutor.environment import run_environment_turn, TutorEnvironment, TutorAction
from agents.tutor.mdp.tools_factory import make_pedagogical_response_tool
from llm.common import model_override_context


logger = logging.getLogger(__name__)
router = APIRouter()


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
    confirmed_action: Optional[str] = None
    action_override: Optional[Dict[str, Any]] = None
    mcq_answer: Optional[Dict[str, Any]] = None
    step_control: Optional[Dict[str, Any]] = None
    # Allow clients to hint a model
    model_hint: Optional[str] = None


class TutorMDPPlaygroundStepRequest(BaseModel):
    concept_id: str
    pedagogical_action: str
    plan_index: int
    plan_length: int
    phase: Optional[str] = "learning"
    student_message: Optional[str] = None
    model_hint: Optional[str] = None


@router.post("/api/tutor/session/start")
async def tutor_session_start(body: TutorSessionStartRequest, user_id: str = Depends(require_auth)):
    """Start a tutor session for the environment / MDP flows.

    This is largely a copy of the original implementation from api.agent,
    but isolated here so tutor-related logic is easier to reason about.
    """
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
                            """
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
                    """
                )
                row_model = cur.fetchone()
                default_id = row_model[0] if row_model else None
                model_ids = [default_id] if default_id else None

            cur.execute(
                """
                INSERT INTO tutor_session (
                    user_id,
                    mode,
                    agent_action_mode,
                    resource_ids,
                    primary_resource_id,
                    model_strategy,
                    target_concepts
                )
                VALUES (%s::uuid, %s, %s, %s, %s, %s, %s)
                RETURNING id::text
                """,
                (
                    user_id,
                    mode,
                    agent_action_mode,
                    Json(resource_ids),
                    primary_resource_id,
                    Json({"type": strategy_type, "model_ids": model_ids} if model_ids else {"type": strategy_type}),
                    body.target_concepts or [],
                ),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=500, detail="failed to create session")
            session_id = row[0]

        conn.commit()
    finally:
        conn.close()

    return {"session_id": session_id}


@router.post("/api/tutor/pedagogy/session/{session_id}/message")
async def tutor_environment_session_message(
    session_id: str,
    body: TutorSessionMessageRequest,
    user_id: str = Depends(require_auth),
):
    """Environment-based MDP endpoint using the 3-layer architecture.

    This is a near-copy of the existing implementation in api.agent, moved
    here to keep tutor environment logic self-contained.
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
                logger.info(
                    "tutor_environment_session_message_api session_id=%s user_id=%s turn_index=%s message_len=%s button_clicked=%s model_hint=%s",
                    session_id,
                    user_id,
                    turn_index,
                    len(message),
                    button_clicked,
                    body.model_hint,
                )
            except Exception:
                logger.exception("tutor_environment_session_message_log_failed")

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

            # Run through environment orchestrator with model hint override
            with model_override_context(body.model_hint):
                result = run_environment_turn(ctx, cur)
        conn.commit()
    finally:
        conn.close()

    return result


@router.post("/api/tutor/mdp/playground/step")
async def tutor_mdp_playground_step(
    body: TutorMDPPlaygroundStepRequest,
    user_id: str = Depends(require_auth),
):
    """Simple MDP playground endpoint.

    Executes a single pedagogical tutor action for a given concept using
    TutorEnvironment + the default pedagogical response tool. This does not
    touch DB state; it is purely for testing the response generator and
    model selection.
    """
    try:
        action = TutorAction[body.pedagogical_action]
    except KeyError:
        raise HTTPException(status_code=400, detail="invalid_pedagogical_action")

    response_tool = make_pedagogical_response_tool()

    env = TutorEnvironment(
        session_id="mdp-playground",
        user_id=user_id,
        concept_id=body.concept_id,
        response_tool=response_tool,
        current_step=None,
        concept_plan=None,
        plan_index=body.plan_index,
        phase=body.phase or "learning",
    )

    with model_override_context(body.model_hint):
        transition = env.step(
            action,
            control_signal=None,
            # For now the playground always represents a single step advance
            # along the current plan, so we tag trigger accordingly.
            context_obs={
                "student_message": body.student_message or "",
                "trigger": "playground_step",
            },
        )

    outputs: Dict[str, Any] = dict(transition.outputs or {})
    debug = outputs.get("debug") or {}
    debug.update(
        {
            "playground": True,
            "requested_action": body.pedagogical_action,
            "plan_index": body.plan_index,
            "plan_length": body.plan_length,
        }
    )
    outputs["debug"] = debug
    return outputs
