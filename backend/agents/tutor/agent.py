"""Thin tutor agent entrypoint.

Validates payload, manages DB connection and model context, delegates per-turn
execution to runtime.orchestrator.run_tutor_turn().
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from core.db import get_db_conn
from llm.common import model_override_context

from .constants import logger
from .utils import normalize_concepts
from .persistence import (
    ensure_session,
    next_turn_index,
)
from .runtime.context import TurnContext
from .runtime.orchestrator import run_tutor_turn


def _validate_uuid(value: Optional[str]) -> Optional[str]:
    """Validate that a string is a valid UUID format, return None if invalid."""
    if not value or not isinstance(value, str):
        return None
    # UUID format: 8-4-4-4-12 hex digits
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    if uuid_pattern.match(value.strip()):
        return value.strip()
    return None


def tutor_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Thin entrypoint: validate payload, manage DB/model context, delegate to orchestrator."""
    # === Payload Validation ===
    message = (payload.get("message") or "").strip()
    # In step-by-step and quiz flows the caller may send a pure control turn
    # (step_control, mcq_answer, and/or confirmed_action like "continue")
    # without free-text. Permit such turns while still requiring a message
    # for normal conversational usage.
    step_control = payload.get("step_control")
    mcq_answer = payload.get("mcq_answer")
    confirmed_action = payload.get("confirmed_action")
    allows_empty = bool(step_control or mcq_answer or confirmed_action)
    if not message and not allows_empty:
        raise ValueError("invalid payload, missing message")

    raw_user_id = (payload.get("user_id") or os.getenv("TEST_USER_ID", "")).strip()
    if not raw_user_id:
        raise ValueError("user_id required (set TEST_USER_ID env var or pass user_id)")

    user_id = _validate_uuid(raw_user_id)
    if not user_id:
        raise ValueError(f"user_id must be a valid UUID format, got: {raw_user_id[:20]}")

    session_id = _validate_uuid(payload.get("session_id"))
    resource_id = _validate_uuid(payload.get("resource_id"))
    target_concepts = normalize_concepts(payload.get("target_concepts"))
    initial_policy = payload.get("session_policy") or {"version": 1, "strategy": "baseline"}
    emit_state_requested = bool(payload.get("emit_state"))
    dry_run = bool(payload.get("dry_run"))
    model_hint = payload.get("model_hint")

    conn = get_db_conn()
    context_manager = model_override_context(model_hint) if model_hint else None

    try:
        if context_manager:
            context_manager.__enter__()
        with conn.cursor() as cur:
            # === Session Setup ===
            session_id = ensure_session(
                cur,
                user_id,
                session_id,
                target_concepts,
                resource_id,
                initial_policy,
            )
            turn_index = next_turn_index(cur, session_id)

            # === Delegate to Runtime Orchestrator ===
            ctx = TurnContext(
                session_id=session_id,
                user_id=user_id,
                turn_index=turn_index,
                message=message,
                target_concepts=target_concepts,
                resource_id=resource_id,
                dry_run=dry_run,
                emit_state_requested=emit_state_requested,
                payload=payload,
            )
            response_payload = run_tutor_turn(ctx, cur)

        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("tutor_agent_failed")
        raise
    finally:
        conn.close()
        if context_manager:
            try:
                context_manager.__exit__(None, None, None)
            except Exception:
                pass

    return response_payload
