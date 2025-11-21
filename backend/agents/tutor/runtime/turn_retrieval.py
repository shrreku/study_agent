from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from ..constants import logger
from ..policy import role_sequence_for_level
from ..persistence import get_last_turn_retrieval
from ..retrieval import rehydrate_chunks_from_ids
from ..decision_engine import ActionDecision
from ..context_model import TutorContext
from .context import RetrievalContext
from .utils import looks_like_confirmation


def run_retrieval_stage(
    *,
    cur: Any,
    session_id: str,
    focus_concept: Optional[str],
    concept_level: str,
    message: str,
    resource_id: Optional[str],
    config,
    agent_action_mode: str,
    confirmed_action: str,
    step_control_type: Optional[str],
    tutor_context: TutorContext,
    action_decision: Optional[ActionDecision],
    srl_mode: bool,
    plan,
) -> Tuple[RetrievalContext, Dict[str, Any], Dict[str, Any]]:
    """Run retrieval stage with optional reuse of previous retrieval.

    Returns (retrieval_context, retrieval_metadata, progress_entry).
    """

    reuse_enabled = os.getenv("TUTOR_RETRIEVAL_REUSE_ENABLED", "true").strip().lower() == "true"
    try:
        min_reuse_chunks = int(os.getenv("TUTOR_RETRIEVAL_MIN_REUSE_CHUNKS", "3"))
    except Exception:
        min_reuse_chunks = 3

    reuse_modes = {"intelligent", "step_by_step", "debug"}
    reuse_allowed_for_mode = config.mode in reuse_modes

    from .retrieval_runtime import run_retrieval  # local import to avoid cycles

    retrieval: RetrievalContext
    reuse_used = False
    retrieval_metadata: Dict[str, Any] = {}
    progress_entry: Dict[str, Any] = {}

    last_retrieval = None
    if reuse_enabled and reuse_allowed_for_mode:
        try:
            last_retrieval = get_last_turn_retrieval(cur, session_id)
        except Exception:
            last_retrieval = None

    if reuse_enabled and reuse_allowed_for_mode and last_retrieval:
        prev_concept = last_retrieval.get("concept")
        prev_source_ids = list(last_retrieval.get("source_chunk_ids") or [])
        prev_metadata = last_retrieval.get("retrieval_metadata") or {}
        prev_query = str(prev_metadata.get("query") or "").strip()
        prev_roles = list(prev_metadata.get("roles") or [])

        same_concept = bool(prev_concept) and prev_concept == focus_concept
        enough_chunks = len(prev_source_ids) >= max(1, min_reuse_chunks)

        is_confirmation = False
        try:
            ts = getattr(tutor_context, "turn_signals", None)
            if ts and getattr(ts, "student_confirmation", "") == "confirmed":
                is_confirmation = True
            else:
                is_confirmation = looks_like_confirmation(message or "")
        except Exception:
            is_confirmation = looks_like_confirmation(message or "")

        is_step_by_step_continue = (
            agent_action_mode == "step_by_step"
            and (
                confirmed_action in {"continue", "next", "yes"}
                or step_control_type == "continue"
            )
        )

        new_query_hint = None
        if isinstance(action_decision, ActionDecision) and action_decision.retrieval_query:
            try:
                new_query_hint = str(action_decision.retrieval_query or "").strip()
            except Exception:
                new_query_hint = None

        query_compatible = (not new_query_hint) or (
            prev_query and new_query_hint and prev_query == new_query_hint
        )

        if same_concept and enough_chunks and (is_confirmation or is_step_by_step_continue) and query_compatible:
            try:
                reused_chunks = rehydrate_chunks_from_ids(prev_source_ids)
            except Exception:
                reused_chunks = []

            if reused_chunks and len(reused_chunks) >= max(1, min_reuse_chunks):
                reuse_used = True
                roles = prev_roles or role_sequence_for_level(concept_level)
                effective_query = prev_query or (focus_concept or message)
                retrieval = RetrievalContext(
                    chunks=reused_chunks,
                    query=effective_query,
                    pedagogy_roles=roles,
                    chunk_ids=prev_source_ids,
                )

                try:
                    logger.info(
                        "tutor_retrieval_reuse_summary turn=%s query=%s focus=%s roles=%s chunk_count=%s",
                        tutor_context.turn_index,
                        effective_query,
                        focus_concept,
                        roles,
                        len(reused_chunks),
                        extra={
                            "session_id": session_id,
                            "user_id": tutor_context.user_id,
                        },
                    )
                except Exception:
                    pass
            else:
                try:
                    logger.info(
                        "tutor_retrieval_fallback_to_hybrid turn=%s reason=%s",
                        tutor_context.turn_index,
                        "insufficient_rehydrated_chunks",
                        extra={
                            "session_id": session_id,
                            "user_id": tutor_context.user_id,
                        },
                    )
                except Exception:
                    pass

    if not reuse_used:
        retrieval = run_retrieval(
            focus_concept=focus_concept,
            message=message,
            resource_id=resource_id,
            concept_level=concept_level,
            action_decision=action_decision,
            policy_decision=None,
            srl_mode=srl_mode,
            plan=plan,
        )

    try:
        progress_entry = {
            "stage": "retrieval",
            "query": retrieval.query,
            "roles": retrieval.pedagogy_roles,
            "count": len(retrieval.chunks),
            "chunk_ids": retrieval.chunk_ids,
        }
        retrieval_metadata = {
            "query": retrieval.query,
            "roles": retrieval.pedagogy_roles,
            "count": len(retrieval.chunks),
            "chunk_ids": retrieval.chunk_ids,
            "reuse": reuse_used,
        }
    except Exception:
        progress_entry = {}
        retrieval_metadata = {}

    try:
        logger.info(
            "tutor_retrieval_summary turn=%s query=%s focus=%s roles=%s chunk_count=%s",
            tutor_context.turn_index,
            retrieval.query,
            focus_concept,
            retrieval.pedagogy_roles,
            len(retrieval.chunks),
            extra={
                "session_id": session_id,
                "user_id": tutor_context.user_id,
            },
        )
    except Exception:
        pass

    return retrieval, retrieval_metadata, progress_entry
