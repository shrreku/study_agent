from __future__ import annotations

from typing import Any, Dict, List, Optional
import os

from core.db import get_db_conn

from .constants import logger
from .state import TutorSessionPolicy
from .classifier import classify_message
from .policy import (
    level_for_mastery,
    needs_cold_start,
    role_sequence_for_level,
    select_focus_concept,
)
from .responses import (
    build_cold_start_question,
    build_hint_response,
    build_reflect_response,
    build_followup_question,
    generate_explain_response,
)
from .knowledge import fetch_mastery_map, fetch_prereq_chain, record_cold_start
from .retrieval import retrieve_chunks
from .utils import normalize_concepts
from .persistence import (
    ensure_session,
    get_session_state,
    insert_turn,
    next_turn_index,
    update_session,
)


def tutor_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    message = (payload.get("message") or "").strip()
    if not message:
        raise ValueError("invalid payload, missing message")

    user_id = (payload.get("user_id") or os.getenv("TEST_USER_ID", "")).strip()
    if not user_id:
        raise ValueError("user_id required (set TEST_USER_ID env var or pass user_id)")

    session_id = (payload.get("session_id") or "").strip() or None
    resource_id = (payload.get("resource_id") or "").strip() or None
    target_concepts = normalize_concepts(payload.get("target_concepts"))
    initial_policy = payload.get("session_policy") or {"version": 1, "strategy": "baseline"}

    conn = get_db_conn()
    response_payload: Dict[str, Any]
    try:
        with conn.cursor() as cur:
            session_id = ensure_session(
                cur,
                user_id,
                session_id,
                target_concepts,
                resource_id,
                initial_policy,
            )
            session_state = get_session_state(cur, session_id)
            turn_index = next_turn_index(cur, session_id)

            policy_state = TutorSessionPolicy.from_dict(session_state.get("policy"))

            classification = classify_message(
                message,
                target_concepts or session_state.get("target_concepts", []),
                session_state.get("last_concept"),
            )

            learning_targets = target_concepts or session_state.get("target_concepts", [])
            mastery_map = fetch_mastery_map(cur, user_id)
            learning_path = fetch_prereq_chain(
                [classification.get("concept")] + list(learning_targets)
                if classification.get("concept")
                else list(learning_targets)
            )

            focus_concept = select_focus_concept(
                classification,
                learning_path,
                mastery_map,
                learning_targets,
            )
            concept_level = level_for_mastery((mastery_map.get(focus_concept) or {}).get("mastery"))

            logger.info(
                "tutor_policy_stage",
                extra={
                    "session_id": session_id,
                    "user_id": user_id,
                    "turn_index": turn_index,
                    "classification": classification,
                    "focus_concept": focus_concept,
                    "concept_level": concept_level,
                    "learning_path": learning_path,
                    "targets": learning_targets,
                },
            )

            role_sequence = role_sequence_for_level(concept_level)

            if focus_concept:
                chunks = retrieve_chunks(focus_concept, resource_id, role_sequence)
                if not chunks:
                    chunks = retrieve_chunks(message, resource_id, role_sequence)
            else:
                chunks = retrieve_chunks(message, resource_id, role_sequence)

            logger.info(
                "tutor_retrieval_summary",
                extra={
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "query": message,
                    "resource_id": resource_id,
                    "focus_concept": focus_concept,
                    "pedagogy_roles": role_sequence,
                    "chunk_ids": [c.get("id") for c in chunks],
                },
            )

            affect = classification.get("affect", "neutral")
            intent = classification.get("intent", "unknown")
            mastery_delta = payload.get("mastery_delta")

            action_type = "explain"
            cold_start_triggered = False
            inference_concept: Optional[str] = focus_concept
            response_text = ""
            confidence = 0.0
            source_chunk_ids: List[str] = []

            if needs_cold_start(focus_concept, mastery_map, policy_state):
                cold_start_triggered = True
                action_type = "ask"
                response_text, confidence, source_chunk_ids = build_cold_start_question(
                    focus_concept,
                    chunks,
                )
                record_cold_start(cur, session_id, focus_concept or "")
                policy_state.mark_cold_start(focus_concept)
                logger.info(
                    "tutor_action_decision",
                    extra={
                        "session_id": session_id,
                        "turn_index": turn_index,
                        "action_type": action_type,
                        "focus_concept": focus_concept,
                        "cause": "cold_start",
                        "confidence": confidence,
                        "chunk_ids": source_chunk_ids,
                    },
                )
            else:
                # Check if we should assess understanding instead of explaining again
                should_assess = (
                    (intent == "reflection" and affect == "engaged" and chunks) or
                    (policy_state.consecutive_explains >= 2 and chunks)
                )

                if should_assess:
                    action_type = "ask"
                    response_text, confidence, source_chunk_ids = build_followup_question(
                        focus_concept,
                        concept_level,
                        chunks,
                    )
                    cause = "reflection_claimed_understanding" if intent == "reflection" else "consecutive_explains_cap"
                    logger.info(
                        "tutor_action_decision",
                        extra={
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "action_type": action_type,
                            "focus_concept": focus_concept,
                            "cause": cause,
                            "confidence": confidence,
                            "chunk_ids": source_chunk_ids,
                            "consecutive_explains": policy_state.consecutive_explains,
                        },
                    )
                elif intent == "answer" and chunks:
                    action_type = "reflect"
                    response_text, confidence, source_chunk_ids = build_reflect_response(
                        focus_concept,
                        concept_level,
                        chunks,
                    )
                    logger.info(
                        "tutor_action_decision",
                        extra={
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "action_type": action_type,
                            "focus_concept": focus_concept,
                            "cause": "student_answer",
                            "confidence": confidence,
                            "chunk_ids": source_chunk_ids,
                        },
                    )
                elif affect in {"confused", "unsure"} and chunks:
                    action_type = "explain"
                    (
                        response_text,
                        confidence,
                        source_chunk_ids,
                        inferred_concept_candidate,
                    ) = generate_explain_response(
                        focus_concept,
                        concept_level,
                        chunks,
                    )
                    inference_concept = inferred_concept_candidate or focus_concept
                    logger.info(
                        "tutor_action_decision",
                        extra={
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "action_type": action_type,
                            "focus_concept": focus_concept,
                            "cause": "affect_confused_explain_basics",
                            "confidence": confidence,
                            "chunk_ids": source_chunk_ids,
                        },
                    )
                else:
                    (
                        response_text,
                        confidence,
                        source_chunk_ids,
                        inferred_concept_candidate,
                    ) = generate_explain_response(
                        focus_concept,
                        concept_level,
                        chunks,
                    )
                    inference_concept = inferred_concept_candidate or focus_concept
                    logger.info(
                        "tutor_action_decision",
                        extra={
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "action_type": action_type,
                            "focus_concept": focus_concept,
                            "cause": "explain_default",
                            "confidence": confidence,
                            "chunk_ids": source_chunk_ids,
                        },
                    )

            policy_state.learning_path = learning_path
            policy_state.focus_concept = focus_concept
            policy_state.focus_level = concept_level
            policy_state.cold_start = cold_start_triggered
            policy_state.update_action(action_type)

            turn_id = insert_turn(
                cur,
                session_id,
                turn_index,
                message,
                intent,
                affect,
                inference_concept,
                action_type,
                response_text,
                source_chunk_ids,
                confidence,
                mastery_delta,
            )

            update_session(
                cur,
                session_id,
                inference_concept,
                action_type,
                policy_state.to_dict(),
            )

            response_payload = {
                "session_id": session_id,
                "turn_id": turn_id,
                "turn_index": turn_index,
                "response": response_text,
                "action_type": action_type,
                "source_chunk_ids": source_chunk_ids,
                "confidence": confidence,
                "intent": intent,
                "affect": affect,
                "concept": inference_concept,
                "level": concept_level,
                "learning_path": learning_path,
                "cold_start": cold_start_triggered,
                "classification_confidence": classification.get("confidence"),
            }

            logger.info(
                "tutor_turn_committed",
                extra={
                    "session_id": session_id,
                    "turn_id": response_payload.get("turn_id"),
                    "turn_index": turn_index,
                    "user_id": user_id,
                    "action_type": action_type,
                    "intent": intent,
                    "affect": affect,
                    "concept": inference_concept,
                    "confidence": confidence,
                    "cold_start": cold_start_triggered,
                },
            )

        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("tutor_agent_failed")
        raise
    finally:
        conn.close()

    return response_payload
