from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import re

from core.db import get_db_conn

from .constants import logger
from .state import TutorSessionPolicy
from .classifier import classify_message
from .policy import (
    level_for_mastery,
    needs_cold_start,
    role_sequence_for_level,
    select_focus_concept_with_prereqs,
)
from .responses import (
    build_cold_start_question,
    build_hint_response,
    build_reflect_response,
    build_followup_question,
    build_override_question,
    build_review_response,
    build_worked_example_response,
    build_prerequisite_review_prompt,
    generate_explain_response,
    generate_explain_response_with_plan,
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
from .tools.mastery_updater import MasteryUpdater
from .tools.example_generator import ExampleGenerator, ExampleRequest
from .validators.assessment import assess_student_response
from .planning import TutorPlanner, TutorPlan
from .self_critique import SelfCritic
from .srl_executor import execute_plan_steps
from .observation import build_observation


@dataclass
class ActionResult:
    """Result of an action handler containing all response details."""
    action_type: str
    response_text: str
    confidence: float
    source_chunk_ids: List[str]
    inference_concept: Optional[str]
    action_params: Dict[str, Any]
    cold_start_triggered: bool = False


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _extract_pedagogy_role(chunk: Dict[str, Any]) -> Optional[str]:
    role = chunk.get("pedagogy_role")
    if role:
        return role
    tags = chunk.get("tags")
    if isinstance(tags, dict):
        candidate = tags.get("pedagogy_role") or tags.get("content_type")
        if candidate:
            return candidate
    return None

 


def _handle_cold_start(
    focus_concept: Optional[str],
    concept_level: str,
    chunks: List[Dict[str, Any]],
    cur: Any,
    session_id: str,
    policy_state: TutorSessionPolicy,
) -> ActionResult:
    """Handle cold start scenario for new concepts.

    Prefer a brief grounded explanation when context is available.
    Fall back to a light diagnostic question only when no context is retrieved.
    """
    record_cold_start(cur, session_id, focus_concept or "")
    policy_state.mark_cold_start(focus_concept)

    action_params = {k: v for k, v in {
        "concept": focus_concept,
        "level": concept_level,
        "mode": "cold_start",
    }.items() if v}

    if chunks:
        # Grounded micro-explanation to kick off the session
        response_text, confidence, source_chunk_ids, inferred = generate_explain_response(
            focus_concept,
            concept_level,
            chunks,
        )
        return ActionResult(
            action_type="explain",
            response_text=response_text,
            confidence=confidence,
            source_chunk_ids=source_chunk_ids,
            inference_concept=inferred or focus_concept,
            action_params=action_params,
            cold_start_triggered=True,
        )
    else:
        # No context available yet: ask a minimal diagnostic question
        response_text, confidence, source_chunk_ids = build_cold_start_question(
            focus_concept,
            chunks,
        )
        return ActionResult(
            action_type="ask",
            response_text=response_text,
            confidence=confidence,
            source_chunk_ids=source_chunk_ids,
            inference_concept=focus_concept,
            action_params=action_params,
            cold_start_triggered=True,
        )


def _handle_override(
    override_type: str,
    focus_concept: Optional[str],
    concept_level: str,
    chunks: List[Dict[str, Any]],
    override_params: Dict[str, Any],
) -> ActionResult:
    """Handle explicit action override requests."""
    concept_for_override = override_params.get("concept") or focus_concept
    level_for_override = override_params.get("level") or concept_level
    difficulty = override_params.get("difficulty")
    question_type = override_params.get("question_type")

    action_params = {
        key: value
        for key, value in {
            "concept": concept_for_override,
            "level": level_for_override,
            "difficulty": difficulty,
            "question_type": question_type,
        }.items()
        if value
    }

    if override_type == "ask":
        response_text, confidence, source_chunk_ids = build_override_question(
            concept_for_override,
            level_for_override,
            chunks,
            difficulty=difficulty,
            question_type=question_type,
        )
        inference_concept = concept_for_override
    elif override_type == "hint":
        response_text, confidence, source_chunk_ids = build_hint_response(
            concept_for_override,
            level_for_override,
            chunks,
        )
        inference_concept = concept_for_override
    elif override_type == "reflect":
        response_text, confidence, source_chunk_ids = build_reflect_response(
            concept_for_override,
            level_for_override,
            chunks,
        )
        inference_concept = concept_for_override
    elif override_type == "worked_example":
        response_text, confidence, source_chunk_ids = build_worked_example_response(
            concept_for_override,
            level_for_override,
            chunks,
        )
        inference_concept = concept_for_override
    elif override_type == "review":
        response_text, confidence, source_chunk_ids = build_review_response(
            concept_for_override,
            level_for_override,
            chunks,
        )
        inference_concept = concept_for_override
    else:
        # Default to explain for unknown override types
        (
            response_text,
            confidence,
            source_chunk_ids,
            inferred_concept_candidate,
        ) = generate_explain_response(
            concept_for_override,
            level_for_override,
            chunks,
        )
        inference_concept = inferred_concept_candidate or concept_for_override
        override_type = "explain"

    return ActionResult(
        action_type=override_type,
        response_text=response_text,
        confidence=confidence,
        source_chunk_ids=source_chunk_ids,
        inference_concept=inference_concept,
        action_params=action_params,
    )


def _handle_assessment(
    focus_concept: Optional[str],
    concept_level: str,
    chunks: List[Dict[str, Any]],
    cause: str,
) -> ActionResult:
    """Handle assessment questions after consecutive explains or student reflection."""
    response_text, confidence, source_chunk_ids = build_followup_question(
        focus_concept,
        concept_level,
        chunks,
    )
    
    action_params = {
        key: value
        for key, value in {
            "concept": focus_concept,
            "level": concept_level,
            "mode": "assessment",
        }.items()
        if value
    }
    
    return ActionResult(
        action_type="ask",
        response_text=response_text,
        confidence=confidence,
        source_chunk_ids=source_chunk_ids,
        inference_concept=focus_concept,
        action_params=action_params,
    )


def _handle_reflection(
    focus_concept: Optional[str],
    concept_level: str,
    chunks: List[Dict[str, Any]],
) -> ActionResult:
    """Handle student answer with reflection prompt."""
    response_text, confidence, source_chunk_ids = build_reflect_response(
        focus_concept,
        concept_level,
        chunks,
    )
    
    action_params = {
        key: value
        for key, value in {
            "concept": focus_concept,
            "level": concept_level,
            "mode": "reflection",
        }.items()
        if value
    }
    
    return ActionResult(
        action_type="reflect",
        response_text=response_text,
        confidence=confidence,
        source_chunk_ids=source_chunk_ids,
        inference_concept=focus_concept,
        action_params=action_params,
    )


def _handle_explain(
    focus_concept: Optional[str],
    concept_level: str,
    chunks: List[Dict[str, Any]],
    mode: str = "default",
) -> ActionResult:
    """Handle explanation response (default or for confusion)."""
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
    
    action_params = {
        key: value
        for key, value in {
            "concept": inference_concept,
            "level": concept_level,
            "mode": mode,
        }.items()
        if value
    }
    
    return ActionResult(
        action_type="explain",
        response_text=response_text,
        confidence=confidence,
        source_chunk_ids=source_chunk_ids,
        inference_concept=inference_concept,
        action_params=action_params,
    )


 


def tutor_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    message = (payload.get("message") or "").strip()
    if not message:
        raise ValueError("invalid payload, missing message")

    raw_user_id = (payload.get("user_id") or os.getenv("TEST_USER_ID", "")).strip()
    if not raw_user_id:
        raise ValueError("user_id required (set TEST_USER_ID env var or pass user_id)")
    
    # Validate UUIDs - reject invalid formats to prevent database errors
    user_id = _validate_uuid(raw_user_id)
    if not user_id:
        raise ValueError(f"user_id must be a valid UUID format, got: {raw_user_id[:20]}")
    
    session_id = _validate_uuid(payload.get("session_id"))
    resource_id = _validate_uuid(payload.get("resource_id"))
    target_concepts = normalize_concepts(payload.get("target_concepts"))
    initial_policy = payload.get("session_policy") or {"version": 1, "strategy": "baseline"}
    emit_state_requested = bool(payload.get("emit_state"))
    
    # Extract model hint for this agent call
    model_hint = payload.get("model_hint")
    
    # Import model_override_context
    from llm.common import model_override_context

    conn = get_db_conn()
    response_payload: Dict[str, Any]
    
    # Use context manager if model_hint is provided
    context_manager = model_override_context(model_hint) if model_hint else None
    
    try:
        if context_manager:
            context_manager.__enter__()
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
            progress: List[Dict[str, Any]] = []

            classification = classify_message(
                message,
                target_concepts or session_state.get("target_concepts", []),
                session_state.get("last_concept"),
            )
            try:
                progress.append({
                    "stage": "classification",
                    "intent": classification.get("intent"),
                    "affect": classification.get("affect"),
                    "concept": classification.get("concept"),
                    "confidence": classification.get("confidence"),
                })
            except Exception:
                pass

            learning_targets = target_concepts or session_state.get("target_concepts", [])
            mastery_map = fetch_mastery_map(cur, user_id)
            learning_path = fetch_prereq_chain(
                [classification.get("concept")] + list(learning_targets)
                if classification.get("concept")
                else list(learning_targets)
            )

            enable_prereq_check = (os.getenv("TUTOR_PREREQ_CHECK_ENABLED", "true").strip().lower() == "true")
            focus_concept, prereq_check = select_focus_concept_with_prereqs(
                classification,
                learning_path,
                mastery_map,
                learning_targets,
                user_id,
                enable_prereq_check=enable_prereq_check,
            )
            concept_level = level_for_mastery((mastery_map.get(focus_concept) or {}).get("mastery"))

            logger.info(
                "tutor_prereq_check",
                extra={
                    "session_id": session_id,
                    "target_concept": classification.get("concept"),
                    "focus_concept": focus_concept,
                    "prereq_ready": getattr(prereq_check, "ready", True),
                    "missing_prereqs": getattr(prereq_check, "missing_prereqs", []),
                    "recommendation": getattr(prereq_check, "recommendation", ""),
                },
            )

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

            # SRL mode flags
            srl_mode = (os.getenv("TUTOR_SRL_MODE", "false").strip().lower() == "true") and (
                os.getenv("TUTOR_SRL_PLANNING_ENABLED", "true").strip().lower() == "true"
            )

            plan: Optional[TutorPlan] = None
            if srl_mode:
                # Stage 2: Planning (Internal reasoning)
                planner = TutorPlanner(enable_srl_mode=True)
                plan = planner.generate_plan(
                    observation={
                        "user": {"message": message},
                        "classifier": classification,
                        "tutor": {
                            "focus_concept": focus_concept,
                            "concept_level": concept_level,
                        },
                        "policy": policy_state.to_dict(),
                    },
                    student_state={
                        "mastery_map": mastery_map,
                        "learning_path": learning_path,
                    },
                    available_actions=["explain", "ask", "hint", "reflect", "review"],
                )
                if os.getenv("TUTOR_SRL_LOG_THINKING", "true").strip().lower() == "true":
                    try:
                        logger.info(
                            "tutor_srl_planning",
                            extra={
                                "session_id": session_id,
                                "turn_index": turn_index,
                                "thinking": (plan.thinking or "")[:200],
                                "intended_action": plan.intended_action,
                                "confidence": plan.confidence,
                                "assumptions": plan.assumptions,
                                "risks": plan.risks,
                            },
                        )
                    except Exception:
                        pass
                try:
                    progress.append({
                        "stage": "planning",
                        "intended_action": getattr(plan, "intended_action", None),
                        "confidence": getattr(plan, "confidence", None),
                    })
                except Exception:
                    pass

            role_sequence = role_sequence_for_level(concept_level)

            # Stage 3: Retrieval (guided by plan if available)
            if srl_mode and plan:
                pedagogy_roles = list(plan.pedagogy_focus or []) or role_sequence
                query = plan.retrieval_query or (focus_concept or message)
                chunks = retrieve_chunks(query, resource_id, pedagogy_roles)
                if not chunks and focus_concept:
                    chunks = retrieve_chunks(focus_concept, resource_id, pedagogy_roles)
                if not chunks:
                    chunks = retrieve_chunks(message, resource_id, pedagogy_roles)
            else:
                if focus_concept:
                    chunks = retrieve_chunks(focus_concept, resource_id, role_sequence)
                    if not chunks:
                        chunks = retrieve_chunks(message, resource_id, role_sequence)
                else:
                    chunks = retrieve_chunks(message, resource_id, role_sequence)

            try:
                roles_for_trace = (list(getattr(plan, "pedagogy_focus", []) or []) if (srl_mode and plan) else role_sequence)
                progress.append({
                    "stage": "retrieval",
                    "query": (locals().get("query") or focus_concept or message),
                    "roles": roles_for_trace,
                    "count": len(chunks or []),
                    "chunk_ids": [c.get("id") for c in (chunks or []) if c.get("id")],
                })
            except Exception:
                pass

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

            enable_mastery_update = (os.getenv("TUTOR_MASTERY_REALTIME_UPDATE", "false").strip().lower() == "true")
            mastery_updater: Optional[MasteryUpdater] = None
            if enable_mastery_update:
                try:
                    lr = float(os.getenv("TUTOR_MASTERY_LEARNING_RATE", "0.1") or 0.1)
                except Exception:
                    lr = 0.1
                try:
                    df = float(os.getenv("TUTOR_MASTERY_DECAY_FACTOR", "0.95") or 0.95)
                except Exception:
                    df = 0.95
                try:
                    mn = float(os.getenv("TUTOR_MASTERY_MIN_UPDATE", "0.05") or 0.05)
                except Exception:
                    mn = 0.05
                try:
                    mx = float(os.getenv("TUTOR_MASTERY_MAX_UPDATE", "0.3") or 0.3)
                except Exception:
                    mx = 0.3
                mastery_updater = MasteryUpdater(
                    learning_rate=lr,
                    decay_factor=df,
                    min_update=mn,
                    max_update=mx,
                )

                # Optionally export SRL reasoning into observation for RL datasets
                try:
                    if (locals().get("srl_mode") is True) and (os.getenv("TUTOR_RL_EXPORT_REASONING", "false").strip().lower() == "true"):
                        srl_blob: Dict[str, Any] = {}
                        if locals().get("plan") is not None:
                            p = plan  # type: ignore[assignment]
                            srl_blob["plan"] = {
                                "thinking": p.thinking,
                                "intended_action": p.intended_action,
                                "rationale": p.action_rationale,
                                "confidence": p.confidence,
                                "assumptions": p.assumptions,
                                "risks": p.risks,
                                "steps": getattr(p, "steps", None),
                                "target_sequence": getattr(p, "target_sequence", None),
                            }
                        if locals().get("critique") is not None and critique is not None:
                            srl_blob["critique"] = {
                                "quality": getattr(critique, "overall_quality", None),
                                "issues": getattr(critique, "issues_found", None),
                                "suggestions": getattr(critique, "suggestions", None),
                                "should_revise": getattr(critique, "should_revise", None),
                            }
                        if srl_blob:
                            try:
                                response_payload["observation"]["srl"] = srl_blob
                            except Exception:
                                pass
                except Exception:
                    pass

            override = payload.get("action_override") or {}
            override_type: Optional[str] = None
            override_params: Dict[str, Any] = {}
            if isinstance(override, dict):
                raw_type = override.get("type")
                if isinstance(raw_type, str):
                    override_type = raw_type.strip() or None
                params = override.get("params")
                if isinstance(params, dict):
                    allowed_keys = {"concept", "level", "difficulty", "question_type"}
                    override_params = {
                        key: value
                        for key, value in params.items()
                        if key in allowed_keys and value not in (None, "")
                    }

            requested_override_type = override_type
            applied_override_type: Optional[str] = None

            # Decision tree: use handler functions to determine action
            result: ActionResult
            cause: str = "default"
            
            if (
                (locals().get("prereq_check") is not None)
                and getattr(prereq_check, "should_review", False)
                and getattr(prereq_check, "missing_prereqs", [])
                and override_type is None
            ):
                # Prerequisite gating: review missing prerequisite before proceeding
                prereq_concept = prereq_check.missing_prereqs[0]
                prereq_level = level_for_mastery((mastery_map.get(prereq_concept) or {}).get("mastery"))
                prereq_chunks = retrieve_chunks(prereq_concept, resource_id, ["definition", "explanation"])
                (
                    response_text,
                    confidence,
                    source_chunk_ids,
                ) = build_prerequisite_review_prompt(
                    target_concept=classification.get("concept"),
                    missing_prereqs=prereq_check.missing_prereqs,
                    chunks=prereq_chunks,
                )
                result = ActionResult(
                    action_type="explain",
                    response_text=response_text,
                    confidence=confidence,
                    source_chunk_ids=source_chunk_ids,
                    inference_concept=prereq_concept,
                    action_params={
                        "concept": prereq_concept,
                        "level": prereq_level,
                        "mode": "prereq_review",
                    },
                    cold_start_triggered=False,
                )
                cause = "prereq_gating"

            elif (
                needs_cold_start(focus_concept, mastery_map, policy_state)
                and override_type is None
                and (os.getenv("TUTOR_COLD_START_ENABLED", "true").strip().lower() == "true")
            ):
                # Cold start: begin with grounded micro-explanation when possible
                result = _handle_cold_start(focus_concept, concept_level, chunks, cur, session_id, policy_state)
                cause = "cold_start"
                
            elif override_type:
                # Explicit override: force specific action type
                result = _handle_override(override_type, focus_concept, concept_level, chunks, override_params)
                applied_override_type = result.action_type if result.action_type == override_type else None
                cause = "override_request"
                
            else:
                # Normal flow: choose action based on student state
                should_assess = (
                    (intent == "reflection" and affect == "engaged" and chunks) or
                    (policy_state.consecutive_explains >= 2 and chunks)
                )
                
                if should_assess:
                    # Follow-up question after explains or claimed understanding
                    cause = "reflection_claimed_understanding" if intent == "reflection" else "consecutive_explains_cap"
                    result = _handle_assessment(focus_concept, concept_level, chunks, cause)
                    
                elif intent == "answer" and chunks:
                    # Student provided answer: prompt reflection
                    result = _handle_reflection(focus_concept, concept_level, chunks)
                    cause = "student_answer"
                    
                elif srl_mode and plan:
                    # SRL-guided execution stage
                    multi = os.getenv("TUTOR_SRL_MULTI_STEP_EXECUTE", "false").strip().lower() in ("1", "true", "yes")
                    if multi:
                        combined = execute_plan_steps(plan, focus_concept, concept_level, message, resource_id)
                        # Replace chunks with the union used across steps for observation
                        chunks = combined.get("chunks", [])
                        try:
                            for sp in combined.get("step_progress", []) or []:
                                progress.append(sp)
                        except Exception:
                            pass
                        result = ActionResult(
                            action_type=str(combined.get("last_action") or "explain"),
                            response_text=str(combined.get("text") or ""),
                            confidence=float(combined.get("confidence") or 0.6),
                            source_chunk_ids=list(combined.get("source_chunk_ids") or []),
                            inference_concept=combined.get("inference_concept") or focus_concept,
                            action_params={
                                "concept": (combined.get("inference_concept") or focus_concept),
                                "level": concept_level,
                                "mode": "srl_plan_multi",
                            },
                        )
                        cause = "srl_plan_multi_steps"
                    else:
                        intended = (plan.intended_action or "").lower()
                        if intended == "explain":
                            (
                                response_text,
                                confidence,
                                source_chunk_ids,
                                inferred_concept_candidate,
                            ) = generate_explain_response_with_plan(
                                plan=plan,
                                concept=focus_concept,
                                level=concept_level,
                                chunks=chunks,
                            )
                            inference_concept = inferred_concept_candidate or focus_concept
                            result = ActionResult(
                                action_type="explain",
                                response_text=response_text,
                                confidence=confidence,
                                source_chunk_ids=source_chunk_ids,
                                inference_concept=inference_concept,
                                action_params={
                                    "concept": inference_concept or focus_concept,
                                    "level": concept_level,
                                    "mode": "srl_plan",
                                },
                            )
                            cause = "srl_plan_explain"
                        elif intended == "ask":
                            response_text, confidence, source_chunk_ids = build_followup_question(
                                focus_concept,
                                concept_level,
                                chunks,
                            )
                            result = ActionResult(
                                action_type="ask",
                                response_text=response_text,
                                confidence=confidence,
                                source_chunk_ids=source_chunk_ids,
                                inference_concept=focus_concept,
                                action_params={
                                    "concept": focus_concept,
                                    "level": concept_level,
                                    "mode": "srl_plan",
                                },
                            )
                            cause = "srl_plan_ask"
                        elif intended == "hint":
                            response_text, confidence, source_chunk_ids = build_hint_response(
                                focus_concept,
                                concept_level,
                                chunks,
                            )
                            result = ActionResult(
                                action_type="hint",
                                response_text=response_text,
                                confidence=confidence,
                                source_chunk_ids=source_chunk_ids,
                                inference_concept=focus_concept,
                                action_params={
                                    "concept": focus_concept,
                                    "level": concept_level,
                                    "mode": "srl_plan",
                                },
                            )
                            cause = "srl_plan_hint"
                        elif intended == "reflect":
                            response_text, confidence, source_chunk_ids = build_reflect_response(
                                focus_concept,
                                concept_level,
                                chunks,
                            )
                            result = ActionResult(
                                action_type="reflect",
                                response_text=response_text,
                                confidence=confidence,
                                source_chunk_ids=source_chunk_ids,
                                inference_concept=focus_concept,
                                action_params={
                                    "concept": focus_concept,
                                    "level": concept_level,
                                    "mode": "srl_plan",
                                },
                            )
                            cause = "srl_plan_reflect"
                        elif intended == "review":
                            response_text, confidence, source_chunk_ids = build_review_response(
                                focus_concept,
                                concept_level,
                                chunks,
                            )
                            result = ActionResult(
                                action_type="review",
                                response_text=response_text,
                                confidence=confidence,
                                source_chunk_ids=source_chunk_ids,
                                inference_concept=focus_concept,
                                action_params={
                                    "concept": focus_concept,
                                    "level": concept_level,
                                    "mode": "srl_plan",
                                },
                            )
                            cause = "srl_plan_review"
                        else:
                            result = _handle_explain(focus_concept, concept_level, chunks, mode="default")
                            cause = "srl_plan_default"

                elif (
                    os.getenv("TUTOR_EXAMPLE_GENERATION_ENABLED", "false").strip().lower() == "true"
                    and focus_concept
                    and learning_path
                ):
                    try:
                        idx = learning_path.index(focus_concept)
                    except ValueError:
                        idx = -1
                    from_concept = None
                    if idx > 0:
                        for c in reversed(learning_path[:idx]):
                            try:
                                m = float((mastery_map.get(c) or {}).get("mastery", 0.0) or 0.0)
                            except Exception:
                                m = 0.0
                            if m >= 0.5:
                                from_concept = c
                                break
                        if not from_concept:
                            from_concept = learning_path[idx - 1]
                    if from_concept:
                        try:
                            gen = ExampleGenerator()
                            br = gen.generate_bridge_example(
                                from_concept=from_concept,
                                to_concept=focus_concept,
                                student_level=concept_level,
                                grounding_chunks=chunks,
                            )
                            min_rel = getattr(gen, "min_relevance", 0.6)
                            min_conf = getattr(gen, "min_confidence", 0.5)
                            if br.relevance_score >= min_rel and br.confidence >= min_conf:
                                text = f"Example: {br.example_text}\n\nWhy this helps: {br.explanation}"
                                result = ActionResult(
                                    action_type="explain",
                                    response_text=text,
                                    confidence=float(br.confidence),
                                    source_chunk_ids=[c.get("id") for c in chunks if c.get("id")],
                                    inference_concept=focus_concept,
                                    action_params={
                                        "concept": focus_concept,
                                        "level": concept_level,
                                        "mode": "bridge_example",
                                    },
                                )
                                cause = "bridge_example"
                            else:
                                result = _handle_explain(focus_concept, concept_level, chunks, mode="confusion_support")
                                cause = "affect_confused_explain_basics"
                        except Exception:
                            logger.exception("tutor_bridge_example_failed")
                            result = _handle_explain(focus_concept, concept_level, chunks, mode="confusion_support")
                            cause = "affect_confused_explain_basics"
                    else:
                        result = _handle_explain(focus_concept, concept_level, chunks, mode="confusion_support")
                        cause = "affect_confused_explain_basics"

                elif affect in {"confused", "unsure"} and chunks:
                    # Confusion detected: explain basics
                    result = _handle_explain(focus_concept, concept_level, chunks, mode="confusion_support")
                    cause = "affect_confused_explain_basics"
                    
                else:
                    # Default: explain the concept
                    result = _handle_explain(focus_concept, concept_level, chunks, mode="default")
                    cause = "explain_default"
            
            # Unpack result
            action_type = result.action_type
            response_text = result.response_text
            confidence = result.confidence
            source_chunk_ids = result.source_chunk_ids
            inference_concept = result.inference_concept
            final_action_params = result.action_params
            cold_start_triggered = result.cold_start_triggered

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
            try:
                progress.append({
                    "stage": "decision",
                    "action_type": action_type,
                    "cause": cause,
                    "confidence": confidence,
                })
            except Exception:
                pass

            critique = None
            if (
                (locals().get("srl_mode") is True)
                and (os.getenv("TUTOR_SRL_SELF_CRITIQUE", "false").strip().lower() == "true")
                and (locals().get("plan") is not None)
            ):
                try:
                    critic = SelfCritic()
                    obs_for_critic = {"tutor": {"concept_level": concept_level}}
                    critique = critic.critique_response(plan, response_text, obs_for_critic)
                    if os.getenv("TUTOR_SRL_LOG_CRITIQUE", "true").strip().lower() == "true":
                        try:
                            logger.info(
                                "tutor_srl_critique",
                                extra={
                                    "session_id": session_id,
                                    "turn_index": turn_index,
                                    "quality": getattr(critique, "overall_quality", None),
                                    "should_revise": getattr(critique, "should_revise", None),
                                    "issues": getattr(critique, "issues_found", None),
                                },
                            )
                        except Exception:
                            pass
                except Exception:
                    logger.exception("tutor_self_critique_failed")
            try:
                if critique is not None:
                    progress.append({
                        "stage": "critique",
                        "quality": getattr(critique, "overall_quality", None),
                        "should_revise": getattr(critique, "should_revise", None),
                    })
            except Exception:
                pass

            policy_state.learning_path = learning_path
            policy_state.focus_concept = focus_concept
            policy_state.focus_level = concept_level
            policy_state.cold_start = cold_start_triggered
            policy_state.update_action(action_type)

            if not final_action_params:
                base_params = {
                    "concept": inference_concept or focus_concept,
                    "level": concept_level,
                }
                final_action_params = {k: v for k, v in base_params.items() if v}

            if mastery_updater and (inference_concept or focus_concept):
                target_concept = inference_concept or focus_concept
                try:
                    current_mastery = float((mastery_map.get(target_concept) or {}).get("mastery", 0.0) or 0.0)
                except Exception:
                    current_mastery = 0.0
                # Optionally evaluate student's response to derive correctness and quality signals
                ans_correct = payload.get("answer_correct")
                expl_quality = payload.get("explanation_quality")
                if intent in {"answer", "reflection", "explanation"}:
                    try:
                        assess = assess_student_response(
                            student_message=message,
                            expected_concept=target_concept or "",
                            reference_chunks=chunks,
                        )
                        if isinstance(assess, dict):
                            if assess.get("correct") is not None:
                                ans_correct = bool(assess.get("correct"))
                            try:
                                qv = assess.get("quality")
                                if qv is not None:
                                    expl_quality = float(qv)
                            except Exception:
                                pass
                    except Exception:
                        pass
                interaction_signals = {
                    "affect": affect,
                    "intent": intent,
                    "classification_confidence": classification.get("confidence"),
                    "answer_correct": ans_correct,
                    "explanation_quality": expl_quality,
                }
                update = mastery_updater.compute_mastery_delta(
                    concept=target_concept,
                    user_id=user_id,
                    interaction_signals=interaction_signals,
                    current_mastery=current_mastery,
                )
                if update.delta != 0.0:
                    new_mastery = mastery_updater.apply_update(
                        user_id=user_id,
                        update=update,
                        db_cursor=cur,
                    )
                    if target_concept in mastery_map:
                        try:
                            mastery_map[target_concept]["mastery"] = float(new_mastery)
                        except Exception:
                            mastery_map[target_concept]["mastery"] = new_mastery
                    mastery_delta = update.delta

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
                "progress": progress,
            }

            # Include SRL artifacts on the response payload (non-schema-critical)
            if locals().get("srl_mode") is True:
                try:
                    if locals().get("plan") is not None:
                        response_payload["srl_plan"] = {
                            "thinking": plan.thinking,  # type: ignore[attr-defined]
                            "rationale": plan.action_rationale,  # type: ignore[attr-defined]
                            "confidence": plan.confidence,  # type: ignore[attr-defined]
                            "assumptions": plan.assumptions,  # type: ignore[attr-defined]
                            "risks": plan.risks,  # type: ignore[attr-defined]
                            "steps": getattr(plan, "steps", None),  # type: ignore[attr-defined]
                            "target_sequence": getattr(plan, "target_sequence", None),  # type: ignore[attr-defined]
                        }
                    if locals().get("critique") is not None and critique is not None:
                        response_payload["srl_critique"] = {
                            "quality": getattr(critique, "overall_quality", None),
                            "issues": getattr(critique, "issues_found", None),
                            "suggestions": getattr(critique, "suggestions", None),
                            "should_revise": getattr(critique, "should_revise", None),
                        }
                except Exception:
                    pass

            if emit_state_requested:
                response_payload["observation"] = build_observation(
                    message=message,
                    user_id=user_id,
                    learning_targets=learning_targets,
                    classification=classification,
                    focus_concept=focus_concept,
                    concept_level=concept_level,
                    inference_concept=inference_concept,
                    learning_path=learning_path,
                    mastery_map=mastery_map,
                    chunks=chunks,
                    role_sequence=role_sequence,
                    source_chunk_ids=source_chunk_ids,
                    policy_state=policy_state,
                    session_id=session_id,
                    turn_index=turn_index,
                    resource_id=resource_id,
                    action_type=action_type,
                    cold_start_triggered=cold_start_triggered,
                    confidence=confidence,
                    mastery_delta=mastery_delta,
                    action_params=final_action_params,
                    requested_override_type=requested_override_type,
                    applied_override_type=applied_override_type,
                    retrieval_query=(locals().get("query") or focus_concept or message),
                )

                if (
                    (locals().get("srl_mode") is True)
                    and (os.getenv("TUTOR_RL_EXPORT_REASONING", "false").strip().lower() == "true")
                ):
                    try:
                        srl_blob: Dict[str, Any] = {}
                        if locals().get("plan") is not None:
                            p = plan  # type: ignore[assignment]
                            srl_blob["plan"] = {
                                "thinking": p.thinking,
                                "intended_action": p.intended_action,
                                "rationale": p.action_rationale,
                                "confidence": p.confidence,
                                "assumptions": p.assumptions,
                                "risks": p.risks,
                            }
                        if locals().get("critique") is not None and critique is not None:
                            srl_blob["critique"] = {
                                "quality": getattr(critique, "overall_quality", None),
                                "issues": getattr(critique, "issues_found", None),
                                "suggestions": getattr(critique, "suggestions", None),
                                "should_revise": getattr(critique, "should_revise", None),
                            }
                        if srl_blob:
                            response_payload["observation"]["srl"] = srl_blob
                    except Exception:
                        pass

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
        if context_manager:
            try:
                context_manager.__exit__(None, None, None)
            except Exception:
                pass

    return response_payload
