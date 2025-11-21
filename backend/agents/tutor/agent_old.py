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
    if not message:
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
    """This function is not used. It's just a marker for deleted code."""
    classification = classify_message(
                message,
                target_concepts or session_state.get("target_concepts", []),
                session_state.get("last_concept"),
            )
            # Normalize classifier concept labels so downstream tools operate on
            # a single concept at a time (important when the classifier emits
            # strings like "convection, diffusion, Fourier's law").
            try:
                raw_cls_concept = classification.get("concept")
            except Exception:
                raw_cls_concept = None
            norm_cls_concept = _normalize_single_concept_label(raw_cls_concept)
            if norm_cls_concept:
                classification["concept"] = norm_cls_concept
            else:
                classification["concept"] = None
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

            primary_concept = classification.get("concept")
            if primary_concept:
                seed_concepts = [primary_concept] + list(learning_targets)
            else:
                seed_concepts = list(learning_targets)
            learning_path = fetch_prereq_chain(seed_concepts)

            enable_prereq_check = (os.getenv("TUTOR_PREREQ_CHECK_ENABLED", "true").strip().lower() == "true")
            focus_concept, prereq_check = select_focus_concept_with_prereqs(
                classification,
                learning_path,
                mastery_map,
                learning_targets,
                user_id,
                enable_prereq_check=enable_prereq_check,
            )
            if policy_state.phase == "orientation" and policy_state.pending_concept:
                focus_concept = policy_state.pending_concept
            if focus_concept:
                focus_concept = _normalize_single_concept_label(focus_concept) or focus_concept
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

            llm_policy_enabled = os.getenv("TUTOR_LLM_POLICY_ENABLED", "false").strip().lower() in ("1", "true", "yes")
            if llm_policy_enabled:
                try:
                    # Build a compact, history-aware view of recent dialogue for the policy.
                    try:
                        max_window = int(getattr(policy_state, "history_window", 3) or 3)
                    except Exception:
                        max_window = 3
                    if max_window <= 0:
                        max_window = 3
                    history_focus = (getattr(policy_state, "history_focus", None) or "mixed").lower()
                    if history_focus == "minimal":
                        max_window = 1
                    if max_window > 4:
                        max_window = 4

                    try:
                        db_turns = get_recent_turns(cur, session_id, limit=max_window)
                    except Exception:
                        db_turns = []

                    recent_turns: List[Dict[str, Any]] = []
                    # db_turns is newest-first; iterate oldest-first to preserve order.
                    for row in reversed(db_turns):
                        concept_item = row.get("concept")
                        action_type = row.get("action_type")
                        user_text = str(row.get("user_text") or "").strip()
                        tutor_text = str(row.get("response_text") or "").strip()
                        if user_text and history_focus in {"mixed", "student", "minimal"}:
                            recent_turns.append(
                                {
                                    "role": "user",
                                    "text": user_text[:280],
                                    "action_type": None,
                                    "concept": concept_item,
                                }
                            )
                        if tutor_text and history_focus in {"mixed", "tutor"}:
                            recent_turns.append(
                                {
                                    "role": "tutor",
                                    "text": tutor_text[:280],
                                    "action_type": action_type,
                                    "concept": concept_item,
                                }
                            )

                    # Always include the current user message as the final entry.
                    recent_turns.append(
                        {
                            "role": "user",
                            "text": str(message or "")[:280],
                            "action_type": None,
                            "concept": classification.get("concept"),
                        }
                    )
                    if len(recent_turns) > 6:
                        recent_turns = recent_turns[-6:]

                    policy_obs = build_policy_observation(
                        message=message,
                        classification=classification,
                        focus_concept=focus_concept,
                        concept_level=concept_level,
                        learning_targets=learning_targets,
                        learning_path=learning_path,
                        mastery_map=mastery_map,
                        policy_state=policy_state,
                        recent_turns=recent_turns,
                    )
                    policy_llm = TutorPolicyLLM()
                    decision = policy_llm.decide(policy_obs)
                    if decision is not None:
                        policy_decision = decision
                        policy_should_update = decision.should_update_mastery
                        # Allow the policy to steer how much history to surface on future turns.
                        try:
                            if decision.history_window is not None:
                                try:
                                    hw = int(decision.history_window)
                                except Exception:
                                    hw = getattr(policy_state, "history_window", 3) or 3
                                if hw <= 0:
                                    hw = 1
                                if hw > 4:
                                    hw = 4
                                policy_state.history_window = hw
                            if decision.history_focus:
                                try:
                                    hf = str(decision.history_focus).strip().lower()
                                except Exception:
                                    hf = ""
                                if hf in {"mixed", "student", "tutor", "minimal"}:
                                    policy_state.history_focus = hf
                        except Exception:
                            pass
                        try:
                            progress.append(
                                {
                                    "stage": "policy_llm",
                                    "decision": decision.to_dict(),
                                }
                            )
                        except Exception:
                            pass
                except Exception:
                    logger.exception("tutor_policy_llm_failed")

            # SRL mode flags (env + optional LLM policy control)
            srl_env_enabled = (os.getenv("TUTOR_SRL_MODE", "false").strip().lower() == "true") and (
                os.getenv("TUTOR_SRL_PLANNING_ENABLED", "true").strip().lower() == "true"
            )
            srl_mode = srl_env_enabled
            if srl_mode and policy_decision is not None and not policy_decision.use_srl_planning:
                # LLM policy can explicitly disable SRL planning for this turn.
                srl_mode = False
            if srl_mode:
                tool_calls["planner"] = {"enabled": True}

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

            # Stage 3: Retrieval (guided by policy and/or plan if available)
            policy_roles: List[str] = []
            if policy_decision is not None:
                try:
                    policy_roles = list(policy_decision.pedagogy_focus or [])
                except Exception:
                    policy_roles = []

            if srl_mode and plan:
                pedagogy_roles = policy_roles or list(plan.pedagogy_focus or []) or role_sequence
                query = (
                    (policy_decision.retrieval_query if policy_decision is not None else None)
                    or getattr(plan, "retrieval_query", None)
                    or focus_concept
                    or message
                )
                chunks = retrieve_chunks(query, resource_id, pedagogy_roles)
                if not chunks and focus_concept:
                    chunks = retrieve_chunks(focus_concept, resource_id, pedagogy_roles)
                if not chunks:
                    chunks = retrieve_chunks(message, resource_id, pedagogy_roles)
            else:
                pedagogy_roles = policy_roles or role_sequence
                query = (
                    (policy_decision.retrieval_query if policy_decision is not None else None)
                    or focus_concept
                    or message
                )
                chunks = retrieve_chunks(query, resource_id, pedagogy_roles)
                if not chunks and focus_concept and query != focus_concept:
                    chunks = retrieve_chunks(focus_concept, resource_id, pedagogy_roles)
                if not chunks and query != message:
                    chunks = retrieve_chunks(message, resource_id, pedagogy_roles)

            try:
                roles_for_trace = pedagogy_roles
                retrieval_query = (locals().get("query") or focus_concept or message)
                retrieval_chunk_ids = [c.get("id") for c in (chunks or []) if c.get("id")]
                meta = {
                    "stage": "retrieval",
                    "query": retrieval_query,
                    "roles": roles_for_trace,
                    "count": len(chunks or []),
                    "chunk_ids": retrieval_chunk_ids,
                }
                progress.append(meta)
                retrieval_metadata = {
                    "query": retrieval_query,
                    "roles": roles_for_trace,
                    "count": meta["count"],
                    "chunk_ids": retrieval_chunk_ids,
                }
            except Exception:
                pass

            logger.info(
                "tutor_retrieval_summary",
                extra={
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "query": retrieval_query,
                    "resource_id": resource_id,
                    "focus_concept": focus_concept,
                    "pedagogy_roles": roles_for_trace,
                    "chunk_ids": [c.get("id") for c in chunks],
                },
            )

            affect = classification.get("affect", "neutral")
            intent = classification.get("intent", "unknown")
            mastery_delta = payload.get("mastery_delta")

            # Default to enabling realtime mastery updates unless explicitly disabled.
            enable_mastery_update = (os.getenv("TUTOR_MASTERY_REALTIME_UPDATE", "true").strip().lower() == "true")
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
                try:
                    tool_calls["mastery_updater"] = {
                        "enabled": True,
                        "learning_rate": lr,
                        "decay_factor": df,
                        "min_update": mn,
                        "max_update": mx,
                    }
                except Exception:
                    pass

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

            wants_orientation = (
                policy_state.phase == "teaching"
                and ((intent == "greeting") or _looks_like_study_plan_request(message))
            )

            if wants_orientation and override_type is None:
                (
                    response_text,
                    confidence,
                    source_chunk_ids,
                    recommended_concept,
                ) = build_orientation_response(
                    message=message,
                    focus_concept=focus_concept,
                    learning_targets=learning_targets,
                    learning_path=learning_path,
                    mastery_map=mastery_map,
                    chunks=chunks,
                )
                inference_concept_for_orientation = recommended_concept or focus_concept
                action_params = {
                    key: value
                    for key, value in {
                        "concept": inference_concept_for_orientation,
                        "level": concept_level,
                        "mode": "orientation",
                    }.items()
                    if value
                }
                result = ActionResult(
                    action_type="explain",
                    response_text=response_text,
                    confidence=confidence,
                    source_chunk_ids=source_chunk_ids,
                    inference_concept=inference_concept_for_orientation,
                    action_params=action_params,
                    cold_start_triggered=False,
                )
                cause = "orientation_greeting"

            elif (
                locals().get("prereq_check") is not None
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
                result = handle_cold_start(
                    focus_concept,
                    concept_level,
                    chunks,
                    cur,
                    session_id,
                    policy_state,
                    dry_run=dry_run,
                )
                cause = "cold_start"
                
            elif override_type:
                # Explicit override: force specific action type
                result = handle_override(override_type, focus_concept, concept_level, chunks, override_params)
                applied_override_type = result.action_type if result.action_type == override_type else None
                cause = "override_request"
                
            else:
                # Normal flow: choose action based on student state, preferring LLM policy when available
                if policy_decision is not None:
                    # Map policy decision to concrete action handlers.
                    na = (policy_decision.next_action or "").strip().lower() or "explain"
                    mode_for_explain = (policy_decision.mode or "default").strip() or "default"

                    if na == "ask":
                        # Policy wants an assessment-style follow-up question.
                        cause = "policy_llm_ask"
                        result = handle_assessment(focus_concept, concept_level, chunks, cause)

                    elif na == "reflect" and chunks:
                        # Policy wants the student to reflect on their answer.
                        cause = "policy_llm_reflect"
                        result = handle_reflection(focus_concept, concept_level, chunks, message)

                    elif na == "hint" and chunks:
                        # Policy wants a gentle hint.
                        cause = "policy_llm_hint"
                        result = handle_override("hint", focus_concept, concept_level, chunks, {})

                    elif na == "review" and chunks:
                        # Policy wants a concise review summary.
                        cause = "policy_llm_review"
                        result = handle_override("review", focus_concept, concept_level, chunks, {})

                    else:
                        # Default to explanation, honoring policy-provided mode when possible.
                        cause = "policy_llm_explain"
                        result = handle_explain(focus_concept, concept_level, chunks, mode=mode_for_explain)

                else:
                    # Heuristic fallback when LLM policy is disabled or unavailable.
                    should_assess = (
                        intent == "reflection" and affect == "engaged" and chunks
                    )
                    
                    if should_assess:
                        # Follow-up question after claimed understanding
                        cause = "reflection_claimed_understanding"
                        result = handle_assessment(focus_concept, concept_level, chunks, cause)
                        
                    elif intent == "answer" and chunks:
                        # Student provided answer: prompt reflection
                        result = handle_reflection(focus_concept, concept_level, chunks, message)
                        cause = "student_answer"
                        
                    elif srl_mode and plan:
                        # SRL-guided execution stage
                        multi_flag = os.getenv("TUTOR_SRL_MULTI_STEP_EXECUTE", "false").strip().lower() in ("1", "true", "yes")
                        if policy_decision is not None:
                            # When LLM policy is active, let it decide whether to execute multiple
                            # planned steps this turn, subject to the global multi-step flag.
                            use_multi = multi_flag and bool(policy_decision.use_multi_step)
                        else:
                            # Heuristic fallback: decide based on intent and policy state.
                            use_multi = multi_flag and _should_use_multi_step(intent, policy_state)
                        if use_multi:
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
                                    message=message,
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
                                result = handle_explain(focus_concept, concept_level, chunks, mode="default")
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
                            # Only use the bridge example if it meets relevance/confidence thresholds.
                            try:
                                min_rel = float(getattr(gen, "min_relevance", 0.6))
                            except Exception:
                                min_rel = 0.6
                            try:
                                min_conf = float(getattr(gen, "min_confidence", 0.5))
                            except Exception:
                                min_conf = 0.5

                            if br.relevance_score >= min_rel and br.confidence >= min_conf:
                                bridge_text = (
                                    f"Example: {br.example_text}\n\n"
                                    f"Why this helps: {br.explanation}"
                                )
                                source_ids = [c.get("id") for c in (chunks or []) if c.get("id")]
                                result = ActionResult(
                                    action_type="explain",
                                    response_text=bridge_text,
                                    confidence=float(br.confidence),
                                    source_chunk_ids=source_ids,
                                    inference_concept=focus_concept,
                                    action_params={
                                        "concept": focus_concept,
                                        "level": concept_level,
                                        "mode": "bridge_example",
                                    },
                                )
                                cause = "bridge_example"
                                try:
                                    tool_calls["example_generator"] = {
                                        "used": True,
                                        "relevance": float(br.relevance_score),
                                        "confidence": float(br.confidence),
                                    }
                                except Exception:
                                    pass
                            else:
                                result = handle_explain(
                                    focus_concept,
                                    concept_level,
                                    chunks,
                                    mode="confusion_support",
                                )
                                cause = "affect_confused_explain_basics"
                        except Exception:
                            logger.exception("tutor_bridge_example_failed")
                            result = handle_explain(focus_concept, concept_level, chunks, mode="confusion_support")
                            cause = "affect_confused_explain_basics"
                    else:
                        result = handle_explain(focus_concept, concept_level, chunks, mode="confusion_support")
                        cause = "affect_confused_explain_basics"

                elif affect in {"confused", "unsure"} and chunks:
                    # Confusion detected: explain basics
                    result = handle_explain(focus_concept, concept_level, chunks, mode="confusion_support")
                    cause = "affect_confused_explain_basics"
                    
                else:
                    # Default: explain the concept
                    result = handle_explain(focus_concept, concept_level, chunks, mode="default")
                    cause = "explain_default"
            
            # Unpack result
            action_type = result.action_type
            response_text = result.response_text
            confidence = result.confidence
            source_chunk_ids = result.source_chunk_ids
            inference_concept = result.inference_concept
            final_action_params = result.action_params
            cold_start_triggered = result.cold_start_triggered

            # Normalize inference concept and action params so that mastery,
            # persistence, and UI metadata all refer to a single concept label.
            if inference_concept:
                inference_concept = _normalize_single_concept_label(inference_concept) or inference_concept
            if final_action_params and isinstance(final_action_params, dict):
                param_concept = final_action_params.get("concept")
                if param_concept:
                    norm_param = _normalize_single_concept_label(param_concept)
                    if norm_param:
                        final_action_params["concept"] = norm_param

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
                    try:
                        tool_calls["self_critic"] = {
                            "enabled": True,
                            "quality": getattr(critique, "overall_quality", None),
                            "should_revise": getattr(critique, "should_revise", None),
                        }
                    except Exception:
                        pass
            except Exception:
                pass

            policy_state.learning_path = learning_path
            policy_state.focus_concept = focus_concept
            policy_state.focus_level = concept_level
            policy_state.cold_start = cold_start_triggered

            if not final_action_params:
                base_params = {
                    "concept": inference_concept or focus_concept,
                    "level": concept_level,
                }
                final_action_params = {k: v for k, v in base_params.items() if v}

            mode_label = str(final_action_params.get("mode") or "").lower()
            if action_type == "explain" and inference_concept:
                policy_state.last_explained_concept = inference_concept
            if mode_label == "orientation":
                concept_for_pending = final_action_params.get("concept") or inference_concept or focus_concept
                policy_state.phase = "orientation"
                policy_state.pending_question_type = "orientation_followup"
                policy_state.pending_concept = concept_for_pending
            elif action_type == "ask":
                concept_for_pending = final_action_params.get("concept") or inference_concept or focus_concept
                policy_state.pending_question_type = mode_label or "question"
                policy_state.pending_concept = concept_for_pending
                policy_state.phase = "assessment"
            elif intent == "answer":
                policy_state.pending_question_type = None
                policy_state.pending_concept = None
                policy_state.phase = "teaching"

            policy_state.update_action(action_type)

            if mastery_updater and (inference_concept or focus_concept) and not dry_run:
                # When an LLM policy decision is present, allow it to explicitly disable
                # mastery updates for this turn by setting should_update_mastery = false.
                if policy_decision is not None and policy_should_update is False:
                    pass
                else:
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

            # Derive effective model name for logging
            try:
                model_name = get_effective_model_name(model_hint=model_hint)
            except Exception:
                model_name = None

            policy_trace = {"progress": progress}

            turn_id = None
            if not dry_run:
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
                    model_id=model_id,
                    model_name=model_name,
                    tool_calls=tool_calls or None,
                    retrieval_metadata=retrieval_metadata or None,
                    policy_trace=policy_trace or None,
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
