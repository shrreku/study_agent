"""Thin per-turn orchestrator that coordinates all runtime stages."""

import os
from typing import Any, Dict, List, Optional

from ..constants import logger
from ..state import TutorSessionPolicy
from ..classifier import classify_message
from ..state_machine import TutorStateManager, TutorState, StateContext
from ..policy import (
    level_for_mastery,
    needs_cold_start,
    select_focus_concept_with_prereqs,
)
from ..planning import TutorPlanner, TutorPlan
from ..self_critique import SelfCritic
from ..knowledge import fetch_mastery_map, fetch_prereq_chain
from ..persistence import (
    get_session_state,
    insert_turn,
    update_session,
    get_recent_turns,
)
from ..observation import build_observation
from ..tools.mastery_updater import MasteryUpdater
from ..tools_runtime import execute_action
from ..config import get_tutor_config
from llm.common import get_effective_model_name
from prompts import prompt_set_override

from .context import (
    TurnContext,
    ClassificationContext,
    ConceptContext,
    RetrievalContext,
)
from .utils import normalize_single_concept_label, looks_like_confirmation
from .policy_runtime import decide_with_policy
from .retrieval_runtime import run_retrieval
from .action_runtime import select_action
from .mastery_runtime import (
    setup_mastery_updater,
    apply_mastery_update,
    apply_srl_step_mastery_delta,
    get_quiz_delta_tables,
    _apply_fixed_mastery_delta,
)
from .turn_retrieval import run_retrieval_stage
from .turn_mastery import apply_mastery_and_quiz
from ..decision_engine import TutorDecisionEngine, ActionDecision
from ..context_model import TutorContext
from ..summarization import HistorySummarizer
from ..retrieval import rehydrate_chunks_from_ids
from ..step_engine import (
    ControlMode,
    StepControl,
    StepEngine,
    build_study_plan_from_policy_state,
    build_study_step_from_action_decision,
    serialize_plan,
    serialize_step,
)
from ..rl_concept_logging import (
    ConceptStepEvent,
    start_concept_episode,
    update_concept_episode_counters,
    finalize_concept_episode,
)


def run_tutor_turn(
    ctx: TurnContext,
    cur: Any,
) -> Dict[str, Any]:
    """Execute a single tutor turn orchestrating all stages.

    Stages:
    1. Configuration (Phase 4: Load unified config)
    2. Classification
    3. Concept & Mastery
    4. LLM Policy Decision
    5. SRL Planning (if enabled)
    6. Retrieval
    7. Action Selection
    8. Mastery Update
    9. Persistence & Observation

    Returns response payload for client.
    """

    # Decide prompt set override early based on requested action mode.
    try:
        raw_mode = (ctx.payload or {}).get("agent_action_mode")
    except Exception:
        raw_mode = None
    early_mode = (raw_mode or "auto").strip().lower()

    # Decide which runtime to use for step-by-step mode based on unified
    # TutorConfig. For agent_action_mode="step_by_step" we now always
    # delegate to the MDP-oriented Runtime v2 orchestrator.
    if early_mode == "step_by_step":
        use_mdp_runtime = (
            os.getenv("TUTOR_STEP_MDP_RUNTIME_ENABLED", "false").strip().lower()
            == "true"
        )
        if use_mdp_runtime:
            from ..runtime_v2.orchestrator_step_mdp import run_step_mdp_turn
        else:
            from .orchestrator_step_srl import run_step_srl_turn

        with prompt_set_override("baseline"):
            if use_mdp_runtime:
                return run_step_mdp_turn(ctx, cur)
            return run_step_srl_turn(ctx, cur)
    return _run_tutor_turn_core(ctx, cur)


def _run_tutor_turn_core(
    ctx: TurnContext,
    cur: Any,
) -> Dict[str, Any]:
    """Core implementation of a tutor turn, shared across modes."""

    # === Stage 0: Load Configuration ===
    config = get_tutor_config()
    logger.info(
        "tutor_turn_config_loaded mode=%s grounding=%s",
        config.mode,
        config.response_grounding_mode,
        extra={"session_id": ctx.session_id},
    )

    message = ctx.message
    session_id = ctx.session_id
    user_id = ctx.user_id
    turn_index = ctx.turn_index
    target_concepts = ctx.target_concepts
    resource_id = ctx.resource_id
    dry_run = ctx.dry_run
    emit_state_requested = ctx.emit_state_requested
    payload = ctx.payload
    agent_action_mode = (payload.get("agent_action_mode") or "auto").strip().lower()

    # Derive an effective per-turn mode label that can be driven by the
    # frontend via agent_action_mode while still defaulting to the
    # environment-configured TutorConfig mode.
    mode_label = (config.mode or "").strip().lower()
    if agent_action_mode == "step_by_step":
        mode_label = "step_by_step"

    control_mode: ControlMode = (
        "step" if agent_action_mode == "step_by_step" else "auto"
    )
    raw_confirmed = payload.get("confirmed_action")
    try:
        confirmed_action = str(raw_confirmed or "").strip().lower()
    except Exception:
        confirmed_action = ""

    raw_mcq_answer = payload.get("mcq_answer")
    mcq_answer = raw_mcq_answer if isinstance(raw_mcq_answer, dict) else None

    # Parse explicit action_override from client (buttons / dropdowns)
    raw_override = payload.get("action_override") or {}
    override_type: Optional[str] = None
    override_params: Dict[str, Any] = {}
    if isinstance(raw_override, dict):
        raw_type = raw_override.get("type")
        if isinstance(raw_type, str):
            override_type = raw_type.strip() or None
        params = raw_override.get("params")
        if isinstance(params, dict):
            allowed_keys = {"concept", "level", "difficulty", "question_type"}
            override_params = {
                key: value
                for key, value in params.items()
                if key in allowed_keys and value not in (None, "")
            }

    # Parse optional step_control for step-by-step SRL mode. This control
    # surface is only honored when agent_action_mode is "step_by_step"; in
    # other modes it is ignored.
    raw_step = payload.get("step_control") or {}
    step_control_type: Optional[str] = None
    step_control_params: Dict[str, Any] = {}
    if isinstance(raw_step, dict):
        raw_step_type = raw_step.get("type")
        if isinstance(raw_step_type, str):
            step_control_type = raw_step_type.strip().lower() or None
        step_params = raw_step.get("params")
        if isinstance(step_params, dict):
            step_control_params = step_params

    if agent_action_mode == "step_by_step" and step_control_type:
        # continue: treat as explicit continue confirmation for retrieval reuse
        if step_control_type == "continue":
            if not confirmed_action:
                confirmed_action = "continue"
        # skip_to_quiz: exhaust current plan so downstream logic enters quiz
        elif step_control_type == "skip_to_quiz":
            override_type = "step_skip_to_assessment"
        # skip_to_next_concept: mark concept as mastered and advance loop
        elif step_control_type in {"skip_to_next_concept", "skip_to_next"}:
            override_type = "step_next_concept"
        # next_step_override: override the next plan step's action
        elif step_control_type == "next_step_override":
            try:
                raw_action = str(step_control_params.get("action") or "").strip().lower()
            except Exception:
                raw_action = ""
            if raw_action in {"ask", "explain", "hint", "review", "reflect"}:
                override_type = raw_action
                # Thread through optional difficulty hint
                diff = step_control_params.get("difficulty")
                if isinstance(diff, str) and diff:
                    if not override_params:
                        override_params = {}
                    override_params.setdefault("difficulty", diff)
        # text_input is intentionally ignored by backend logic; the message
        # content is already taken into account by classification and policy.

    # Derive normalized StepControl for downstream tools/debugging.
    step_control_obj: Optional[StepControl] = None
    try:
        if agent_action_mode == "step_by_step":
            if step_control_type == "continue" or confirmed_action in {"continue", "next", "yes"}:
                step_control_obj = StepControl(type="continue", params=dict(step_control_params or {}))
            elif step_control_type == "skip_to_quiz" or override_type == "step_skip_to_assessment":
                step_control_obj = StepControl(type="skip_to_quiz", params=dict(step_control_params or {}))
            elif step_control_type in {"skip_to_next_concept", "skip_to_next"} or override_type == "step_next_concept":
                step_control_obj = StepControl(type="skip_to_next_concept", params=dict(step_control_params or {}))
            elif override_type == "session_end":
                step_control_obj = StepControl(type="end_session", params=dict(step_control_params or {}))
    except Exception:
        step_control_obj = None

    progress: List[Dict[str, Any]] = []
    tool_calls: Dict[str, Any] = {}
    retrieval_metadata: Dict[str, Any] = {}

    # Log turn start
    logger.info(
        "tutor_turn_start turn=%s msg_preview=%r targets=%s",
        turn_index,
        (message or "")[:100],
        target_concepts or [],
    extra={
        "session_id": session_id,
        "user_id": user_id,
    }
    )

    # === Stage 1: Classification ===
    session_state = get_session_state(cur, session_id)
    policy_state = TutorSessionPolicy.from_dict(session_state.get("policy"))
    
    # === Initialize State Machine ===
    state_manager = TutorStateManager.from_dict(policy_state.state_machine)
    logger.info(
        "tutor_state_manager_init state=%s",
        state_manager.current_state.value,
        extra={"session_id": session_id, "user_id": user_id},
    )

    message_for_classification = message
    if agent_action_mode == "step_by_step" and step_control_type:
        message_for_classification = ""
    is_control_turn = bool(
        agent_action_mode == "step_by_step"
        and (step_control_type or mcq_answer is not None)
        and not (isinstance(message_for_classification, str) and message_for_classification.strip())
    )
    if is_control_turn:
        last_concept = session_state.get("last_concept")
        classification_raw = {
            "intent": "control",
            "affect": "neutral",
            "concept": last_concept,
            "confidence": None,
        }
    else:
        classification_raw = classify_message(
            message_for_classification,
            target_concepts or session_state.get("target_concepts", []),
            session_state.get("last_concept"),
        )
    # Normalize classifier concept labels
    try:
        raw_cls_concept = classification_raw.get("concept")
    except Exception:
        raw_cls_concept = None
    norm_cls_concept = normalize_single_concept_label(raw_cls_concept)
    if norm_cls_concept:
        classification_raw["concept"] = norm_cls_concept
    
    classification = ClassificationContext(
        intent=classification_raw.get("intent", "unknown"),
        affect=classification_raw.get("affect", "neutral"),
        concept=classification_raw.get("concept"),
        confidence=classification_raw.get("confidence"),
    )

    # If the client provided a structured MCQ answer, treat this turn as an answer
    # for downstream mastery/state logic regardless of free-text content.
    if mcq_answer is not None:
        classification.intent = "answer"
    
    try:
        progress.append({
            "stage": "classification",
            "intent": classification.intent,
            "affect": classification.affect,
            "concept": classification.concept,
            "confidence": classification.confidence,
        })
    except Exception:
        pass

    logger.info(
        "tutor_stage_classification turn=%s intent=%s affect=%s concept=%s confidence=%s",
        turn_index,
        classification.intent,
        classification.affect,
        classification.concept,
        classification.confidence,
    extra={
        "session_id": session_id,
        "user_id": user_id,
    }
    )

    # === Stage 2: Concept & Mastery ===
    learning_targets = target_concepts or session_state.get("target_concepts", [])
    mastery_map = fetch_mastery_map(cur, user_id)

    primary_concept = classification.concept
    if primary_concept:
        seed_concepts = [primary_concept] + list(learning_targets)
    else:
        seed_concepts = list(learning_targets)
    learning_path = fetch_prereq_chain(seed_concepts)

    focus_concept, prereq_check = select_focus_concept_with_prereqs(
        classification_raw,
        learning_path,
        mastery_map,
        learning_targets,
        user_id,
        enable_prereq_check=True,
    )
    if policy_state.phase == "orientation" and policy_state.pending_concept:
        focus_concept = policy_state.pending_concept
    if focus_concept:
        focus_concept = normalize_single_concept_label(focus_concept) or focus_concept
    concept_level = level_for_mastery((mastery_map.get(focus_concept) or {}).get("mastery"))

    concepts = ConceptContext(
        focus_concept=focus_concept,
        concept_level=concept_level,
        learning_path=learning_path,
        learning_targets=learning_targets,
        mastery_map=mastery_map,
        prereq_check=prereq_check,
    )

    # Initialize concept-level episode tracking when entering step-by-step mode
    # for a focus concept. This is idempotent for the same concept.
    if agent_action_mode == "step_by_step" and focus_concept:
        try:
            raw_target = os.getenv("TUTOR_STEP_SRL_TARGET_MASTERY", "0.8") or "0.8"
            try:
                target_mastery_for_episode = float(raw_target)
            except Exception:
                target_mastery_for_episode = 0.8
            start_concept_episode(
                policy_state=policy_state,
                session_id=session_id,
                user_id=user_id,
                concept_id=focus_concept,
                turn_index=turn_index,
                target_mastery=target_mastery_for_episode,
                mastery_map=mastery_map,
            )
        except Exception:
            pass

    # Derive MCQ outcome (if any) using the last stored MCQ in policy_state.
    mcq_outcome: Optional[Dict[str, Any]] = None
    if mcq_answer is not None and isinstance(policy_state.last_mcq, dict):
        last_mcq = policy_state.last_mcq or {}
        try:
            last_qid = str(last_mcq.get("question_id") or "").strip()
            ans_qid = str(mcq_answer.get("question_id") or "").strip()
        except Exception:
            last_qid = ""
            ans_qid = ""
        if (not ans_qid) or (last_qid and ans_qid and last_qid == ans_qid):
            try:
                opt_id = str(mcq_answer.get("option_id") or "").strip()
            except Exception:
                opt_id = ""
            options = last_mcq.get("options") or []
            difficulty = None
            if isinstance(options, list) and opt_id:
                for opt in options:
                    if not isinstance(opt, dict):
                        continue
                    if str(opt.get("id") or "").strip() == opt_id:
                        try:
                            difficulty = str(opt.get("difficulty") or "").strip().lower() or None
                        except Exception:
                            difficulty = None
                        break
            if not difficulty:
                try:
                    difficulty = str(last_mcq.get("difficulty") or "").strip().lower() or None
                except Exception:
                    difficulty = None
            try:
                correct_id = str(last_mcq.get("correct_option_id") or "").strip()
            except Exception:
                correct_id = ""
            try:
                explain_id = str(last_mcq.get("explain_option_id") or "").strip()
            except Exception:
                explain_id = ""
            chose_explain = bool(opt_id and explain_id and opt_id == explain_id)
            answer_correct: Optional[bool] = None
            if opt_id:
                if opt_id == correct_id:
                    answer_correct = True
                elif not chose_explain:
                    answer_correct = False
            concept_for_mcq = last_mcq.get("concept") or focus_concept
            mcq_outcome = {
                "question_id": last_mcq.get("question_id"),
                "concept": concept_for_mcq,
                "difficulty": difficulty,
                "answer_correct": answer_correct,
                "chose_explain": chose_explain,
                "option_id": opt_id or None,
            }
            try:
                policy_state.recent_mcq_outcomes.append(mcq_outcome)
                if len(policy_state.recent_mcq_outcomes) > 20:
                    policy_state.recent_mcq_outcomes = policy_state.recent_mcq_outcomes[-20:]
            except Exception:
                pass

    logger.info(
        "tutor_prereq_check focus=%s ready=%s missing=%s recommend=%s",
        focus_concept,
        getattr(prereq_check, "ready", True),
        getattr(prereq_check, "missing_prereqs", []) if prereq_check else [],
        getattr(prereq_check, "recommendation", ""),
    extra={
        "session_id": session_id,
        "user_id": user_id,
    }
    )

    logger.info(
        "tutor_stage_concept_mastery turn=%s classified=%s focus=%s level=%s targets=%s mastery_count=%s prereq=%s",
        turn_index,
        classification.concept,
        focus_concept,
        concept_level,
        learning_targets,
        len(mastery_map),
        prereq_check is not None,
    extra={
        "session_id": session_id,
        "user_id": user_id,
    }
    )

    # In step-by-step SRL mode, if the student sends a new free-text message
    # (i.e., not a pure step_control or MCQ answer turn), treat this as a
    # signal to regenerate the SRL plan instead of continuing the previous
    # one. Clear any existing plan so the decision engine will request a
    # fresh plan aligned with the updated query and state.
    if (
        mode_label == "step_by_step"
        and agent_action_mode == "step_by_step"
        and isinstance(message, str)
        and message.strip()
        and not step_control_type
        and confirmed_action != "continue"
    ):
        try:
            if isinstance(policy_state.srl_plan, dict) and policy_state.srl_plan:
                policy_state.srl_plan = None
                policy_state.srl_plan_step_index = 0
        except Exception:
            pass

    try:
        last_turns = get_recent_turns(cur, session_id, limit=10)
    except Exception:
        last_turns = []
    try:
        session_summary = getattr(policy_state, "session_summary", "")
    except Exception:
        session_summary = ""
    cold_start_check = False
    if os.getenv("TUTOR_COLD_START_ENABLED", "true").strip().lower() == "true":
        cold_start_check = needs_cold_start(focus_concept, mastery_map, policy_state)

    tutor_context = TutorContext(
        session_id=session_id,
        user_id=user_id,
        turn_index=turn_index,
        message=message,
        intent=classification.intent,
        affect=classification.affect,
        inferred_concept=classification.concept,
        current_state=state_manager.current_state,
        focus_concept=focus_concept,
        concept_level=concept_level,
        mastery_map=mastery_map,
        prerequisites=getattr(prereq_check, "missing_prereqs", []) if prereq_check else [],
        learning_path=learning_path,
        retrieval_chunks=[],
        recent_turns=last_turns,
        session_summary=session_summary,
        current_plan=getattr(policy_state, "srl_plan", None),
        plan_step_index=getattr(policy_state, "srl_plan_step_index", 0),
        cold_start_eligible=cold_start_check,
        recent_mcq_outcomes=list(getattr(policy_state, "recent_mcq_outcomes", []) or []),
    )

    # === Stage 3: Unified Decision Making via StepEngine ===

    step_engine = StepEngine()
    engine_result = step_engine.decide_step(
        tutor_context=tutor_context,
        state_manager=state_manager,
        control_mode=control_mode,
        control=step_control_obj,
        policy_state=policy_state,
        override_type=override_type,
        override_params=override_params,
        mode_label=mode_label,
    )
    action_decision = engine_result.action_decision

    # Legacy compatibility variables
    policy_decision = None
    plan = None
    srl_mode = (mode_label != "simple")
    
    # Handle Generated Plan Persistence
    if action_decision.generated_plan:
        # Persist the plan in policy state
        try:
            p = action_decision.generated_plan
            policy_state.srl_plan = {
                "thinking": p.get("thinking", ""),
                "intended_action": p.get("intended_action", "explain"),
                "action_rationale": p.get("action_rationale", ""),
                "retrieval_query": p.get("retrieval_query", ""),
                "pedagogy_focus": list(p.get("pedagogy_focus", []) or []),
                "difficulty_cap": p.get("difficulty_cap", "intermediate"),
                "confidence": float(p.get("confidence", 0.6)),
                "assumptions": list(p.get("assumptions", []) or []),
                "risks": list(p.get("risks", []) or []),
                "steps": list(p.get("steps", []) or []),
                "target_sequence": list(p.get("target_sequence", []) or []),
            }
            policy_state.srl_plan_step_index = 0
            
            # Create TutorPlan object for downstream compatibility
            plan = TutorPlan(**policy_state.srl_plan)
            
            tool_calls["planner"] = {"enabled": True}
            progress.append({
                "stage": "planning",
                "intended_action": plan.intended_action,
                "confidence": plan.confidence,
            })
        except Exception as e:
            logger.warning(f"Failed to persist generated plan: {e}")

    # If no new plan was generated this turn but a plan exists in policy
    # state, rehydrate it so we can continue executing it and surface the
    # next recommended step on subsequent turns.
    if plan is None and isinstance(policy_state.srl_plan, dict):
        try:
            plan = TutorPlan(**policy_state.srl_plan)
        except Exception:
            plan = None

    # Build a StudyPlan projection from policy_state for debugging/tools.
    study_plan = None
    try:
        study_plan = build_study_plan_from_policy_state(
            policy_state=policy_state,
            focus_concept=focus_concept,
            session_id=session_id,
        )
    except Exception:
        study_plan = None

    # === Stage 4: Retrieval (Guided by Decision, with Reuse) ===
    retrieval, retrieval_metadata, retrieval_progress = run_retrieval_stage(
        cur=cur,
        session_id=session_id,
        focus_concept=focus_concept,
        concept_level=concept_level,
        message=message,
        resource_id=resource_id,
        config=config,
        agent_action_mode=agent_action_mode,
        confirmed_action=confirmed_action,
        step_control_type=step_control_type,
        tutor_context=tutor_context,
        action_decision=action_decision,
        srl_mode=srl_mode,
        plan=plan,
    )

    # Update context with retrieval results
    tutor_context.retrieval_chunks = retrieval.chunks

    if retrieval_progress:
        try:
            progress.append(retrieval_progress)
        except Exception:
            pass

    # === Stage 5: Action Selection (Execution) ===
    # For execution, only pass through override types that are actual
    # action overrides (ask/explain/hint/etc.). Step-by-step control
    # overrides (skip_to_assessment, next_concept, session_end) are
    # handled by the step-by-step runtime and mastery logic instead.
    override_type_for_execution: Optional[str] = override_type
    if mode_label == "step_by_step" and override_type in {
        "step_skip_to_assessment",
        "step_next_concept",
        "session_end",
    }:
        override_type_for_execution = None

    requested_override_type = override_type

    # Pass action_decision to select_action to handle execution
    # Note: override_type (from payload) still takes precedence if set
    
    result, cause, applied_override_type = select_action(
        cur=cur,
        session_id=session_id,
        classification=classification,
        concepts=concepts,
        retrieval=retrieval,
        policy_state=policy_state,
        policy_decision=None, 
        srl_mode=srl_mode,
        plan=plan,
        override_type=override_type_for_execution,
        override_params=override_params,
        payload=payload,
        dry_run=dry_run,
        agent_action_mode=agent_action_mode,
        force_cold_start=action_decision.cold_start,
        action_decision=action_decision,
        recent_turns=last_turns,
    )

    # Unpack result
    action_type = result.action_type
    response_text = result.response_text
    confidence = result.confidence
    source_chunk_ids = result.source_chunk_ids
    inference_concept = result.inference_concept
    final_action_params = result.action_params
    cold_start_triggered = result.cold_start_triggered
    mcq = getattr(result, "mcq", None)

    # Persist the last MCQ so that a subsequent mcq_answer can be resolved.
    if action_type == "ask":
        try:
            policy_state.last_mcq = mcq if isinstance(mcq, dict) else None
        except Exception:
            pass

    # === State Machine: Update Action & Check Transitions ===
    try:
        # Track the action taken
        state_manager.update_action(action_type)
        
        # Build context for transition checking
        state_context = StateContext(
            focus_concept=focus_concept,
            student_message=message,
            last_action=action_type,
            answer_correct=None,  # Will be updated if student answered
            answer_quality=None,
            mastery_map=mastery_map,
            session_turn_count=turn_index,
            turn_signals=getattr(tutor_context, "turn_signals", None),
            phase_suggestion=getattr(tutor_context, "phase_suggestion", None),
        )
        
        # Check if state transition needed
        next_state = state_manager.check_transitions(state_context)
        if next_state and next_state != state_manager.current_state:
            old_state = state_manager.current_state
            state_manager.transition_to(
                next_state,
                reason=f"auto_transition_action_{action_type}",
            )
            logger.info(
                "tutor_state_changed turn=%s old_state=%s new_state=%s action=%s",
                turn_index,
                old_state.value,
                next_state.value,
                action_type,
                extra={"session_id": session_id, "user_id": user_id},
            )
    except Exception as e:
        logger.warning(
            "tutor_state_machine_error error=%s",
            str(e),
            extra={"session_id": session_id, "user_id": user_id},
        )

    # === Response Deduplication Check ===
    # Prevent identical responses from being sent in consecutive turns (loop prevention)
    if srl_mode and agent_action_mode == "step_by_step":
        try:
            if last_turns:
                prev_response = last_turns[0].get("response_text", "").strip()
                if prev_response and response_text.strip() == prev_response:
                    logger.warning(
                        "tutor_duplicate_response_detected turn=%s action=%s switching_to=ask",
                        turn_index,
                        action_type,
                    )
                    # Switch to a different action to break the loop
                    from ..responses import build_followup_question
                    response_text, confidence, source_chunk_ids = build_followup_question(
                        inference_concept or focus_concept,
                        concept_level,
                        retrieval.chunks,
                    )
                    action_type = "ask"
                    cause = "duplicate_prevention"
        except Exception as e:
            logger.exception("tutor_duplicate_check_failed error=%s", str(e))

    # Normalize inference concept and action params
    if inference_concept:
        inference_concept = normalize_single_concept_label(inference_concept) or inference_concept
    if final_action_params and isinstance(final_action_params, dict):
        param_concept = final_action_params.get("concept")
        if param_concept:
            norm_param = normalize_single_concept_label(param_concept)
            if norm_param:
                final_action_params["concept"] = norm_param

    logger.info(
        "tutor_action_decision turn=%s action=%s concept=%s cause=%s conf=%.2f chunks=%s policy=%s srl=%s cold_start=%s override=%s resp=%r",
        turn_index,
        action_type,
        inference_concept or focus_concept,
        cause,
        confidence or 0.0,
        len(source_chunk_ids or []),
        bool(policy_decision),
        srl_mode,
        cold_start_triggered,
        applied_override_type,
        (response_text or "")[:80],
    extra={
        "session_id": session_id,
        "user_id": user_id,
    }
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

    # === Stage 7: SRL Self-Critique (optional) ===
    critique = None
    if (
        (srl_mode is True)
        and (os.getenv("TUTOR_SRL_SELF_CRITIQUE", "false").strip().lower() == "true")
        and (plan is not None)
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

    # Construct StudyStep projection from the ActionDecision and current state
    study_step = None
    try:
        is_mcq_flag = bool(action_type == "ask" and isinstance(mcq, dict))
        expects_answer = bool(
            action_type in {"ask"}
            or classification.intent in {"answer", "reflection", "question"}
        )
        study_step = build_study_step_from_action_decision(
            decision=action_decision,
            state=state_manager.current_state,
            focus_concept=inference_concept or focus_concept,
            session_id=session_id,
            turn_index=turn_index,
            plan_index=getattr(policy_state, "srl_plan_step_index", None),
            is_mcq=is_mcq_flag,
            expects_answer=expects_answer,
        )
    except Exception:
        study_step = None

    # === Stage 8: Update Policy State ===
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
        if mode_label == "step_by_step" and policy_state.srl_plan:
            try:
                max_q = int(os.getenv("TUTOR_STEP_SRL_QUIZ_MAX_QUESTIONS", "3") or 3)
            except Exception:
                max_q = 3
            if max_q < 1:
                max_q = 1
            policy_state.quiz_phase = "quiz"
            policy_state.quiz_question_index = 0
            policy_state.quiz_max_questions = max_q
    elif classification.intent == "answer":
        policy_state.pending_question_type = None
        policy_state.pending_concept = None
        policy_state.phase = "teaching"

    policy_state.update_action(action_type)

    pre_mastery_for_logging = None
    if focus_concept:
        try:
            pre_mastery_for_logging = float(
                (mastery_map.get(focus_concept) or {}).get("mastery", 0.0) or 0.0
            )
        except Exception:
            pre_mastery_for_logging = None

    # === Stage 9: Mastery Update ===
    mastery_delta, srl_step_delta, quiz_delta = apply_mastery_and_quiz(
        cur=cur,
        user_id=user_id,
        session_id=session_id,
        turn_index=turn_index,
        message=message,
        classification=classification,
        focus_concept=focus_concept,
        inference_concept=inference_concept,
        concept_level=concept_level,
        mastery_map=mastery_map,
        retrieval=retrieval,
        srl_mode=srl_mode,
        plan=plan,
        final_action_params=final_action_params,
        action_type=action_type,
        policy_state=policy_state,
        last_turns=last_turns,
        config=config,
        mcq_outcome=mcq_outcome,
        dry_run=dry_run,
        requested_override_type=requested_override_type,
        tool_calls=tool_calls,
    )

    post_mastery_for_logging = None
    if focus_concept:
        try:
            post_mastery_for_logging = float(
                (mastery_map.get(focus_concept) or {}).get("mastery", 0.0) or 0.0
            )
        except Exception:
            post_mastery_for_logging = None

    # === Concept-level Step Logging (Step-by-Step Mode) ===
    try:
        if agent_action_mode == "step_by_step" and (focus_concept or inference_concept):
            concept_id_for_logging = inference_concept or focus_concept
            control_type_for_logging: Optional[str] = None
            if mcq_answer is not None:
                control_type_for_logging = "mcq_answer"
            elif step_control_type:
                control_type_for_logging = step_control_type
            elif confirmed_action:
                control_type_for_logging = confirmed_action

            # Update per-episode counters now that we know the control type and MCQ outcome.
            try:
                update_concept_episode_counters(
                    policy_state=policy_state,
                    control_type=control_type_for_logging,
                    mcq_outcome=mcq_outcome,
                )
            except Exception:
                pass

            # Best-effort current SRL plan step snapshot (if available)
            plan_step_for_logging: Optional[Dict[str, Any]] = None
            if plan is not None:
                try:
                    steps_for_logging = list(getattr(plan, "steps", []) or [])
                    raw_cursor = getattr(policy_state, "srl_plan_step_index", 0)
                    try:
                        cursor = int(raw_cursor or 0)
                    except Exception:
                        cursor = 0
                    idx = cursor - 1 if cursor > 0 else 0
                    if 0 <= idx < len(steps_for_logging):
                        raw_step = steps_for_logging[idx] or {}
                        if isinstance(raw_step, dict):
                            plan_step_for_logging = dict(raw_step)
                except Exception:
                    plan_step_for_logging = None

            reward_for_logging: Dict[str, Any] = {}
            if mastery_delta is not None:
                reward_for_logging["mastery_delta"] = mastery_delta
            if srl_step_delta is not None:
                reward_for_logging["srl_step_delta"] = srl_step_delta
            if quiz_delta is not None:
                reward_for_logging["quiz_delta"] = quiz_delta

            # Map control surface + overrides into a concept-level MDP action label.
            mdp_action: Optional[str] = None
            if step_control_obj is not None:
                if step_control_obj.type == "continue":
                    mdp_action = "FOLLOW_PLAN_STEP"
                elif step_control_obj.type == "skip_to_quiz":
                    mdp_action = "JUMP_TO_ASSESSMENT"
                elif step_control_obj.type == "skip_to_next_concept":
                    mdp_action = "ADVANCE_CONCEPT"
                elif step_control_obj.type == "end_session":
                    mdp_action = "TERMINATE_CONCEPT"
            elif requested_override_type:
                if requested_override_type == "step_skip_to_assessment":
                    mdp_action = "JUMP_TO_ASSESSMENT"
                elif requested_override_type == "step_next_concept":
                    mdp_action = "ADVANCE_CONCEPT"
                elif requested_override_type == "session_end":
                    mdp_action = "TERMINATE_CONCEPT"
            elif control_type_for_logging in {"continue", "next", "yes"}:
                mdp_action = "FOLLOW_PLAN_STEP"
            elif control_type_for_logging == "skip_to_quiz":
                mdp_action = "JUMP_TO_ASSESSMENT"
            elif control_type_for_logging in {"skip_to_next_concept", "skip_to_next"}:
                mdp_action = "ADVANCE_CONCEPT"

            concept_for_episode = (
                getattr(policy_state, "concept_episode_concept", None)
                or concept_id_for_logging
            )

            episode_id_for_logging = (
                getattr(policy_state, "concept_episode_id", None)
                or f"ce-{session_id}-{concept_for_episode}"
            )

            mdp_state_payload: Dict[str, Any] = {
                "episode_id": episode_id_for_logging,
                "episode_step_count": getattr(
                    policy_state, "concept_episode_step_count", 0
                ),
                "plan_phase": getattr(policy_state, "phase", None),
                "quiz_phase": getattr(policy_state, "quiz_phase", None),
                "quiz_question_index": getattr(
                    policy_state, "quiz_question_index", 0
                ),
                "quiz_max_questions": getattr(
                    policy_state, "quiz_max_questions", 0
                ),
            }
            if mdp_action:
                mdp_state_payload["action"] = mdp_action

            control_payload: Dict[str, Any] = {
                "step_control": step_control_params or {},
                "mcq_answer": mcq_answer or {},
                "override_type": requested_override_type,
                "agent_action_mode": agent_action_mode,
                "mdp": mdp_state_payload,
            }

            event = ConceptStepEvent(
                episode_id=episode_id_for_logging,
                session_id=session_id,
                user_id=user_id,
                concept_id=concept_id_for_logging,
                step_index=turn_index,
                turn_start_index=turn_index,
                turn_end_index=turn_index,
                control_type=control_type_for_logging,
                control_payload=control_payload,
                source="plan" if not requested_override_type else "override",
                plan_step=plan_step_for_logging,
                pre_mastery=pre_mastery_for_logging,
                post_mastery=post_mastery_for_logging,
                quiz_result=mcq_outcome or {},
                reward=reward_for_logging,
                outcome=None,
            )

            try:
                logger.info(
                    "tutor_concept_step_event",
                    extra={
                        "session_id": session_id,
                        "user_id": user_id,
                        "concept_id": concept_id_for_logging,
                        "concept_step_event": event.__dict__,
                    },
                )
            except Exception:
                pass

            # If this turn satisfies a concept-episode termination condition,
            # build and emit a ConceptEpisode summary for downstream analysis.
            try:
                raw_target = os.getenv("TUTOR_STEP_SRL_TARGET_MASTERY", "0.8") or "0.8"
                try:
                    target_mastery_for_episode = float(raw_target)
                except Exception:
                    target_mastery_for_episode = 0.8

                termination_reason: Optional[str] = None

                # Mastery-based termination
                if (
                    concept_for_episode
                    and post_mastery_for_logging is not None
                    and target_mastery_for_episode is not None
                    and post_mastery_for_logging >= target_mastery_for_episode
                ):
                    termination_reason = "mastery_reached"

                # Max-steps guardrail (optional)
                if termination_reason is None:
                    max_steps_raw = os.getenv("TUTOR_STEP_SRL_MAX_STEPS", "0") or "0"
                    try:
                        max_steps = int(max_steps_raw)
                    except Exception:
                        max_steps = 0
                    if max_steps > 0 and getattr(
                        policy_state, "concept_episode_step_count", 0
                    ) >= max_steps:
                        termination_reason = "max_steps_reached"

                # Explicit skip to next concept
                if termination_reason is None and (
                    requested_override_type == "step_next_concept"
                    or control_type_for_logging in {"skip_to_next_concept", "skip_to_next"}
                ):
                    termination_reason = "skip_next_concept"

                # Explicit session end
                if termination_reason is None and (
                    requested_override_type == "session_end"
                    or (
                        step_control_obj is not None
                        and step_control_obj.type == "end_session"
                    )
                ):
                    termination_reason = "session_end"

                if termination_reason and concept_for_episode:
                    episode = finalize_concept_episode(
                        policy_state=policy_state,
                        session_id=session_id,
                        user_id=user_id,
                        concept_id=concept_for_episode,
                        turn_index=turn_index,
                        mastery_map=mastery_map,
                        target_mastery=target_mastery_for_episode,
                        termination_reason=termination_reason,
                        pre_mastery=pre_mastery_for_logging,
                        post_mastery=post_mastery_for_logging,
                    )
                    if episode is not None:
                        try:
                            logger.info(
                                "tutor_concept_episode",
                                extra={
                                    "session_id": session_id,
                                    "user_id": user_id,
                                    "concept_id": concept_for_episode,
                                    "concept_episode": episode.__dict__,
                                },
                            )
                        except Exception:
                            pass
            except Exception:
                logger.exception("tutor_concept_episode_logging_failed")
    except Exception:
        logger.exception("tutor_concept_step_logging_failed")

    # === Stage 10: Persistence ===
    logger.info(
        "tutor_stage_persistence_start turn=%s dry_run=%s action=%s tool_calls=%s",
        turn_index,
        dry_run,
        action_type,
        len(tool_calls),
    extra={
        "session_id": session_id,
        "user_id": user_id,
    }
    )
    
    policy_trace = {"progress": progress}

    # === Persist State Machine ===
    try:
        policy_state.state_machine = state_manager.get_state_for_persistence()
    except Exception as e:
        logger.warning(
            "tutor_state_machine_persist_error error=%s",
            str(e),
            extra={"session_id": session_id, "user_id": user_id},
        )

    turn_id = None
    if not dry_run:
        turn_id = insert_turn(
            cur,
            session_id,
            turn_index,
            message,
            classification.intent,
            classification.affect,
            inference_concept,
            action_type,
            response_text,
            source_chunk_ids,
            confidence,
            mastery_delta,
            model_id=payload.get("model_id"),
            model_name=get_effective_model_name(model_hint=payload.get("model_hint")),
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
        
        logger.info(
            "tutor_stage_persistence_complete turn=%s turn_id=%s tools=%s",
            turn_index,
            turn_id,
            list(tool_calls.keys()),
        extra={
            "session_id": session_id,
            "user_id": user_id,
        }
        )

    # === Stage 11: Build Response ===
    response_payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "turn_index": turn_index,
        "response": response_text,
        "action_type": action_type,
        "source_chunk_ids": source_chunk_ids,
        "confidence": confidence,
        "intent": classification.intent,
        "affect": classification.affect,
        "concept": inference_concept,
        "level": concept_level,
        "learning_path": learning_path,
        "cold_start": cold_start_triggered,
        "classification_confidence": classification.confidence,
        "progress": progress,
        "mcq": mcq,
        # V2 Architecture: Include tutor state
        "current_state": state_manager.current_state.value,
        "state_history": [s.value for s in state_manager.state_history],
        # Phase 5: Include mastery tracking
        "mastery_delta": mastery_delta,
        "mastery_updated": mastery_delta is not None and mastery_delta != 0.0,
        "quiz_phase": getattr(policy_state, "quiz_phase", ""),
        "quiz_question_index": getattr(policy_state, "quiz_question_index", 0),
        "quiz_max_questions": getattr(policy_state, "quiz_max_questions", 0),
    }

    # Attach canonical step/plan projections for step-by-step mode, along with
    # internal-only debug variants for tooling and inspection.
    try:
        if study_step is not None:
            step_dict = serialize_step(study_step)
            # For now, expose a simple fixed control surface that matches the
            # existing step-by-step buttons. Later tickets may make this
            # dynamic based on StepEngine policy.
            step_controls: List[Dict[str, Any]] = [
                {"type": "continue", "label": "Continue"},
                {"type": "skip_to_quiz", "label": "Skip to quiz"},
                {"type": "skip_to_next_concept", "label": "Next concept"},
                {"type": "end_session", "label": "End session"},
            ]
            step_dict["controls"] = step_controls

            # Public field for clients
            if agent_action_mode == "step_by_step":
                response_payload["step"] = step_dict

            # Internal debug projection mirrors the raw StudyStep.
            response_payload["debug_step"] = step_dict

        if study_plan is not None:
            plan_dict = serialize_plan(study_plan)
            if agent_action_mode == "step_by_step":
                response_payload["plan"] = plan_dict
            response_payload["debug_plan"] = plan_dict

        if step_control_obj is not None:
            response_payload["debug_step_control"] = {
                "type": step_control_obj.type,
                "params": dict(step_control_obj.params or {}),
                "mode": control_mode,
            }
    except Exception:
        pass

    # Include SRL artifacts on the response payload
    if srl_mode is True:
        try:
            if plan is not None:
                response_payload["srl_plan"] = {
                    "thinking": plan.thinking,
                    "rationale": plan.action_rationale,
                    "confidence": plan.confidence,
                    "assumptions": plan.assumptions,
                    "risks": plan.risks,
                    "steps": getattr(plan, "steps", None),
                    "target_sequence": getattr(plan, "target_sequence", None),
                }
                # Surface the next recommended step from the plan, based on
                # the per-session cursor. This enables the frontend to show a
                # "continue" button for step-by-step mode.
                try:
                    next_idx = int(getattr(policy_state, "srl_plan_step_index", 0) or 0)
                except Exception:
                    next_idx = 0
                steps = list(getattr(plan, "steps", []) or [])
                next_step = None
                if 0 <= next_idx < len(steps):
                    s = steps[next_idx] or {}
                    next_step = {
                        "index": next_idx,
                        "action": s.get("action"),
                        "pedagogy_focus": s.get("pedagogy_focus"),
                        "target_concept": s.get("target_concept"),
                    }
                response_payload["srl_next_step"] = next_step
            if critique is not None:
                response_payload["srl_critique"] = {
                    "quality": getattr(critique, "overall_quality", None),
                    "issues": getattr(critique, "issues_found", None),
                    "suggestions": getattr(critique, "suggestions", None),
                    "should_revise": getattr(critique, "should_revise", None),
                }
        except Exception:
            pass

    # === Stage 12: Observation (optional) ===
    if emit_state_requested:
        response_payload["observation"] = build_observation(
            message=message,
            user_id=user_id,
            learning_targets=learning_targets,
            classification=classification_raw,
            focus_concept=focus_concept,
            concept_level=concept_level,
            inference_concept=inference_concept,
            learning_path=learning_path,
            mastery_map=mastery_map,
            chunks=retrieval.chunks,
            role_sequence=[],  # TODO: pass from retrieval context
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
            retrieval_query=retrieval.query,
        )

        if (srl_mode is True) and (os.getenv("TUTOR_RL_EXPORT_REASONING", "false").strip().lower() == "true"):
            try:
                srl_blob: Dict[str, Any] = {}
                if plan is not None:
                    srl_blob["plan"] = {
                        "thinking": plan.thinking,
                        "intended_action": plan.intended_action,
                        "rationale": plan.action_rationale,
                        "confidence": plan.confidence,
                        "assumptions": plan.assumptions,
                        "risks": plan.risks,
                    }
                if critique is not None:
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
        "tutor_turn_committed session=%s turn=%s turn_id=%s user=%s action=%s intent=%s affect=%s concept=%s conf=%s chunks=%s mastery_delta=%s policy=%s srl=%s cold_start=%s",
        session_id,
        turn_index,
        response_payload.get("turn_id"),
        user_id,
        action_type,
        classification.intent,
        classification.affect,
        inference_concept,
        confidence,
        len(retrieval.chunks),
        mastery_delta,
        bool(policy_decision),
        srl_mode,
        cold_start_triggered,
    )

    return response_payload
