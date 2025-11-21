"""Action selection and execution orchestration."""

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..constants import logger
from ..state import TutorSessionPolicy
from ..policy import level_for_mastery
from ..policy_decision import TutorPolicyDecision
from ..planning import TutorPlan
from ..responses import (
    build_orientation_response,
    build_prerequisite_review_prompt,
    build_followup_question,
    build_hint_response,
    build_reflect_response,
    build_review_response,
    generate_explain_response_with_plan,
)
from ..retrieval import retrieve_chunks
from ..tools.example_generator import ExampleGenerator
from ..srl_executor import execute_plan_steps
from ..actions.types import ActionResult
from ..config import get_tutor_config
from ..actions.handlers import (
    handle_cold_start,
    handle_override,
    handle_assessment,
    handle_reflection,
    handle_explain,
)
from .context import ClassificationContext, ConceptContext, RetrievalContext
from .utils import normalize_single_concept_label, looks_like_study_plan_request, should_use_multi_step, looks_like_confirmation

if TYPE_CHECKING:
    from ..decision_engine import ActionDecision


def select_action(
    *,
    cur: Any,
    session_id: str,
    classification: ClassificationContext,
    concepts: ConceptContext,
    retrieval: RetrievalContext,
    policy_state: TutorSessionPolicy,
    policy_decision: Optional[TutorPolicyDecision],
    srl_mode: bool,
    plan: Optional[TutorPlan],
    override_type: Optional[str],
    override_params: Dict[str, Any],
    payload: Dict[str, Any],
    dry_run: bool,
    agent_action_mode: Optional[str] = None,
    force_cold_start: bool = False,
    action_decision: Optional["ActionDecision"] = None,
    recent_turns: Optional[List[Dict[str, Any]]] = None,
) -> tuple[ActionResult, str, Optional[str]]:
    """Select and execute an action based on student state and policy.
    
    Returns (result, cause, applied_override_type).
    """
    message = payload.get("message", "")
    resource_id = payload.get("resource_id")
    mode_label = (agent_action_mode or payload.get("agent_action_mode") or "auto").strip().lower()
    
    logger.info(
        "tutor_tool_action_selection_start",
        extra={
            "session_id": session_id,
            "intent": classification.intent,
            "focus_concept": concepts.focus_concept,
            "policy_decision": policy_decision is not None,
            "srl_mode": srl_mode,
            "override_type": override_type,
            "has_action_decision": action_decision is not None,
        },
    )
    
    result: ActionResult
    cause: str = "default"
    applied_override_type: Optional[str] = None

    # === Override (highest priority) ===
    if override_type:
        result = handle_override(override_type, concepts.focus_concept, concepts.concept_level, retrieval.chunks, override_params)
        applied_override_type = result.action_type if result.action_type == override_type else None
        cause = "override_request"

    # === Action Decision (Unified Engine) ===
    elif action_decision:
        result, cause = _execute_action_decision(
            action_decision,
            concepts,
            retrieval,
            message,
            recent_turns=recent_turns,
        )
        # Cold start handling if decision flagged it
        if action_decision.cold_start and not result.cold_start_triggered:
             # Usually explain handles this via mode="cold_start" or similar, 
             # but let's ensure metadata is correct
             pass

    # === Legacy Fallbacks (Safety/Heuristics) ===
    # These should largely be superseded by ActionDecision, but kept for safety
    
    # === Orientation check ===
    elif (
        policy_state.phase == "teaching"
        and ((classification.intent == "greeting") or looks_like_study_plan_request(message))
    ):
        (
            response_text,
            confidence,
            source_chunk_ids,
            recommended_concept,
        ) = build_orientation_response(
            message=message,
            focus_concept=concepts.focus_concept,
            learning_targets=concepts.learning_targets,
            learning_path=concepts.learning_path,
            mastery_map=concepts.mastery_map,
            chunks=retrieval.chunks,
        )
        inference_concept_for_orientation = recommended_concept or concepts.focus_concept
        action_params = {
            key: value
            for key, value in {
                "concept": inference_concept_for_orientation,
                "level": concepts.concept_level,
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

    # === Prerequisite gating ===
    elif (
        concepts.prereq_check is not None
        and getattr(concepts.prereq_check, "should_review", False)
        and getattr(concepts.prereq_check, "missing_prereqs", [])
    ):
        prereq_concept = concepts.prereq_check.missing_prereqs[0]
        prereq_level = level_for_mastery((concepts.mastery_map.get(prereq_concept) or {}).get("mastery"))
        prereq_chunks = retrieve_chunks(prereq_concept, resource_id, ["definition", "explanation"])
        (
            response_text,
            confidence,
            source_chunk_ids,
        ) = build_prerequisite_review_prompt(
            target_concept=classification.concept,
            missing_prereqs=concepts.prereq_check.missing_prereqs,
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

    # === Cold start ===
    elif (
        force_cold_start
        or (
            os.getenv("TUTOR_COLD_START_ENABLED", "true").strip().lower() == "true"
            and not policy_decision  # Only check if not already decided by policy
        )
    ):
        should_cold_start = force_cold_start
        if not should_cold_start:
            from ..policy import needs_cold_start
            should_cold_start = needs_cold_start(concepts.focus_concept, concepts.mastery_map, policy_state)
        
        if should_cold_start:
            result = handle_cold_start(
                concepts.focus_concept,
                concepts.concept_level,
                retrieval.chunks,
                cur,
                session_id,
                policy_state,
                dry_run=dry_run,
            )
            cause = "cold_start"
        else:
             # Fallthrough for normal action selection
             result, cause, applied_override_type = _select_normal_action(
                classification,
                concepts,
                retrieval,
                policy_state,
                policy_decision,
                srl_mode,
                plan,
                override_type,
                override_params,
                message,
                resource_id,
                mode_label,
                recent_turns=recent_turns,
            )

    # === Normal action selection ===
    else:
        result, cause, applied_override_type = _select_normal_action(
            classification,
            concepts,
            retrieval,
            policy_state,
            policy_decision,
            srl_mode,
            plan,
            override_type,
            override_params,
            message,
            resource_id,
            mode_label,
        )

    return result, cause, applied_override_type


def _execute_action_decision(
    decision: "ActionDecision",
    concepts: ConceptContext,
    retrieval: RetrievalContext,
    message: str,
    recent_turns: Optional[List[Dict[str, Any]]] = None,
) -> tuple[ActionResult, str]:
    """Execute an ActionDecision from the unified engine."""
    action = decision.action.strip().lower()
    cause = f"decision_engine_{action}"
    
    if action == "explain":
        # Use handle_explain with optional plan integration
        if decision.generated_plan:
             # Use plan-aware explanation
             return _execute_srl_plan_single_step(
                 TutorPlan(**decision.generated_plan),
                 concepts,
                 retrieval,
                 0, # Step 0
                 override_type="explain"
             )
        else:
            result = handle_explain(
                concepts.focus_concept,
                concepts.concept_level,
                retrieval.chunks,
                mode="default"
            )
            
    elif action == "ask":
        result = handle_assessment(
            concepts.focus_concept,
            concepts.concept_level,
            retrieval.chunks,
            cause
        )
        
    elif action == "hint":
        result = handle_override(
            "hint",
            concepts.focus_concept,
            concepts.concept_level,
            retrieval.chunks,
            {}
        )
        
    elif action == "reflect":
        # Format recent history for context
        history_str = ""
        if recent_turns:
            # Take last 2 turns (Tutor question + Student answer)
            # But 'recent_turns' usually excludes current turn, so it has the Tutor question.
            # Let's format the last turn (Tutor's question)
            try:
                last_tutor = recent_turns[0] if recent_turns else {}
                if last_tutor.get("role") == "tutor":
                    history_str = f"Tutor: {last_tutor.get('response_text', '')}"
            except Exception:
                pass

        result = handle_reflection(
            concepts.focus_concept,
            concepts.concept_level,
            retrieval.chunks,
            message or "",
            history=history_str,
        )
        
    elif action == "review":
        result = handle_override(
            "review",
            concepts.focus_concept,
            concepts.concept_level,
            retrieval.chunks,
            {}
        )
        
    elif action == "preview":
        # Fallback for preview/closure
        result = ActionResult(
            action_type="explain",
            response_text="That wraps up our session! Let me know if you'd like to continue or review anything else.",
            confidence=0.9,
            source_chunk_ids=[],
            inference_concept=concepts.focus_concept,
            action_params={"mode": "closure"},
        )
        
    elif action == "orient":
        (
            response_text,
            confidence,
            source_chunk_ids,
            recommended_concept,
        ) = build_orientation_response(
            message=message,
            focus_concept=concepts.focus_concept,
            learning_targets=concepts.learning_targets,
            learning_path=concepts.learning_path,
            mastery_map=concepts.mastery_map,
            chunks=retrieval.chunks,
        )
        result = ActionResult(
            action_type="explain",
            response_text=response_text,
            confidence=confidence,
            source_chunk_ids=source_chunk_ids,
            inference_concept=recommended_concept or concepts.focus_concept,
            action_params={"mode": "orientation"},
        )
        
    else:
        # Default fallback
        result = handle_explain(
            concepts.focus_concept,
            concepts.concept_level,
            retrieval.chunks,
            mode="default"
        )
        
    return result, decision.rationale


def _select_normal_action(
    classification: ClassificationContext,
    concepts: ConceptContext,
    retrieval: RetrievalContext,
    policy_state: TutorSessionPolicy,
    policy_decision: Optional[TutorPolicyDecision],
    srl_mode: bool,
    plan: Optional[TutorPlan],
    override_type: Optional[str],
    override_params: Dict[str, Any],
    message: str,
    resource_id: Optional[str],
    agent_action_mode: str,
    recent_turns: Optional[List[Dict[str, Any]]] = None,
) -> tuple[ActionResult, str, Optional[str]]:
    """Select action in normal flow: policy-driven or heuristic fallback."""
    result: ActionResult
    cause: str = "default"
    applied_override_type: Optional[str] = None

    if policy_decision is not None:
        # LLM policy-driven action selection
        result, cause = _select_action_from_policy(
            policy_decision,
            concepts,
            retrieval,
            recent_turns=recent_turns,
        )
    else:
        # Heuristic fallback
        result, cause = _select_action_heuristic(
            classification,
            concepts,
            retrieval,
            policy_state,
            srl_mode,
            plan,
            message,
            resource_id,
            agent_action_mode,
            override_type=override_type,
            recent_turns=recent_turns,
        )

    return result, cause, applied_override_type


def _select_action_from_policy(
    policy_decision: TutorPolicyDecision,
    concepts: ConceptContext,
    retrieval: RetrievalContext,
    recent_turns: Optional[List[Dict[str, Any]]] = None,
) -> tuple[ActionResult, str]:
    """Map policy decision to concrete action."""
    na = (policy_decision.next_action or "").strip().lower() or "explain"
    mode_for_explain = (policy_decision.mode or "default").strip() or "default"

    if na == "ask":
        cause = "policy_llm_ask"
        result = handle_assessment(concepts.focus_concept, concepts.concept_level, retrieval.chunks, cause)

    elif na == "reflect" and retrieval.chunks:
        cause = "policy_llm_reflect"
        # Format history
        history_str = ""
        if recent_turns:
            try:
                last_tutor = recent_turns[0] if recent_turns else {}
                if last_tutor.get("role") == "tutor":
                    history_str = f"Tutor: {last_tutor.get('response_text', '')}"
            except Exception:
                pass
        result = handle_reflection(concepts.focus_concept, concepts.concept_level, retrieval.chunks, "", history=history_str)

    elif na == "hint" and retrieval.chunks:
        cause = "policy_llm_hint"
        result = handle_override("hint", concepts.focus_concept, concepts.concept_level, retrieval.chunks, {})

    elif na == "review" and retrieval.chunks:
        cause = "policy_llm_review"
        result = handle_override("review", concepts.focus_concept, concepts.concept_level, retrieval.chunks, {})

    else:
        cause = "policy_llm_explain"
        result = handle_explain(concepts.focus_concept, concepts.concept_level, retrieval.chunks, mode=mode_for_explain)

    return result, cause


def _select_action_heuristic(
    classification: ClassificationContext,
    concepts: ConceptContext,
    retrieval: RetrievalContext,
    policy_state: TutorSessionPolicy,
    srl_mode: bool,
    plan: Optional[TutorPlan],
    message: str,
    resource_id: Optional[str],
    agent_action_mode: str,
    override_type: Optional[str] = None,
    recent_turns: Optional[List[Dict[str, Any]]] = None,
) -> tuple[ActionResult, str]:
    """Heuristic action selection when LLM policy is disabled."""
    result: ActionResult
    cause: str = "default"

    mode_label = (agent_action_mode or "auto").strip().lower()

    wants_assessment_after_explain = (
        mode_label == "step_by_step"
        and policy_state.last_action == "explain"
        and classification.intent in {"answer", "reflection"}
        and retrieval.chunks
        and looks_like_confirmation(message)
    )

    # When an SRL plan is available, prefer executing the planned action in
    # step-by-step "plan" mode rather than relying on heuristics.
    if srl_mode and plan:
        result, cause = _execute_srl_plan(
            plan,
            concepts,
            retrieval,
            policy_state,
            classification.intent,
            agent_action_mode,
            override_type=override_type,
        )

    elif wants_assessment_after_explain:
        cause = "claimed_understanding_assess"
        result = handle_assessment(concepts.focus_concept, concepts.concept_level, retrieval.chunks, cause)

    elif (
        os.getenv("TUTOR_EXAMPLE_GENERATION_ENABLED", "false").strip().lower() == "true"
        and concepts.focus_concept
        and concepts.learning_path
    ):
        result, cause = _try_bridge_example(
            concepts,
            retrieval,
        )

    elif classification.intent == "answer" and retrieval.chunks:
        # Format history
        history_str = ""
        if recent_turns:
            try:
                last_tutor = recent_turns[0] if recent_turns else {}
                if last_tutor.get("role") == "tutor":
                    history_str = f"Tutor: {last_tutor.get('response_text', '')}"
            except Exception:
                pass
        result = handle_reflection(concepts.focus_concept, concepts.concept_level, retrieval.chunks, message, history=history_str)
        cause = "student_answer"

    elif classification.affect in {"confused", "unsure"} and retrieval.chunks:
        result = handle_explain(concepts.focus_concept, concepts.concept_level, retrieval.chunks, mode="confusion_support")
        cause = "affect_confused_explain_basics"

    else:
        result = handle_explain(concepts.focus_concept, concepts.concept_level, retrieval.chunks, mode="default")
        cause = "explain_default"

    return result, cause


def _execute_srl_plan(
    plan: TutorPlan,
    concepts: ConceptContext,
    retrieval: RetrievalContext,
    policy_state: TutorSessionPolicy,
    intent: str,
    agent_action_mode: str,
    override_type: Optional[str] = None,
) -> tuple[ActionResult, str]:
    """Execute SRL plan or fallback to single-step action.

    In step-by-step mode with an override_type, the override takes precedence
    over the planned action for this step, but the plan cursor still advances.
    """
    mode_label = (agent_action_mode or "auto").strip().lower()
    use_multi = False

    # Use unified TutorConfig instead of the legacy TUTOR_SRL_MULTI_STEP_EXECUTE
    # environment variable. Multi-step execution is only considered outside of
    # explicit step_by_step mode.
    try:
        config = get_tutor_config()
        multi_flag = bool(getattr(config, "enable_multi_step_execution", False))
    except Exception:
        multi_flag = False

    if mode_label != "step_by_step":
        use_multi = multi_flag and should_use_multi_step(intent, policy_state.phase)

    if use_multi:
        combined = execute_plan_steps(
            plan,
            concepts.focus_concept,
            concepts.concept_level,
            "",
            None,
        )
        result = ActionResult(
            action_type=str(combined.get("last_action") or "explain"),
            response_text=str(combined.get("text") or ""),
            confidence=float(combined.get("confidence") or 0.6),
            source_chunk_ids=list(combined.get("source_chunk_ids") or []),
            inference_concept=combined.get("inference_concept") or concepts.focus_concept,
            action_params={
                "concept": (combined.get("inference_concept") or concepts.focus_concept),
                "level": concepts.concept_level,
                "mode": "srl_plan_multi",
            },
        )
        return result, "srl_plan_multi_steps"
    else:
        # Single-step execution respects a per-session cursor so that
        # step-by-step mode can follow the same plan across turns.
        try:
            step_index = int(getattr(policy_state, "srl_plan_step_index", 0) or 0)
        except Exception:
            step_index = 0
        result, cause = _execute_srl_plan_single_step(
            plan, concepts, retrieval, step_index, override_type=override_type
        )
        try:
            policy_state.srl_plan_step_index = step_index + 1
        except Exception:
            pass
        return result, cause


def _execute_srl_plan_single_step(
    plan: TutorPlan,
    concepts: ConceptContext,
    retrieval: RetrievalContext,
    step_index: int,
    override_type: Optional[str] = None,
) -> tuple[ActionResult, str]:
    """Execute a single step from the SRL plan.

    The caller is responsible for threading a step_index cursor (e.g. from
    TutorSessionPolicy.srl_plan_step_index). If the index is out of range or
    no explicit step is available, we fall back to the plan's top-level
    intended_action.
    
    If override_type is provided, it takes precedence over the planned action.
    
    Loop prevention: If step_index >= len(steps), we've exhausted the plan
    and should switch to a safe fallback action (ask) to avoid loops.
    """
    steps = list(getattr(plan, "steps", []) or [])
    intended = None
    
    # Check if plan exhausted - prevent infinite loops
    if step_index >= len(steps) and not override_type:
        logger.info(
            "tutor_srl_plan_exhausted step_index=%s steps_count=%s fallback=ask",
            step_index,
            len(steps),
        )
        intended = "ask"
    # If override is provided, use it; otherwise use the planned action
    elif override_type:
        intended = override_type.strip().lower()
    elif steps and 0 <= step_index < len(steps):
        try:
            intended = str(steps[step_index].get("action") or "").lower()
        except Exception:
            intended = None
    
    if not intended:
        intended = (plan.intended_action or "").lower()

    if intended == "explain":
        (
            response_text,
            confidence,
            source_chunk_ids,
            inferred_concept_candidate,
        ) = generate_explain_response_with_plan(
            plan=plan,
            concept=concepts.focus_concept,
            level=concepts.concept_level,
            chunks=retrieval.chunks,
        )
        inference_concept = inferred_concept_candidate or concepts.focus_concept
        result = ActionResult(
            action_type="explain",
            response_text=response_text,
            confidence=confidence,
            source_chunk_ids=source_chunk_ids,
            inference_concept=inference_concept,
            action_params={
                "concept": inference_concept or concepts.focus_concept,
                "level": concepts.concept_level,
                "mode": "srl_plan",
            },
        )
        return result, "srl_plan_explain"

    elif intended == "ask":
        result = handle_assessment(
            concepts.focus_concept,
            concepts.concept_level,
            retrieval.chunks,
            "srl_plan_ask",
        )
        # Tag mode so downstream state/policy can distinguish SRL-plan-driven asks.
        if "mode" not in result.action_params:
            try:
                result.action_params["mode"] = "srl_plan"
            except Exception:
                pass
        return result, "srl_plan_ask"

    elif intended == "hint":
        response_text, confidence, source_chunk_ids = build_hint_response(
            concepts.focus_concept,
            concepts.concept_level,
            retrieval.chunks,
        )
        result = ActionResult(
            action_type="hint",
            response_text=response_text,
            confidence=confidence,
            source_chunk_ids=source_chunk_ids,
            inference_concept=concepts.focus_concept,
            action_params={
                "concept": concepts.focus_concept,
                "level": concepts.concept_level,
                "mode": "srl_plan",
            },
        )
        return result, "srl_plan_hint"

    elif intended == "reflect":
        response_text, confidence, source_chunk_ids = build_reflect_response(
            concepts.focus_concept,
            concepts.concept_level,
            retrieval.chunks,
            message="",
        )
        result = ActionResult(
            action_type="reflect",
            response_text=response_text,
            confidence=confidence,
            source_chunk_ids=source_chunk_ids,
            inference_concept=concepts.focus_concept,
            action_params={
                "concept": concepts.focus_concept,
                "level": concepts.concept_level,
                "mode": "srl_plan",
            },
        )
        return result, "srl_plan_reflect"

    elif intended == "review":
        response_text, confidence, source_chunk_ids = build_review_response(
            concepts.focus_concept,
            concepts.concept_level,
            retrieval.chunks,
        )
        result = ActionResult(
            action_type="review",
            response_text=response_text,
            confidence=confidence,
            source_chunk_ids=source_chunk_ids,
            inference_concept=concepts.focus_concept,
            action_params={
                "concept": concepts.focus_concept,
                "level": concepts.concept_level,
                "mode": "srl_plan",
            },
        )
        return result, "srl_plan_review"

    else:
        result = handle_explain(concepts.focus_concept, concepts.concept_level, retrieval.chunks, mode="default")
        return result, "srl_plan_default"


def _try_bridge_example(
    concepts: ConceptContext,
    retrieval: RetrievalContext,
) -> tuple[ActionResult, str]:
    """Try to generate a bridge example, fallback to basic explanation."""
    try:
        idx = concepts.learning_path.index(concepts.focus_concept)
    except (ValueError, TypeError):
        idx = -1

    from_concept = None
    if idx > 0:
        for c in reversed(concepts.learning_path[:idx]):
            try:
                m = float((concepts.mastery_map.get(c) or {}).get("mastery", 0.0) or 0.0)
            except Exception:
                m = 0.0
            if m >= 0.5:
                from_concept = c
                break
        if not from_concept:
            from_concept = concepts.learning_path[idx - 1]

    if from_concept:
        try:
            gen = ExampleGenerator()
            br = gen.generate_bridge_example(
                from_concept=from_concept,
                to_concept=concepts.focus_concept,
                student_level=concepts.concept_level,
                grounding_chunks=retrieval.chunks,
            )
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
                source_ids = [c.get("id") for c in (retrieval.chunks or []) if c.get("id")]
                result = ActionResult(
                    action_type="explain",
                    response_text=bridge_text,
                    confidence=float(br.confidence),
                    source_chunk_ids=source_ids,
                    inference_concept=concepts.focus_concept,
                    action_params={
                        "concept": concepts.focus_concept,
                        "level": concepts.concept_level,
                        "mode": "bridge_example",
                    },
                )
                return result, "bridge_example"
        except Exception:
            logger.exception("tutor_bridge_example_failed")

    result = handle_explain(concepts.focus_concept, concepts.concept_level, retrieval.chunks, mode="confusion_support")
    return result, "affect_confused_explain_basics"
