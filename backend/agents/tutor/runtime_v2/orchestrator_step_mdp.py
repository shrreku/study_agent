from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import os

from ..constants import logger
from ..config import get_tutor_config
from ..state import TutorSessionPolicy
from ..state_machine import TutorStateManager
from ..policy import (
    level_for_mastery,
    needs_cold_start,
    select_focus_concept_with_prereqs,
)
from ..knowledge import fetch_mastery_map, fetch_prereq_chain
from ..persistence import get_session_state, get_recent_turns, insert_turn, update_session
from ..context_model import TutorContext
from ..tools import turn_controls, turn_classification, history_tools
from ..mdp.session import (
    SessionState,
    SessionObservation,
    SessionMDPAction,
    build_session_state_from_session,
    build_session_observation,
    apply_session_transition,
)
from ..mdp.concept import (
    ConceptState,
    ConceptObservation,
    build_concept_state_from_session,
    build_concept_observation,
    apply_concept_transition,
)
from ..mdp.actions import ConceptMDPAction
from ..mdp.policy import (
    make_session_policy,
    make_concept_policy,
    make_tutor_step_policy,
)
from ..mdp.adapters import (
    session_plan_from_policy,
    session_plan_to_policy,
    concept_plan_from_policy,
    concept_plan_to_policy,
)
from ..mdp.plans import ConceptPlanStep
from ..mdp.tutor import (
    TutorStepState,
    TutorStepObservation,
    TutorStepMDPAction,
    build_tutor_step_state,
    build_tutor_step_observation,
    apply_tutor_step_transition,
)
from ..mdp.tools_factory import (
    make_session_planner_tool,
    make_concept_planner_tool,
    make_step_executor_tool,
    make_quiz_evaluator_tool,
)

from .context import (
    TurnContext,
    ClassificationContext,
    ConceptContext,
)
from ..runtime.utils import normalize_single_concept_label


@dataclass
class ParsedStepPayload:
    """Normalized view of step-by-step payload controls and overrides."""

    agent_action_mode: str
    step_control_type: Optional[str]
    step_control_params: Dict[str, Any]
    override_type: Optional[str]
    override_params: Dict[str, Any]
    confirmed_action: Optional[str]
    mcq_answer: Optional[Dict[str, Any]]
    is_control_turn: bool
    canonical_control_label: Optional[str]
    message_for_classification: str


def run_step_mdp_turn(ctx: TurnContext, cur: Any) -> Dict[str, Any]:
    """Execute a single step-by-step turn using the SRL MDP stack.

    Stage 02 are wired here (state load, payload parsing, classification,
    concept & mastery context). For now, execution is delegated to the
    legacy step-by-step SRL runtime to preserve behaviour while we
    incrementally migrate logic into the MDP orchestrator.
    """

    config = get_tutor_config()
    logger.info(
        "step_mdp_runtime_called",
        extra={"session_id": ctx.session_id},
    )

    # === Stage 0: Load session + policy state ===
    session_state, policy_state, state_manager = _load_session_and_policy_state(
        cur=cur,
        session_id=ctx.session_id,
    )

    # === Stage 1: Parse payload + classification ===
    parsed = _parse_step_payload(ctx)
    classification = _classify_turn(
        ctx=ctx,
        session_state=session_state,
        parsed=parsed,
    )

    # === Stage 2: Concept & mastery context ===
    mastery_map, concept_ctx = _build_concept_and_mastery_context(
        ctx=ctx,
        cur=cur,
        policy_state=policy_state,
        classification=classification,
        session_state=session_state,
    )

    # Build TutorContext and recent turns for upcoming MDP wiring.
    last_turns = _load_recent_turns(cur, ctx.session_id)
    tutor_context = _build_tutor_context(
        ctx=ctx,
        policy_state=policy_state,
        classification=classification,
        concept_ctx=concept_ctx,
        mastery_map=mastery_map,
        last_turns=last_turns,
    )

    # === Session MDP: build state/observation and ensure SessionPlan ===
    session_planner = make_session_planner_tool(config)
    session_policy = make_session_policy(config)

    session_state_mdp: SessionState = build_session_state_from_session(
        policy_state=policy_state,
        tutor_context=tutor_context,
        mastery_map=mastery_map,
    )
    session_obs: SessionObservation = build_session_observation(session_state_mdp)

    session_plan = session_plan_from_policy(policy_state)
    if session_plan is None or not session_plan.entries:
        learning_targets = list(ctx.target_concepts or session_state.get("target_concepts", []))
        if learning_targets:
            strategy = getattr(policy_state, "session_strategy", None) or "learning_path"
            session_plan = session_planner(
                user_id=ctx.user_id,
                session_id=ctx.session_id,
                strategy=strategy,
                target_concepts=learning_targets,
                mastery_map=mastery_map,
            )
            session_plan_to_policy(session_plan, policy_state)
            policy_state.session_plan_index = 0
            session_state_mdp = build_session_state_from_session(
                policy_state=policy_state,
                tutor_context=tutor_context,
                mastery_map=mastery_map,
            )
            session_obs = build_session_observation(session_state_mdp)

    session_action: SessionMDPAction = session_policy.decide(
        observation=session_obs,
        session_plan=session_plan,
    )

    session_outcome = apply_session_transition(
        prev_state=session_state_mdp,
        mdp_action=session_action,
        concept_termination_reason=None,
    )

    try:
        policy_state.session_plan_index = int(session_outcome.state.plan_index or 0)
    except Exception:
        pass

    if session_outcome.terminated:
        messages = [
            {
                "role": "assistant",
                "content": "Weve completed your current study plan for this session. Feel free to start a new session or ask another question.",
            }
        ]
        return {
            "messages": messages,
            "ui_mode": "free_text",
            "mcq_payload": None,
            "agent_action_mode": "step_by_step",
            "debug": {
                "session_mdp_action": session_action.value,
                "session_terminated": True,
                "session_termination_reason": session_outcome.termination_reason,
            },
        }

    current_concept_id = session_outcome.state.current_concept_id or concept_ctx.focus_concept
    if not current_concept_id:
        messages = [
            {
                "role": "assistant",
                "content": "Tell me which concept you would like to study and I will help you with it.",
            }
        ]
        return {
            "messages": messages,
            "ui_mode": "free_text",
            "mcq_payload": None,
            "agent_action_mode": "step_by_step",
            "debug": {
                "session_mdp_action": session_action.value,
                "session_terminated": False,
                "missing_concept": True,
            },
        }

    target_mastery: Optional[float] = None
    if session_plan is not None:
        try:
            for entry in session_plan.entries:
                if entry.concept_id == current_concept_id:
                    target_mastery = entry.target_mastery
                    break
        except Exception:
            target_mastery = None
    if target_mastery is None:
        try:
            target_mastery = float(os.getenv("TUTOR_STEP_SRL_TARGET_MASTERY", "0.8") or 0.8)
        except Exception:
            target_mastery = 0.8

    concept_state: ConceptState = build_concept_state_from_session(
        policy_state=policy_state,
        tutor_context=tutor_context,
    )
    concept_state.concept_id = current_concept_id
    concept_state.target_mastery = target_mastery

    concept_plan = concept_plan_from_policy(policy_state)
    concept_planner = make_concept_planner_tool(config)
    if (
        concept_plan is None
        or concept_plan.concept_id != current_concept_id
        or not concept_plan.steps
    ):
        planning_obs: Dict[str, Any] = {
            "student_level": concept_ctx.concept_level,
            "mastery": (mastery_map.get(current_concept_id) or {}).get("mastery"),
            "recent_turns": last_turns,
        }
        concept_plan = concept_planner(
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            concept_id=current_concept_id,
            target_mastery=target_mastery,
            context_obs=planning_obs,
        )
        concept_plan_to_policy(concept_plan, policy_state)
        policy_state.srl_plan_step_index = 0
        concept_state.plan_index = 0

    last_intent = classification.intent
    last_affect = classification.affect
    last_action_type = "unknown"
    last_control_type = parsed.canonical_control_label
    concept_obs: ConceptObservation = build_concept_observation(
        state=concept_state,
        last_intent=last_intent,
        last_affect=last_affect,
        last_action_type=last_action_type,
        last_control_type=last_control_type,
    )

    concept_policy = make_concept_policy(config)
    concept_action: ConceptMDPAction = concept_policy.decide(
        observation=concept_obs,
        concept_plan=concept_plan,
    )

    # Tutor-level MDP: build a lightweight state/observation view over the
    # current SRL plan cursor and feedback signals. For now this is used for
    # logging and analysis; it does not change control flow.
    tutor_state: TutorStepState = build_tutor_step_state(
        policy_state=policy_state,
        concept_plan=concept_plan,
        concept_action=concept_action,
        session_id=ctx.session_id,
        user_id=ctx.user_id,
        concept_id=current_concept_id,
        last_intent=last_intent,
        last_affect=last_affect,
        last_control_type=last_control_type,
    )
    tutor_obs: TutorStepObservation = build_tutor_step_observation(state=tutor_state)
    tutor_policy = make_tutor_step_policy(config)
    tutor_action: TutorStepMDPAction = tutor_policy.decide(
        observation=tutor_obs,
        concept_action=concept_action,
    )
    tutor_outcome = apply_tutor_step_transition(prev_state=tutor_state, mdp_action=tutor_action)

    step_executor = make_step_executor_tool(config)
    quiz_evaluator = make_quiz_evaluator_tool(config)

    # Decide which behaviour to take this turn (concept-level end, replan, or execute step).
    messages: List[Dict[str, Any]] = []
    ui_mode: str = "free_text"
    mcq_payload: Optional[Dict[str, Any]] = None
    debug_payload: Dict[str, Any] = {}

    step: Optional[ConceptPlanStep] = None
    execute_step_index: Optional[int] = None
    steps: List[ConceptPlanStep] = list(concept_plan.steps or [])

    # Branch 1: concept-level actions that advance or terminate the concept.
    if concept_action in {
        ConceptMDPAction.ADVANCE_CONCEPT,
        ConceptMDPAction.TERMINATE_CONCEPT,
    }:
        if concept_action is ConceptMDPAction.ADVANCE_CONCEPT:
            messages = [
                {
                    "role": "assistant",
                    "content": "Let's wrap up this concept and move on. If you'd like to revisit it later, you can always come back.",
                }
            ]
        else:
            messages = [
                {
                    "role": "assistant",
                    "content": "Let's end the session here. You can start a new session whenever you're ready to continue.",
                }
            ]
    # Branch 2: tutor-step layer requests a concept-level replan.
    elif tutor_action == TutorStepMDPAction.REPLAN_CONCEPT:
        planning_obs: Dict[str, Any] = {
            "student_level": concept_ctx.concept_level,
            "mastery": (mastery_map.get(current_concept_id) or {}).get("mastery"),
            "recent_turns": last_turns,
        }
        concept_plan = concept_planner(
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            concept_id=current_concept_id,
            target_mastery=target_mastery,
            context_obs=planning_obs,
        )
        concept_plan_to_policy(concept_plan, policy_state)
        policy_state.srl_plan_step_index = 0
        concept_state.plan_index = 0

        messages = [
            {
                "role": "assistant",
                "content": "I'll adjust our plan for this concept so it better fits where you are right now.",
            }
        ]
    # Branch 3: execute the next SRL plan step.
    else:
        idx = concept_state.plan_index or 0
        if 0 <= idx < len(steps):
            step = steps[idx]
            execute_step_index = idx

        short_history = history_tools.build_short_context(last_turns, max_chars=1000)
        context_obs: Dict[str, Any] = {
            "student_level": concept_ctx.concept_level,
            "retrieval_chunks": [],
            "student_message": ctx.message,
            "recent_history": short_history,
        }

        if step is not None:
            exec_result = step_executor(
                session_id=ctx.session_id,
                user_id=ctx.user_id,
                concept_id=current_concept_id,
                step=step,
                context_obs=context_obs,
            )
            messages = list(exec_result.get("messages") or [])
            ui_mode = str(exec_result.get("ui_mode") or "free_text")
            mcq_payload = exec_result.get("mcq_payload")
            debug_payload = dict(exec_result.get("debug") or {})

            if ui_mode == "mcq" and isinstance(mcq_payload, dict):
                try:
                    policy_state.last_mcq = dict(mcq_payload)
                except Exception:
                    pass

    mcq_outcome: Optional[Dict[str, Any]] = None
    quiz_delta: Optional[float] = None
    if parsed.mcq_answer is not None and isinstance(policy_state.last_mcq, dict):
        last_mcq = policy_state.last_mcq or {}
        question = last_mcq
        user_answer = parsed.mcq_answer
        correct_answer = last_mcq.get("correct_option_id")
        mcq_outcome, quiz_delta = quiz_evaluator(
            question=question,
            user_answer=user_answer,
            correct_answer=correct_answer,
        )

    current_mastery_raw = (mastery_map.get(current_concept_id) or {}).get("mastery")
    try:
        mastery_before = float(current_mastery_raw) if current_mastery_raw is not None else 0.0
    except Exception:
        mastery_before = 0.0

    delta_learn_step = 0.03
    delta_quiz_correct = 0.2
    delta_quiz_wrong = -0.15

    mastery_delta: Optional[float] = None
    if mcq_outcome is not None:
        if mcq_outcome.get("answer_correct") is True:
            mastery_delta = delta_quiz_correct
        elif mcq_outcome.get("answer_correct") is False:
            mastery_delta = delta_quiz_wrong
    elif step is not None and step.step_type != "QUIZ_MCQ":
        mastery_delta = delta_learn_step

    if mastery_delta is None:
        mastery_delta = 0.0

    post_mastery = mastery_before + mastery_delta
    if post_mastery < 0.0:
        post_mastery = 0.0
    if post_mastery > 1.0:
        post_mastery = 1.0

    if current_concept_id not in mastery_map:
        mastery_map[current_concept_id] = {"mastery": post_mastery}
    else:
        mastery_map[current_concept_id]["mastery"] = post_mastery

    # Advance the local concept plan cursor when a SRL step was executed.
    if execute_step_index is not None:
        try:
            next_index = execute_step_index + 1
        except Exception:
            next_index = concept_state.plan_index or 0
        total_steps = len(steps)
        if next_index < 0:
            next_index = 0
        if total_steps >= 0 and next_index > total_steps:
            next_index = total_steps
        concept_state.plan_index = next_index

    concept_outcome = apply_concept_transition(
        prev_state=concept_state,
        mdp_action=concept_action,
        mastery_delta=mastery_delta,
        quiz_delta=quiz_delta,
        mcq_outcome=mcq_outcome,
        control_type=parsed.canonical_control_label,
        post_mastery=post_mastery,
        requested_override_type=parsed.override_type,
        step_control_type=parsed.step_control_type,
        last_intent=last_intent,
        last_affect=last_affect,
        last_action_type=step.step_type if step is not None else last_action_type,
    )

    try:
        policy_state.concept_episode_step_count = concept_outcome.state.step_count
        policy_state.concept_episode_quiz_correct = concept_outcome.state.quiz_correct
        policy_state.concept_episode_quiz_wrong = concept_outcome.state.quiz_wrong
        policy_state.concept_episode_last_control_type = concept_outcome.state.last_control_type
        policy_state.srl_plan_step_index = concept_outcome.state.plan_index
        policy_state.concept_episode_concept = current_concept_id
    except Exception:
        pass

    try:
        state_manager.update_action("step")
    except Exception:
        pass

    confidence = 1.0
    source_chunk_ids: List[str] = []
    mastery_delta_for_turn = mastery_delta

    try:
        insert_turn(
            cur,
            session_id=ctx.session_id,
            turn_index=ctx.turn_index,
            user_text=ctx.message,
            intent=classification.intent,
            affect=classification.affect,
            concept=current_concept_id,
            action_type="step",
            response_text="\n".join(m.get("content", "") for m in messages) if messages else "",
            source_chunk_ids=source_chunk_ids,
            confidence=confidence,
            mastery_delta=mastery_delta_for_turn,
            model_id=None,
            model_name=None,
            tool_calls={"step_executor": True},
            retrieval_metadata={},
            policy_trace={},
        )
    except Exception:
        pass

    try:
        update_session(
            cur,
            session_id=ctx.session_id,
            concept=current_concept_id,
            action_type="step",
            policy=policy_state.to_dict(),
        )
    except Exception:
        pass

    debug_payload.update(
        {
            "session_mdp_action": session_action.value,
            "concept_mdp_action": concept_action.value,
            "tutor_mdp_action": tutor_action.value,
            "mastery_before": mastery_before,
            "mastery_after": post_mastery,
            "concept_terminated": concept_outcome.terminated,
            "concept_termination_reason": concept_outcome.termination_reason,
        }
    )

    return {
        "messages": messages,
        "ui_mode": ui_mode,
        "mcq_payload": mcq_payload,
        "agent_action_mode": "step_by_step",
        "debug": debug_payload,
    }


def _load_session_and_policy_state(
    *,
    cur: Any,
    session_id: str,
) -> Tuple[Dict[str, Any], TutorSessionPolicy, TutorStateManager]:
    """Load raw session_state, policy_state, and state machine."""

    session_state = get_session_state(cur, session_id)
    policy_raw = session_state.get("policy") or {}
    policy_state = TutorSessionPolicy.from_dict(policy_raw)
    state_manager = TutorStateManager.from_dict(policy_state.state_machine)

    try:
        logger.info(
            "step_mdp_state_loaded",
            extra={
                "session_id": session_id,
                "user_id": getattr(policy_state, "user_id", None),
                "state": state_manager.current_state.value,
            },
        )
    except Exception:
        pass

    return session_state, policy_state, state_manager


def _parse_step_payload(ctx: TurnContext) -> ParsedStepPayload:
    """Parse and normalize step-by-step payload into a canonical structure."""

    parsed = turn_controls.parse_turn_controls(
        ctx.payload or {},
        message=ctx.message,
        default_mode="step_by_step",
    )

    return ParsedStepPayload(
        agent_action_mode=parsed.agent_action_mode,
        step_control_type=parsed.step_control_type,
        step_control_params=parsed.step_control_params,
        override_type=parsed.override_type,
        override_params=parsed.override_params,
        confirmed_action=parsed.confirmed_action,
        mcq_answer=parsed.mcq_answer,
        is_control_turn=parsed.is_control_turn,
        canonical_control_label=parsed.canonical_control_label,
        message_for_classification=parsed.message_for_classification,
    )


def _classify_turn(
    *,
    ctx: TurnContext,
    session_state: Dict[str, Any],
    parsed: ParsedStepPayload,
) -> ClassificationContext:
    """Run classifier or synthesize a control-turn classification."""

    classifier = turn_classification.TurnClassifier()
    return classifier.classify(ctx, session_state, parsed)


def _build_concept_and_mastery_context(
    *,
    ctx: TurnContext,
    cur: Any,
    policy_state: TutorSessionPolicy,
    classification: ClassificationContext,
    session_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], ConceptContext]:
    """Build mastery map, learning path, and ConceptContext."""

    learning_targets = ctx.target_concepts or session_state.get("target_concepts", [])
    mastery_map = fetch_mastery_map(cur, ctx.user_id)

    primary_concept = classification.concept
    if primary_concept:
        seed_concepts: List[str] = [primary_concept] + list(learning_targets)
    else:
        seed_concepts = list(learning_targets)

    learning_path = fetch_prereq_chain(seed_concepts)

    classification_for_focus: Dict[str, Any] = {
        "intent": classification.intent,
        "affect": classification.affect,
        "concept": classification.concept,
        "confidence": classification.confidence,
    }

    focus_concept, prereq_check = select_focus_concept_with_prereqs(
        classification_for_focus,
        learning_path,
        mastery_map,
        learning_targets,
        ctx.user_id,
        enable_prereq_check=True,
    )

    if getattr(policy_state, "phase", None) == "orientation" and getattr(
        policy_state, "pending_concept", None
    ):
        focus_concept = policy_state.pending_concept

    if focus_concept:
        focus_concept = normalize_single_concept_label(focus_concept) or focus_concept

    concept_level = level_for_mastery((mastery_map.get(focus_concept) or {}).get("mastery"))

    concepts = ConceptContext(
        focus_concept=focus_concept,
        concept_level=concept_level,
        learning_path=learning_path,
        learning_targets=list(learning_targets),
        mastery_map=mastery_map,
        prereq_check=prereq_check,
    )

    try:
        logger.info(
            "step_mdp_concept_mastery",
            extra={
                "session_id": ctx.session_id,
                "user_id": ctx.user_id,
                "classified_concept": classification.concept,
                "focus_concept": concepts.focus_concept,
                "concept_level": concepts.concept_level,
                "learning_targets": concepts.learning_targets,
                "mastery_count": len(mastery_map),
                "has_prereq_check": prereq_check is not None,
            },
        )
    except Exception:
        pass

    return mastery_map, concepts


def _load_recent_turns(cur: Any, session_id: str) -> List[Dict[str, Any]]:
    """Fetch recent turns with defensive defaults."""

    try:
        return get_recent_turns(cur, session_id, limit=10)
    except Exception:
        return []


def _build_tutor_context(
    *,
    ctx: TurnContext,
    policy_state: TutorSessionPolicy,
    classification: ClassificationContext,
    concept_ctx: ConceptContext,
    mastery_map: Dict[str, Any],
    last_turns: List[Dict[str, Any]],
) -> TutorContext:
    """Build TutorContext view for policies and tools."""

    try:
        session_summary = getattr(policy_state, "session_summary", "")
    except Exception:
        session_summary = ""

    cold_start_check = False
    try:
        if (
            os.getenv("TUTOR_COLD_START_ENABLED", "true").strip().lower()
            == "true"
        ):
            cold_start_check = needs_cold_start(
                concept_ctx.focus_concept,
                mastery_map,
                policy_state,
            )
    except Exception:
        cold_start_check = False

    tutor_context = TutorContext(
        session_id=ctx.session_id,
        user_id=ctx.user_id,
        turn_index=ctx.turn_index,
        message=ctx.message,
        intent=classification.intent,
        affect=classification.affect,
        inferred_concept=classification.concept,
        current_state=TutorStateManager.from_dict(policy_state.state_machine).current_state,
        focus_concept=concept_ctx.focus_concept,
        concept_level=concept_ctx.concept_level,
        mastery_map=mastery_map,
        prerequisites=getattr(concept_ctx.prereq_check, "missing_prereqs", [])
        if concept_ctx.prereq_check
        else [],
        learning_path=concept_ctx.learning_path,
        retrieval_chunks=[],
        recent_turns=last_turns,
        session_summary=session_summary,
        recent_mcq_outcomes=list(getattr(policy_state, "recent_mcq_outcomes", []) or []),
        current_plan=getattr(policy_state, "srl_plan", None),
        plan_step_index=getattr(policy_state, "srl_plan_step_index", 0),
        cold_start_eligible=cold_start_check,
    )

    return tutor_context

