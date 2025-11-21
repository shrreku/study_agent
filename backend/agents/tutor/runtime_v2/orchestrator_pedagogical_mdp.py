from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..constants import logger
from ..state import TutorSessionPolicy
from ..context_model import TutorContext
from ..runtime.context import TurnContext, ClassificationContext, ConceptContext
from ..knowledge import fetch_mastery_map, fetch_prereq_chain
from ..persistence import (
    get_session_state,
    get_recent_turns,
    insert_turn,
    update_session,
)
from ..tools.turn_controls import parse_turn_controls
from ..tools.turn_classification import TurnClassifier
from ..tools.history_tools import build_short_context
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
from ..mdp.adapters import (
    session_plan_from_policy,
    session_plan_to_policy,
    concept_plan_from_policy,
    concept_plan_to_policy,
)
from ..mdp.plans import ConceptPlan
from ..mdp.pedagogical_tutor import (
    PedagogicalTutorState,
    PedagogicalTutorObservation,
    PedagogicalTutorAction,
    build_pedagogical_tutor_state,
    build_pedagogical_tutor_observation,
    apply_pedagogical_transition,
)
from ..mdp.policy import (
    make_session_policy,
    make_concept_policy,
    make_pedagogical_tutor_policy,
)
from ..mdp.tools_factory import (
    make_session_planner_tool,
    make_concept_planner_tool,
    make_mastery_estimator_tool,
    make_quiz_evaluator_tool,
    make_pedagogical_response_tool,
)
from ..rl_pedagogical_logging import (
    build_pedagogical_step_event,
    pedagogical_step_event_to_dict,
)


@dataclass
class ParsedTurn:
    controls: Any
    classification: ClassificationContext
    mastery_map: Dict[str, Dict[str, Any]]
    recent_turns: List[Dict[str, Any]]
    tutor_context: TutorContext
    concept_ctx: ConceptContext


def _build_tutor_context(
    *,
    ctx: TurnContext,
    policy_state: TutorSessionPolicy,
    classification: ClassificationContext,
    mastery_map: Dict[str, Dict[str, Any]],
    recent_turns: List[Dict[str, Any]],
    learning_targets: List[str],
) -> TutorContext:
    from ..state_machine import TutorState

    last_concept = policy_state.concept_episode_concept or classification.concept
    prereqs = fetch_prereq_chain(learning_targets)

    tutor_ctx = TutorContext(
        session_id=ctx.session_id,
        user_id=ctx.user_id,
        turn_index=ctx.turn_index,
        message=ctx.message,
        intent=classification.intent,
        affect=classification.affect,
        inferred_concept=classification.concept,
        current_state=TutorState.TEACHING,
        focus_concept=last_concept,
        concept_level=policy_state.focus_level or "unknown",
        mastery_map=mastery_map,
        prerequisites=prereqs,
        learning_path=list(policy_state.learning_path or []),
        retrieval_chunks=[],
        recent_turns=recent_turns,
        session_summary=policy_state.session_summary or "",
        recent_mcq_outcomes=list(policy_state.recent_mcq_outcomes or []),
        current_plan=None,
        plan_step_index=0,
        cold_start_eligible=False,
        turn_signals=None,
        phase_suggestion=None,
        action_suggestion=None,
        retrieval_suggestion=None,
    )

    return tutor_ctx


def _build_concept_context(
    *,
    tutor_context: TutorContext,
    learning_targets: List[str],
) -> ConceptContext:
    return ConceptContext(
        focus_concept=tutor_context.focus_concept,
        concept_level=tutor_context.concept_level,
        learning_path=tutor_context.learning_path,
        learning_targets=learning_targets,
        mastery_map=tutor_context.mastery_map,
        prereq_check=None,
    )


def run_pedagogical_mdp_turn(ctx: TurnContext, cur: Any) -> Dict[str, Any]:
    """Execute a single tutoring turn using the 3-layer pedagogical MDP stack.

    This orchestrator is self-contained and does not depend on legacy
    orchestrator implementations. It wires together:

    - Session MDP (concept selection over the session)
    - Concept MDP (macro decisions within a concept episode)
    - Pedagogical Tutor MDP (pedagogical moves for this turn)
    """

    logger.info(
        "pedagogical_mdp_runtime_called",
        extra={"session_id": ctx.session_id},
    )

    # === Stage 0: Load session + policy state ===
    raw_session = get_session_state(cur, ctx.session_id)
    raw_policy = raw_session.get("policy") or {}
    policy_state = TutorSessionPolicy.from_dict(raw_policy)
    learning_targets: List[str] = list(ctx.target_concepts or raw_session.get("target_concepts", []) or [])

    turn_index = ctx.turn_index
    recent_turns = get_recent_turns(cur, ctx.session_id, limit=6)

    # === Stage 1: Parse payload + classification ===
    controls = parse_turn_controls(ctx.payload or {}, message=ctx.message, default_mode="auto")

    try:
        logger.info(
            "tutor_pedagogical_controls",
            extra={
                "session_id": ctx.session_id,
                "turn_index": turn_index,
                "control_label": controls.canonical_control_label,
                "step_control_type": controls.step_control_type,
                "confirmed_action": controls.confirmed_action,
                "has_mcq_answer": controls.mcq_answer is not None,
                "override_type": controls.override_type,
            },
        )
    except Exception:
        pass

    classifier = TurnClassifier()
    classification = classifier.classify(
        ctx=ctx,
        session_state=raw_session,
        controls=controls,
    )

    # === Stage 2: Mastery and TutorContext ===
    mastery_map = fetch_mastery_map(cur, ctx.user_id)

    tutor_context = _build_tutor_context(
        ctx=ctx,
        policy_state=policy_state,
        classification=classification,
        mastery_map=mastery_map,
        recent_turns=recent_turns,
        learning_targets=learning_targets,
    )
    concept_ctx = _build_concept_context(tutor_context=tutor_context, learning_targets=learning_targets)

    short_history = build_short_context(recent_turns, max_chars=1000)

    # === Stage 3: Session MDP ===
    session_planner = make_session_planner_tool()
    session_policy = make_session_policy()

    session_plan = session_plan_from_policy(policy_state)
    if session_plan is None or not getattr(session_plan, "entries", None):
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
            try:
                entries = list(getattr(session_plan, "entries", []) or [])
                logger.info(
                    "tutor_session_plan_created",
                    extra={
                        "user_id": ctx.user_id,
                        "session_id": ctx.session_id,
                        "strategy": strategy,
                        "target_concepts_count": len(learning_targets),
                        "plan_length": len(entries),
                        "plan_first_concepts": [getattr(e, "concept_id", None) for e in entries[:3]],
                    },
                )
            except Exception:
                pass

    session_state_mdp: SessionState = build_session_state_from_session(
        policy_state=policy_state,
        tutor_context=tutor_context,
        mastery_map=mastery_map,
    )
    session_obs: SessionObservation = build_session_observation(session_state_mdp)

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
                "content": "We've completed your current study plan for this session.",
            }
        ]
        debug = {
            "session_mdp_action": session_action.value,
            "session_terminated": True,
            "session_termination_reason": session_outcome.termination_reason,
        }
        try:
            update_session(cur, ctx.session_id, None, "pedagogical_step", policy_state.to_dict())
        except Exception:
            pass
        return {
            "messages": messages,
            "ui_mode": "free_text",
            "mcq_payload": None,
            "agent_action_mode": "pedagogical_step_by_step",
            "debug": debug,
        }

    current_concept_id = session_outcome.state.current_concept_id or concept_ctx.focus_concept
    if not current_concept_id:
        messages = [
            {
                "role": "assistant",
                "content": "Tell me which concept you would like to study and I will help you with it.",
            }
        ]
        debug = {
            "session_mdp_action": session_action.value,
            "missing_concept": True,
        }
        return {
            "messages": messages,
            "ui_mode": "free_text",
            "mcq_payload": None,
            "agent_action_mode": "pedagogical_step_by_step",
            "debug": debug,
        }

    # === Stage 4: Concept MDP ===
    concept_state: ConceptState = build_concept_state_from_session(
        policy_state=policy_state,
        tutor_context=tutor_context,
    )
    concept_state.concept_id = current_concept_id

    target_mastery: Optional[float] = None
    if session_plan is not None:
        try:
            for entry in session_plan.entries:
                if entry.concept_id == current_concept_id:
                    target_mastery = entry.target_mastery
                    break
        except Exception:
            target_mastery = None
    concept_state.target_mastery = target_mastery

    concept_plan = concept_plan_from_policy(policy_state)
    concept_planner = make_concept_planner_tool()
    if (
        concept_plan is None
        or concept_plan.concept_id != current_concept_id
        or not getattr(concept_plan, "steps", None)
    ):
        planning_obs: Dict[str, Any] = tutor_context.to_planning_observation()
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
        try:
            steps = list(getattr(concept_plan, "steps", []) or [])
            logger.info(
                "tutor_concept_plan_created",
                extra={
                    "user_id": ctx.user_id,
                    "session_id": ctx.session_id,
                    "concept_id": current_concept_id,
                    "plan_id": getattr(concept_plan, "plan_id", None),
                    "plan_length": len(steps),
                    "target_mastery": target_mastery,
                },
            )
            logger.info(
                "tutor_concept_plan_started",
                extra={
                    "user_id": ctx.user_id,
                    "session_id": ctx.session_id,
                    "concept_id": current_concept_id,
                    "plan_id": getattr(concept_plan, "plan_id", None),
                    "plan_index": concept_state.plan_index,
                },
            )
        except Exception:
            pass

    last_intent = classification.intent
    last_affect = classification.affect
    last_action_type = "unknown"
    last_control_type = controls.canonical_control_label

    concept_obs: ConceptObservation = build_concept_observation(
        state=concept_state,
        last_intent=last_intent,
        last_affect=last_affect,
        last_action_type=last_action_type,
        last_control_type=last_control_type,
    )

    concept_policy = make_concept_policy()
    concept_action: ConceptMDPAction = concept_policy.decide(
        observation=concept_obs,
        concept_plan=concept_plan,
    )

    if concept_action is ConceptMDPAction.REPLAN_CONCEPT:
        planning_obs: Dict[str, Any] = tutor_context.to_planning_observation()
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
        try:
            steps = list(getattr(concept_plan, "steps", []) or [])
            logger.info(
                "tutor_concept_plan_replanned",
                extra={
                    "user_id": ctx.user_id,
                    "session_id": ctx.session_id,
                    "concept_id": current_concept_id,
                    "plan_id": getattr(concept_plan, "plan_id", None),
                    "plan_length": len(steps),
                    "target_mastery": target_mastery,
                    "reason": "explicit_replan_control" if last_control_type == "replan_concept" else "policy_replan",
                },
            )
            logger.info(
                "tutor_concept_plan_started",
                extra={
                    "user_id": ctx.user_id,
                    "session_id": ctx.session_id,
                    "concept_id": current_concept_id,
                    "plan_id": getattr(concept_plan, "plan_id", None),
                    "plan_index": concept_state.plan_index,
                },
            )
        except Exception:
            pass

    # === Stage 5: Pedagogical Tutor MDP + tools ===
    messages: List[Dict[str, Any]] = []
    ui_mode: str = "free_text"
    mcq_payload: Optional[Dict[str, Any]] = None
    debug_payload: Dict[str, Any] = {}

    mastery_estimator = make_mastery_estimator_tool()
    quiz_evaluator = make_quiz_evaluator_tool()
    ped_response_tool = make_pedagogical_response_tool()

    mcq_outcome: Optional[Dict[str, Any]] = None
    quiz_delta: Optional[float] = None
    mastery_delta: Optional[float] = None

    if concept_action in {ConceptMDPAction.ADVANCE_CONCEPT, ConceptMDPAction.TERMINATE_CONCEPT}:
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
    else:
        # Build pedagogical state/observation
        episode_id = concept_state.episode_id
        plan_id = getattr(concept_plan, "plan_id", f"cp-{ctx.session_id}-{current_concept_id}")
        plan_length = len(list(concept_plan.steps or [])) if isinstance(concept_plan, ConceptPlan) else 0
        ped_state: PedagogicalTutorState = build_pedagogical_tutor_state(
            episode_id=episode_id,
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            concept_id=current_concept_id,
            plan_id=plan_id,
            plan_index=concept_state.plan_index,
            plan_length=plan_length,
            phase=concept_state.quiz_phase or "instruction",
            mastery=concept_state.mastery,
            target_mastery=concept_state.target_mastery,
            last_intent=last_intent,
            last_affect=last_affect,
            last_control_type=last_control_type,
            awaiting_mcq_answer=False,
            last_mcq_answered=False,
            last_quiz_correct=None,
            previous_state=None,
            last_pedagogical_action=None,
        )
        ped_obs: PedagogicalTutorObservation = build_pedagogical_tutor_observation(state=ped_state)

        ped_policy = make_pedagogical_tutor_policy()
        if concept_action is ConceptMDPAction.JUMP_TO_ASSESSMENT:
            ped_action = PedagogicalTutorAction.QUIZ_MCQ
        else:
            ped_action: PedagogicalTutorAction = ped_policy.decide(
                observation=ped_obs,
                concept_plan=concept_plan,
            )

        ped_outcome = apply_pedagogical_transition(
            prev_state=ped_state,
            mdp_action=ped_action,
            mastery_delta=None,
            quiz_delta=None,
            last_quiz_correct=None,
            awaiting_mcq_answer=False,
            last_mcq_answered=False,
            last_control_type=last_control_type,
        )

        context_obs: Dict[str, Any] = {
            "student_level": concept_ctx.concept_level,
            "student_message": ctx.message,
            "recent_history": short_history,
        }

        response = ped_response_tool(
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            concept_id=current_concept_id,
            pedagogical_action=ped_action,
            ped_state=ped_outcome.state,
            concept_plan=concept_plan,
            context_obs=context_obs,
        )
        messages = list(response.get("messages") or [])
        ui_mode = str(response.get("ui_mode") or "free_text")
        mcq_payload = response.get("mcq_payload")
        debug_payload = dict(response.get("debug") or {})

        if ui_mode == "mcq" and isinstance(mcq_payload, dict):
            try:
                policy_state.last_mcq = dict(mcq_payload)
            except Exception:
                pass

        # MCQ evaluation if user answered a previous question
        if controls.mcq_answer is not None and isinstance(policy_state.last_mcq, dict):
            last_mcq = policy_state.last_mcq or {}
            question = last_mcq
            user_answer = controls.mcq_answer
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

        recent_interactions: List[Dict[str, Any]] = [
            {
                "user_text": ctx.message,
                "pedagogical_action": ped_action.value,
                "mcq_outcome": mcq_outcome,
            }
        ]
        try:
            post_mastery, mastery_delta = mastery_estimator(
                concept_id=current_concept_id,
                mastery_before=mastery_before,
                recent_interactions=recent_interactions,
            )
        except Exception:
            post_mastery, mastery_delta = None, None

        if post_mastery is None:
            post_mastery = mastery_before
        if mastery_delta is None:
            mastery_delta = post_mastery - mastery_before

        if current_concept_id not in mastery_map:
            mastery_map[current_concept_id] = {"mastery": post_mastery}
        else:
            mastery_map[current_concept_id]["mastery"] = post_mastery

        concept_outcome = apply_concept_transition(
            prev_state=concept_state,
            mdp_action=concept_action,
            mastery_delta=mastery_delta,
            quiz_delta=quiz_delta,
            mcq_outcome=mcq_outcome,
            control_type=controls.canonical_control_label,
            post_mastery=post_mastery,
            requested_override_type=controls.override_type,
            step_control_type=controls.step_control_type,
            last_intent=last_intent,
            last_affect=last_affect,
            last_action_type=ped_action.value,
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

        # RL logging: emit a pedagogical tutor step event.
        try:
            ped_event = build_pedagogical_step_event(
                state=ped_outcome.state,
                observation=ped_outcome.observation,
                action=ped_outcome.action,
                reward=ped_outcome.reward,
                turn_index=turn_index,
                mcq_outcome=mcq_outcome,
                mastery_before=mastery_before,
                mastery_after=post_mastery,
            )
            logger.info(
                "pedagogical_tutor_step_event",
                extra={"pedagogical_step": pedagogical_step_event_to_dict(ped_event)},
            )
        except Exception:
            # Logging must not break the main tutor flow.
            pass

        # Persist turn + session
        confidence = 1.0
        source_chunk_ids: List[str] = []

        try:
            insert_turn(
                cur,
                session_id=ctx.session_id,
                turn_index=turn_index,
                user_text=ctx.message,
                intent=classification.intent,
                affect=classification.affect,
                concept=current_concept_id,
                action_type="pedagogical_step",
                response_text="\n".join(m.get("content", "") for m in messages) if messages else "",
                source_chunk_ids=source_chunk_ids,
                confidence=confidence,
                mastery_delta=mastery_delta,
                model_id=None,
                model_name=None,
                tool_calls={"pedagogical_response_tool": True},
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
                action_type="pedagogical_step",
                policy=policy_state.to_dict(),
            )
        except Exception:
            pass

        concept_plan_debug: Dict[str, Any] = {}
        if isinstance(concept_plan, ConceptPlan):
            try:
                steps_preview: List[Dict[str, Any]] = []
                for step in list(concept_plan.steps or [])[:10]:
                    steps_preview.append(
                        {
                            "step_id": getattr(step, "step_id", None),
                            "step_type": getattr(step, "step_type", None),
                            "subgoal": getattr(step, "subgoal", None),
                        }
                    )
                concept_plan_debug = {
                    "plan_id": getattr(concept_plan, "plan_id", None),
                    "concept_id": getattr(concept_plan, "concept_id", None),
                    "plan_length": len(list(concept_plan.steps or [])),
                    "plan_index": concept_outcome.state.plan_index,
                    "steps": steps_preview,
                }
            except Exception:
                concept_plan_debug = {}

        debug_payload.update(
            {
                "session_id": ctx.session_id,
                "current_concept_id": current_concept_id,
                "session_mdp_action": session_action.value,
                "concept_mdp_action": concept_action.value,
                "pedagogical_tutor_action": ped_action.value,
                "mastery_before": mastery_before,
                "mastery_after": post_mastery,
                "concept_terminated": concept_outcome.terminated,
                "concept_termination_reason": concept_outcome.termination_reason,
                "concept_plan": concept_plan_debug,
            }
        )

    return {
        "messages": messages,
        "ui_mode": ui_mode,
        "mcq_payload": mcq_payload,
        "agent_action_mode": "pedagogical_step_by_step",
        "debug": debug_payload,
    }
