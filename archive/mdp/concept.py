from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import os

from .base import MDPState, MDPObservation, StepOutcome
from .actions import ConceptMDPAction
from ..state import TutorSessionPolicy
from ..context_model import TutorContext


@dataclass
class ConceptState(MDPState):
    """Internal concept-level MDP state for a single episode.

    This is derived from TutorSessionPolicy + TutorContext and mirrors the
    concept episode fields we already persist, but in a MDP-friendly shape.
    """

    episode_id: str
    session_id: str
    user_id: str
    concept_id: str

    target_mastery: Optional[float]
    mastery: Optional[float]
    mastery_start: Optional[float]

    plan_index: int
    quiz_phase: str
    quiz_question_index: int
    quiz_max_questions: int
    quiz_correct: int
    quiz_wrong: int

    step_count: int
    last_control_type: Optional[str]
    last_mdp_action: Optional[ConceptMDPAction]


@dataclass
class ConceptObservation(MDPObservation):
    """Filtered observation for the concept-level MDP.

    This is what a RL policy or analysis code would typically consume.
    """

    concept_id: str
    mastery: Optional[float]
    target_mastery: Optional[float]

    plan_index: int
    quiz_phase: str
    quiz_question_index: int
    quiz_max_questions: int
    quiz_correct: int
    quiz_wrong: int

    last_intent: str
    last_affect: str
    last_action_type: str
    last_control_type: Optional[str]


def _normalize_control_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    text = label.strip().lower()
    if not text:
        return None
    if text in {"continue", "next", "yes"}:
        return "continue"
    if text in {"re-plan", "replan", "replan_concept"}:
        return "replan_concept"
    if text == "skip_to_quiz":
        return "skip_to_quiz"
    if text in {"skip_to_next_concept", "skip_to_next"}:
        return "skip_to_next_concept"
    if text in {"end_session", "end"}:
        return "end_session"
    if text == "mcq_answer":
        return "mcq_answer"
    return text


def build_concept_state_from_session(
    *,
    policy_state: TutorSessionPolicy,
    tutor_context: TutorContext,
    episode_id_fallback: Optional[str] = None,
) -> ConceptState:
    """Best-effort ConceptState builder from session policy + context.

    Phase 1: this helper is introduced but not yet wired into the main
    runtime. It mirrors the data carried in TutorSessionPolicy so that
    later we can route all concept-level logic through this type.
    """

    concept_id = tutor_context.focus_concept or tutor_context.inferred_concept or "unknown"

    episode_id = getattr(policy_state, "concept_episode_id", None) or episode_id_fallback or (
        f"ce-{tutor_context.session_id}-{concept_id}"
    )

    # Mastery snapshot for the focus concept
    mastery: Optional[float] = None
    try:
        raw = (tutor_context.mastery_map.get(concept_id) or {}).get("mastery")
        if raw not in (None, ""):
            mastery = float(raw)
    except Exception:
        mastery = None

    return ConceptState(
        episode_id=episode_id,
        session_id=tutor_context.session_id,
        user_id=tutor_context.user_id,
        concept_id=concept_id,
        target_mastery=None,  # Filled in by callers when available
        mastery=mastery,
        mastery_start=getattr(policy_state, "concept_episode_mastery_start", None),
        plan_index=int(getattr(policy_state, "srl_plan_step_index", 0) or 0),
        quiz_phase=str(getattr(policy_state, "quiz_phase", "") or ""),
        quiz_question_index=int(getattr(policy_state, "quiz_question_index", 0) or 0),
        quiz_max_questions=int(getattr(policy_state, "quiz_max_questions", 0) or 0),
        quiz_correct=int(getattr(policy_state, "concept_episode_quiz_correct", 0) or 0),
        quiz_wrong=int(getattr(policy_state, "concept_episode_quiz_wrong", 0) or 0),
        step_count=int(getattr(policy_state, "concept_episode_step_count", 0) or 0),
        last_control_type=getattr(policy_state, "concept_episode_last_control_type", None),
        last_mdp_action=None,
    )


def build_concept_observation(
    *,
    state: ConceptState,
    last_intent: str,
    last_affect: str,
    last_action_type: str,
    last_control_type: Optional[str] = None,
) -> ConceptObservation:
    """Construct a ConceptObservation from ConceptState + light context."""

    return ConceptObservation(
        concept_id=state.concept_id,
        mastery=state.mastery,
        target_mastery=state.target_mastery,
        plan_index=state.plan_index,
        quiz_phase=state.quiz_phase,
        quiz_question_index=state.quiz_question_index,
        quiz_max_questions=state.quiz_max_questions,
        quiz_correct=state.quiz_correct,
        quiz_wrong=state.quiz_wrong,
        last_intent=last_intent,
        last_affect=last_affect,
        last_action_type=last_action_type,
        last_control_type=last_control_type,
    )


def apply_concept_transition(
    *,
    prev_state: ConceptState,
    mdp_action: ConceptMDPAction,
    mastery_delta: Optional[float],
    quiz_delta: Optional[float],
    mcq_outcome: Optional[Dict[str, Any]],
    control_type: Optional[str] = None,
    post_mastery: Optional[float] = None,
    requested_override_type: Optional[str] = None,
    step_control_type: Optional[str] = None,
    last_intent: str = "unknown",
    last_affect: str = "neutral",
    last_action_type: str = "unknown",
) -> StepOutcome[ConceptState, ConceptObservation, ConceptMDPAction]:
    """Apply a single concept-level MDP transition.

    This function owns the per-episode counters that are concept-MDP
    specific (step count, quiz counters, last control type / action).
    """

    state = ConceptState(**prev_state.__dict__)
    state.step_count = (state.step_count or 0) + 1
    state.last_mdp_action = mdp_action
    normalized_control = _normalize_control_label(control_type)
    state.last_control_type = normalized_control or state.last_control_type

    # Quiz counters from MCQ outcome, if any.
    if mcq_outcome is not None:
        answer_correct = mcq_outcome.get("answer_correct")
        if answer_correct is True:
            state.quiz_correct = (state.quiz_correct or 0) + 1
        elif answer_correct is False:
            state.quiz_wrong = (state.quiz_wrong or 0) + 1

    # Simple reward dictionary mirroring existing logging signals.
    reward: Dict[str, float] = {}
    if mastery_delta is not None:
        try:
            reward["mastery_delta"] = float(mastery_delta)
        except Exception:
            pass
    if quiz_delta is not None:
        try:
            reward["quiz_delta"] = float(quiz_delta)
        except Exception:
            pass

    # Compute termination using the same rules previously embedded in the
    # orchestrator: mastery-based, then max-steps, then explicit skip /
    # session-end overrides.
    terminated = False
    termination_reason: Optional[str] = None

    # 1) Mastery-based termination
    if (
        post_mastery is not None
        and state.target_mastery is not None
        and post_mastery >= state.target_mastery
    ):
        terminated = True
        termination_reason = "mastery_reached"

    # 2) Max-steps guardrail
    if not terminated:
        max_steps_raw = os.getenv("TUTOR_STEP_SRL_MAX_STEPS", "0") or "0"
        try:
            max_steps = int(max_steps_raw)
        except Exception:
            max_steps = 0
        if max_steps > 0 and (state.step_count or 0) >= max_steps:
            terminated = True
            termination_reason = "max_steps_reached"

    # 3) Explicit skip to next concept
    if not terminated and (
        requested_override_type == "step_next_concept"
        or control_type in {"skip_to_next_concept", "skip_to_next"}
    ):
        terminated = True
        termination_reason = "skip_next_concept"

    # 4) Explicit session end
    if not terminated and (
        requested_override_type == "session_end"
        or step_control_type == "end_session"
    ):
        terminated = True
        termination_reason = "session_end"

    obs = build_concept_observation(
        state=state,
        last_intent=last_intent,
        last_affect=last_affect,
        last_action_type=last_action_type,
        last_control_type=state.last_control_type,
    )

    return StepOutcome(
        state=state,
        observation=obs,
        action=mdp_action,
        reward=reward,
        terminated=terminated,
        termination_reason=termination_reason,
        info={},
    )
