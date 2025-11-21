from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .base import MDPState, MDPObservation, StepOutcome


class PedagogicalTutorAction(str, Enum):
    """Canonical action space for the pedagogical tutor MDP.

    These actions represent *how* the tutor teaches within the current concept
    on a single turn. Structural decisions (replan / skip concept / quiz phase)
    remain the responsibility of the concept-level MDP.
    """

    EXPLAIN = "EXPLAIN"
    ASK_QUESTION = "ASK_QUESTION"
    WORKED_EXAMPLE = "WORKED_EXAMPLE"
    GUIDED_PRACTICE = "GUIDED_PRACTICE"
    DEFINE_TERM = "DEFINE_TERM"
    REFLECTION_PROMPT = "REFLECTION_PROMPT"
    SUMMARY = "SUMMARY"
    QUIZ_MCQ = "QUIZ_MCQ"


@dataclass
class PedagogicalTutorState(MDPState):
    """Internal state for the pedagogical tutor MDP.

    One episode corresponds to a concept episode for a given
    (session_id, user_id, concept_id).
    """

    episode_id: str
    session_id: str
    user_id: str
    concept_id: str

    plan_id: str
    plan_index: int
    plan_length: int
    phase: str

    mastery: Optional[float]
    target_mastery: Optional[float]

    step_count: int

    last_intent: str
    last_affect: str
    last_control_type: Optional[str]

    last_pedagogical_action: Optional[PedagogicalTutorAction]

    awaiting_mcq_answer: bool
    last_mcq_answered: bool
    last_quiz_correct: Optional[bool]

    ped_action_counts: Dict[str, int] = field(default_factory=dict)
    recent_ped_actions: List[str] = field(default_factory=list)


@dataclass
class PedagogicalTutorObservation(MDPObservation):
    """Reduced feature view for the pedagogical tutor policy.

    This is derived from :class:`PedagogicalTutorState` plus a small amount of
    derived feedback such as mastery gap and simple history summaries.
    """

    plan_index: int
    plan_length: int
    at_end_of_plan: bool
    phase: str

    mastery: Optional[float]
    target_mastery: Optional[float]
    mastery_gap: Optional[float]

    last_intent: str
    last_affect: str
    last_control_type: Optional[str]

    awaiting_mcq_answer: bool
    last_mcq_answered: bool
    last_quiz_correct: Optional[bool]

    num_explain_steps: int
    num_practice_steps: int
    num_quiz_steps: int


def build_pedagogical_tutor_state(
    *,
    episode_id: str,
    session_id: str,
    user_id: str,
    concept_id: str,
    plan_id: str,
    plan_index: int,
    plan_length: int,
    phase: str,
    mastery: Optional[float],
    target_mastery: Optional[float],
    last_intent: str,
    last_affect: str,
    last_control_type: Optional[str] = None,
    awaiting_mcq_answer: bool = False,
    last_mcq_answered: bool = False,
    last_quiz_correct: Optional[bool] = None,
    previous_state: Optional[PedagogicalTutorState] = None,
    last_pedagogical_action: Optional[PedagogicalTutorAction] = None,
) -> PedagogicalTutorState:
    """Construct a :class:`PedagogicalTutorState` from core identifiers and
    lightweight context.

    When ``previous_state`` is provided, history-related fields such as
    ``step_count``, ``ped_action_counts`` and ``recent_ped_actions`` are
    carried forward; otherwise they are initialised to empty values.
    """

    step_count = 0
    ped_action_counts: Dict[str, int] = {}
    recent_ped_actions: List[str] = []

    if previous_state is not None:
        step_count = previous_state.step_count
        ped_action_counts = dict(previous_state.ped_action_counts or {})
        recent_ped_actions = list(previous_state.recent_ped_actions or [])

    return PedagogicalTutorState(
        episode_id=episode_id,
        session_id=session_id,
        user_id=user_id,
        concept_id=concept_id,
        plan_id=plan_id,
        plan_index=plan_index,
        plan_length=plan_length,
        phase=phase,
        mastery=mastery,
        target_mastery=target_mastery,
        step_count=step_count,
        last_intent=last_intent,
        last_affect=last_affect,
        last_control_type=last_control_type,
        last_pedagogical_action=last_pedagogical_action,
        awaiting_mcq_answer=awaiting_mcq_answer,
        last_mcq_answered=last_mcq_answered,
        last_quiz_correct=last_quiz_correct,
        ped_action_counts=ped_action_counts,
        recent_ped_actions=recent_ped_actions,
    )


def build_pedagogical_tutor_observation(
    *, state: PedagogicalTutorState
) -> PedagogicalTutorObservation:
    """Construct a :class:`PedagogicalTutorObservation` from state.

    This helper computes simple derived features such as mastery gap and
    summary counts over recent pedagogical actions.
    """

    at_end = state.plan_length > 0 and state.plan_index >= state.plan_length

    mastery_gap: Optional[float]
    if state.mastery is not None and state.target_mastery is not None:
        mastery_gap = float(state.target_mastery) - float(state.mastery)
    else:
        mastery_gap = None

    num_explain_steps = int(state.ped_action_counts.get(PedagogicalTutorAction.EXPLAIN.value, 0))
    num_practice_steps = int(
        state.ped_action_counts.get(PedagogicalTutorAction.GUIDED_PRACTICE.value, 0)
        + state.ped_action_counts.get(PedagogicalTutorAction.WORKED_EXAMPLE.value, 0)
    )
    num_quiz_steps = int(state.ped_action_counts.get(PedagogicalTutorAction.QUIZ_MCQ.value, 0))

    return PedagogicalTutorObservation(
        plan_index=state.plan_index,
        plan_length=state.plan_length,
        at_end_of_plan=at_end,
        phase=state.phase,
        mastery=state.mastery,
        target_mastery=state.target_mastery,
        mastery_gap=mastery_gap,
        last_intent=state.last_intent,
        last_affect=state.last_affect,
        last_control_type=state.last_control_type,
        awaiting_mcq_answer=state.awaiting_mcq_answer,
        last_mcq_answered=state.last_mcq_answered,
        last_quiz_correct=state.last_quiz_correct,
        num_explain_steps=num_explain_steps,
        num_practice_steps=num_practice_steps,
        num_quiz_steps=num_quiz_steps,
    )


def apply_pedagogical_transition(
    *,
    prev_state: PedagogicalTutorState,
    mdp_action: PedagogicalTutorAction,
    mastery_delta: Optional[float] = None,
    quiz_delta: Optional[float] = None,
    last_quiz_correct: Optional[bool] = None,
    awaiting_mcq_answer: Optional[bool] = None,
    last_mcq_answered: Optional[bool] = None,
    last_control_type: Optional[str] = None,
) -> StepOutcome[PedagogicalTutorState, PedagogicalTutorObservation, PedagogicalTutorAction]:
    """Apply a single pedagogical tutor MDP transition.

    This owns local counters and lightweight history; structural decisions
    and termination are handled by higher-level MDPs.
    """

    state = PedagogicalTutorState(**prev_state.__dict__)
    state.step_count = (state.step_count or 0) + 1
    state.last_pedagogical_action = mdp_action

    if last_control_type is not None:
        state.last_control_type = last_control_type

    if awaiting_mcq_answer is not None:
        state.awaiting_mcq_answer = awaiting_mcq_answer
    if last_mcq_answered is not None:
        state.last_mcq_answered = last_mcq_answered
    if last_quiz_correct is not None:
        state.last_quiz_correct = last_quiz_correct

    key = mdp_action.value
    state.ped_action_counts[key] = int(state.ped_action_counts.get(key, 0)) + 1
    state.recent_ped_actions.append(key)
    if len(state.recent_ped_actions) > 5:
        state.recent_ped_actions = state.recent_ped_actions[-5:]

    # Simple plan cursor: advance one step when possible.
    if state.plan_length > 0 and state.plan_index < state.plan_length:
        state.plan_index += 1

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

    obs = build_pedagogical_tutor_observation(state=state)

    return StepOutcome(
        state=state,
        observation=obs,
        action=mdp_action,
        reward=reward,
        terminated=False,
        termination_reason=None,
        info={},
    )
