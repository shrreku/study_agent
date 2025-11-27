from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .base import MDPState, MDPObservation, StepOutcome
from .plans import ConceptPlan
from .actions import ConceptMDPAction
from ..state import TutorSessionPolicy


@dataclass
class TutorStepState(MDPState):
    """Internal tutor-level MDP state for a single step-by-step turn.

    This sits on top of the concept-level MDP and focuses on SRL plan cursor
    and recent feedback signals from the user.
    """

    episode_id: str
    session_id: str
    user_id: str
    concept_id: str

    plan_id: str
    plan_index: int
    plan_length: int

    last_concept_action: Optional[str]

    last_intent: str
    last_affect: str
    last_control_type: Optional[str]
    last_step_type: Optional[str]
    last_step_id: Optional[str]

    step_count: int
    replan_count: int


@dataclass
class TutorStepObservation(MDPObservation):
    """Filtered observation for the tutor-level MDP.

    For now this is intentionally minimal and focuses on whether there is a
    usable plan step and the last macro decision taken at the concept layer.
    """

    plan_index: int
    plan_length: int
    at_end_of_plan: bool

    last_concept_action: Optional[str]

    last_intent: str
    last_affect: str
    last_control_type: Optional[str]
    last_step_type: Optional[str]


class TutorStepMDPAction(str, Enum):
    """Canonical action space for the tutor-level MDP."""

    NEXT_STEP = "NEXT_STEP"
    REPLAN_CONCEPT = "REPLAN_CONCEPT"


def build_tutor_step_state(
    *,
    policy_state: TutorSessionPolicy,
    concept_plan: ConceptPlan,
    concept_action: ConceptMDPAction,
    session_id: str,
    user_id: str,
    concept_id: str,
    last_intent: str,
    last_affect: str,
    last_control_type: Optional[str],
) -> TutorStepState:
    """Best-effort TutorStepState builder from policy + concept plan.

    This does not persist any additional state; it derives identifiers and
    plan cursor position from the existing TutorSessionPolicy and plan.
    """

    try:
        raw_episode_id = getattr(policy_state, "concept_episode_id", None)
    except Exception:
        raw_episode_id = None
    episode_id = raw_episode_id or f"ce-{session_id}-{concept_id}"

    try:
        raw_index = getattr(policy_state, "srl_plan_step_index", 0)
    except Exception:
        raw_index = 0
    try:
        plan_index = int(raw_index or 0)
    except Exception:
        plan_index = 0

    steps = list(concept_plan.steps or [])
    plan_length = len(steps)

    # Clamp index defensively into [0, plan_length].
    if plan_index < 0:
        plan_index = 0
    if plan_index > plan_length:
        plan_index = plan_length

    last_step_id: Optional[str] = None
    last_step_type: Optional[str] = None
    if 0 <= plan_index - 1 < plan_length:
        try:
            prev_step = steps[plan_index - 1]
            last_step_id = prev_step.step_id
            last_step_type = prev_step.step_type
        except Exception:
            last_step_id = None
            last_step_type = None

    return TutorStepState(
        episode_id=episode_id,
        session_id=session_id,
        user_id=user_id,
        concept_id=concept_id,
        plan_id=concept_plan.plan_id,
        plan_index=plan_index,
        plan_length=plan_length,
        last_concept_action=concept_action.value if concept_action is not None else None,
        last_intent=last_intent,
        last_affect=last_affect,
        last_control_type=last_control_type,
        last_step_type=last_step_type,
        last_step_id=last_step_id,
        step_count=0,
        replan_count=0,
    )


def build_tutor_step_observation(*, state: TutorStepState) -> TutorStepObservation:
    """Construct a TutorStepObservation from TutorStepState."""

    at_end = state.plan_length > 0 and state.plan_index >= state.plan_length

    return TutorStepObservation(
        plan_index=state.plan_index,
        plan_length=state.plan_length,
        at_end_of_plan=at_end,
        last_concept_action=state.last_concept_action,
        last_intent=state.last_intent,
        last_affect=state.last_affect,
        last_control_type=state.last_control_type,
        last_step_type=state.last_step_type,
    )


def apply_tutor_step_transition(
    *,
    prev_state: TutorStepState,
    mdp_action: TutorStepMDPAction,
) -> StepOutcome[TutorStepState, TutorStepObservation, TutorStepMDPAction]:
    """Apply a single tutor-level MDP transition.

    This owns only lightweight counters and the local plan cursor; it does
    not modify the underlying ConceptPlan or TutorSessionPolicy.
    """

    state = TutorStepState(**prev_state.__dict__)
    state.step_count = (state.step_count or 0) + 1

    if mdp_action == TutorStepMDPAction.REPLAN_CONCEPT:
        state.replan_count = (state.replan_count or 0) + 1

    obs = build_tutor_step_observation(state=state)

    # No explicit reward shaping yet; this can be extended later.
    reward = {}

    return StepOutcome(
        state=state,
        observation=obs,
        action=mdp_action,
        reward=reward,
        terminated=False,
        termination_reason=None,
        info={},
    )
