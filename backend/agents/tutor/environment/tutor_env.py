from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from .base import BaseEnvironment, EnvironmentTransition
from ..mdp.pedagogical_tutor import PedagogicalTutorAction
from ..mdp.plans import ConceptPlan, ConceptPlanStep
from ..mdp.tools import PedagogicalResponseGeneratorTool


@dataclass
class TutorState:
    """Environment-facing tutor state.

    Tracks the current pedagogical step, last action, and whether the
    tutor is awaiting a button signal from the learner.
    """

    session_id: str
    user_id: str
    concept_id: str

    current_step: Optional[ConceptPlanStep] = None
    last_action: Optional[PedagogicalTutorAction] = None

    awaiting_user_input: bool = False
    turn_in_step: int = 0

    last_control_signal: Optional[str] = None
    plan_index: int = 0
    plan_length: int = 0
    phase: str = "learning"


@dataclass
class _PedagogicalShim:
    """Minimal state object for the pedagogical response tool.

    The default response tool only needs ``phase``, ``plan_index`` and
    ``plan_length`` from the pedagogical tutor state, so we provide a
    lightweight shim instead of the full MDP state.
    """

    phase: str
    plan_index: int
    plan_length: int


class TutorAction(str, Enum):
    """Alias of pedagogical tutor actions for the environment layer."""

    EXPLAIN = PedagogicalTutorAction.EXPLAIN.value
    ASK_QUESTION = PedagogicalTutorAction.ASK_QUESTION.value
    WORKED_EXAMPLE = PedagogicalTutorAction.WORKED_EXAMPLE.value
    GUIDED_PRACTICE = PedagogicalTutorAction.GUIDED_PRACTICE.value
    DEFINE_TERM = PedagogicalTutorAction.DEFINE_TERM.value
    REFLECTION_PROMPT = PedagogicalTutorAction.REFLECTION_PROMPT.value
    SUMMARY = PedagogicalTutorAction.SUMMARY.value
    QUIZ_MCQ = PedagogicalTutorAction.QUIZ_MCQ.value
    WAIT_FOR_CONFIRMATION = PedagogicalTutorAction.WAIT_FOR_CONFIRMATION.value

    def to_pedagogical_action(self) -> PedagogicalTutorAction:
        return PedagogicalTutorAction(self.value)


class TutorEnvironment(BaseEnvironment[TutorState]):
    """Deterministic tutor environment.

    Given a current plan step and a tutor action (chosen by a policy),
    produces tutor-facing outputs (messages, buttons) via the
    PedagogicalResponseGeneratorTool.
    """

    def __init__(
        self,
        *,
        session_id: str,
        user_id: str,
        concept_id: str,
        response_tool: PedagogicalResponseGeneratorTool,
        current_step: Optional[ConceptPlanStep] = None,
        concept_plan: Optional[ConceptPlan] = None,
        plan_index: int = 0,
        phase: str = "learning",
    ) -> None:
        self.state = TutorState(
            session_id=session_id,
            user_id=user_id,
            concept_id=concept_id,
            current_step=current_step,
        )
        self.state.plan_index = int(plan_index or 0)
        self.state.plan_length = (
            len(concept_plan.steps) if concept_plan and concept_plan.steps else 0
        )
        self.state.phase = phase
        self._concept_plan = concept_plan
        self._response_tool = response_tool

    def reset(
        self,
        current_step: Optional[ConceptPlanStep] = None,
        concept_plan: Optional[ConceptPlan] = None,
        plan_index: int = 0,
        phase: str = "learning",
    ) -> TutorState:
        self.state.current_step = current_step
        self.state.last_action = None
        self.state.awaiting_user_input = False
        self.state.turn_in_step = 0
        self.state.last_control_signal = None
        self.state.plan_index = int(plan_index or 0)
        self.state.plan_length = (
            len(concept_plan.steps) if concept_plan and concept_plan.steps else 0
        )
        self.state.phase = phase
        self._concept_plan = concept_plan
        return self.state

    def _build_pedagogical_state(self) -> _PedagogicalShim:
        """Construct the lightweight state view for the response tool."""

        return _PedagogicalShim(
            phase=self.state.phase,
            plan_index=self.state.plan_index,
            plan_length=self.state.plan_length,
        )

    def step(self, action: TutorAction, **kwargs: object) -> EnvironmentTransition[TutorState]:
        state = self.state

        # Buttons are interpreted as control signals at the environment
        # level, not as stochastic transitions.
        control_signal = kwargs.get("control_signal")  # type: ignore[assignment]
        if isinstance(control_signal, str):
            state.last_control_signal = control_signal

        state.last_action = action.to_pedagogical_action()
        state.turn_in_step += 1

        # For MVP we always await a button after emitting a pedagogical
        # response, except for WAIT_FOR_CONFIRMATION which explicitly
        # indicates that we are already waiting.
        if state.last_action == PedagogicalTutorAction.WAIT_FOR_CONFIRMATION:
            state.awaiting_user_input = True
        else:
            state.awaiting_user_input = True

        # Delegate message generation to the response tool.
        ped_state = self._build_pedagogical_state()
        context_obs = kwargs.get("context_obs")  # type: ignore[assignment]
        if not isinstance(context_obs, dict):
            context_obs = {}
        response = self._response_tool(
            user_id=state.user_id,
            session_id=state.session_id,
            concept_id=state.concept_id,
            pedagogical_action=state.last_action,
            ped_state=ped_state,
            concept_plan=self._concept_plan,
            context_obs=context_obs,
        )

        outputs: Dict[str, object] = dict(response or {})

        return EnvironmentTransition(
            state=state,
            outputs=outputs,
            terminated=False,
            termination_reason=None,
        )

    def get_state(self) -> TutorState:
        return self.state

    def is_terminated(self) -> bool:
        # Tutor layer is logically non-terminating; higher layers decide
        # when the episode ends.
        return False
