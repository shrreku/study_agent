from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from .base import BaseEnvironment, EnvironmentTransition
from ..mdp.plans import ConceptPlan, ConceptPlanStep


@dataclass
class ConceptState:
    """Environment-facing concept state.

    This represents a single concept episode: a concept plan plus a cursor
    over its steps, along with a minimal mastery signal.
    """

    session_id: str
    user_id: str
    concept_id: str

    concept_plan: Optional[ConceptPlan] = None
    current_step_index: int = 0

    current_mastery: float = 0.0
    target_mastery: float = 0.8

    phase: str = "learning"
    steps_completed: int = 0
    plan_version: int = 0

    terminated: bool = False
    termination_reason: Optional[str] = None

    def current_step(self) -> Optional[ConceptPlanStep]:
        if not self.concept_plan:
            return None
        if self.current_step_index < 0:
            return None
        steps = self.concept_plan.steps or []
        if self.current_step_index >= len(steps):
            return None
        return steps[self.current_step_index]

    def is_mastery_reached(self) -> bool:
        try:
            return self.current_mastery >= self.target_mastery
        except Exception:
            return False


class ConceptAction(str, Enum):
    """High-level concept environment actions.

    These map to the structural decisions of the concept MDP: when to
    generate a plan, which step to expose to the tutor, and when to mark
    the concept complete.
    """

    GENERATE_PLAN = "GENERATE_PLAN"
    EXECUTE_STEP = "EXECUTE_STEP"
    ADVANCE_STEP = "ADVANCE_STEP"
    COMPLETE_CONCEPT = "COMPLETE_CONCEPT"


class ConceptEnvironment(BaseEnvironment[ConceptState]):
    """Deterministic concept environment.

    - Planning is delegated to an external tool (LLM planner) when
      GENERATE_PLAN is invoked.
    - EXECUTE_STEP exposes the current plan step to the tutor layer.
    - ADVANCE_STEP moves the plan cursor forward.
    - COMPLETE_CONCEPT marks the episode as terminated.
    """

    def __init__(
        self,
        *,
        session_id: str,
        user_id: str,
        concept_id: str,
        concept_plan: Optional[ConceptPlan] = None,
        current_mastery: float = 0.0,
        target_mastery: float = 0.8,
    ) -> None:
        self.state = ConceptState(
            session_id=session_id,
            user_id=user_id,
            concept_id=concept_id,
            concept_plan=concept_plan,
            current_mastery=current_mastery,
            target_mastery=target_mastery,
        )

    def reset(
        self,
        concept_plan: Optional[ConceptPlan] = None,
        current_mastery: float = 0.0,
        target_mastery: float = 0.8,
    ) -> ConceptState:
        self.state.concept_plan = concept_plan
        self.state.current_step_index = 0
        self.state.current_mastery = current_mastery
        self.state.target_mastery = target_mastery
        self.state.phase = "learning"
        self.state.steps_completed = 0
        self.state.plan_version += 1
        self.state.terminated = False
        self.state.termination_reason = None
        return self.state

    def step(self, action: ConceptAction, **kwargs: object) -> EnvironmentTransition[ConceptState]:
        state = self.state

        if state.terminated:
            return EnvironmentTransition(state=state, outputs={}, terminated=True, termination_reason=state.termination_reason)

        outputs: Dict[str, object] = {}

        if action == ConceptAction.GENERATE_PLAN:
            # Planner tool is expected to be passed via kwargs for MVP.
            planner = kwargs.get("planner")  # type: ignore[assignment]
            if planner is None:
                raise ValueError("planner is required for GENERATE_PLAN")

            plan = planner(  # type: ignore[call-arg]
                user_id=state.user_id,
                session_id=state.session_id,
                concept_id=state.concept_id,
                target_mastery=state.target_mastery,
                context_obs=kwargs.get("context_obs", {}),
            )
            self.reset(concept_plan=plan, current_mastery=state.current_mastery, target_mastery=state.target_mastery)
            outputs["concept_plan"] = plan
            outputs["current_step"] = self.state.current_step()

        elif action == ConceptAction.EXECUTE_STEP:
            # Expose the current step to the tutor layer; tutor will decide
            # which pedagogical action to take and handle user interaction.
            step = state.current_step()
            outputs["current_step"] = step

        elif action == ConceptAction.ADVANCE_STEP:
            if state.concept_plan and state.current_step_index < len(state.concept_plan.steps):
                state.current_step_index += 1
                state.steps_completed += 1

            step = state.current_step()
            outputs["current_step"] = step

            # Terminate when we have exhausted the plan. For the MVP we do
            # not gate this on mastery; higher-level policies can decide how
            # to schedule review concepts.
            if step is None:
                state.terminated = True
                state.termination_reason = "plan_exhausted"

        elif action == ConceptAction.COMPLETE_CONCEPT:
            state.terminated = True
            state.termination_reason = "explicit_completion"

        else:
            outputs["current_step"] = state.current_step()

        outputs["concept_complete"] = bool(state.terminated)

        return EnvironmentTransition(
            state=state,
            outputs=outputs,
            terminated=state.terminated,
            termination_reason=state.termination_reason,
        )

    def get_state(self) -> ConceptState:
        return self.state

    def is_terminated(self) -> bool:
        return bool(self.state.terminated)
