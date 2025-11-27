from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from .base import BaseEnvironment, EnvironmentTransition
from ..mdp.plans import SessionPlan


@dataclass
class SessionState:
    """Environment-facing session state.

    This is a compact, deterministic view of the session-level state used
    by the environment orchestrator. Persistence of richer MDP state
    remains in the TutorSessionPolicy / mdp_state.
    """

    session_id: str
    user_id: str

    session_plan: Optional[SessionPlan] = None
    current_concept_index: int = 0
    concepts_completed: int = 0

    mastery_map: Dict[str, float] = field(default_factory=dict)

    turn_count: int = 0
    terminated: bool = False
    termination_reason: Optional[str] = None

    @property
    def current_concept_id(self) -> Optional[str]:
        if not self.session_plan:
            return None
        if self.current_concept_index < 0:
            return None
        entries = self.session_plan.entries or []
        if self.current_concept_index >= len(entries):
            return None
        return entries[self.current_concept_index].concept_id


class SessionAction(str, Enum):
    """High-level session environment actions.

    These are macro decisions about which concept to study and when to end
    the session. Policies operate over this space given the current
    SessionState (or a derived observation).
    """

    START_CONCEPT = "START_CONCEPT"
    ADVANCE_CONCEPT = "ADVANCE_CONCEPT"
    END_SESSION = "END_SESSION"


class SessionEnvironment(BaseEnvironment[SessionState]):
    """Deterministic session environment.

    The transition dynamics are stationary and purely a function of the
    current state and action. Buttons from the UI are interpreted by the
    orchestrator/policies and passed in as actions.
    """

    def __init__(
        self,
        *,
        session_id: str,
        user_id: str,
        session_plan: Optional[SessionPlan] = None,
        mastery_map: Optional[Dict[str, float]] = None,
    ) -> None:
        self.state = SessionState(
            session_id=session_id,
            user_id=user_id,
            session_plan=session_plan,
            mastery_map=dict(mastery_map or {}),
        )

    def reset(
        self,
        session_plan: Optional[SessionPlan] = None,
        mastery_map: Optional[Dict[str, float]] = None,
    ) -> SessionState:
        self.state.session_plan = session_plan
        self.state.current_concept_index = 0
        self.state.concepts_completed = 0
        self.state.mastery_map = dict(mastery_map or {})
        self.state.turn_count = 0
        self.state.terminated = False
        self.state.termination_reason = None
        return self.state

    def step(self, action: SessionAction, **kwargs: object) -> EnvironmentTransition[SessionState]:
        state = self.state

        if state.terminated:
            # No further transitions once terminated.
            return EnvironmentTransition(state=state, outputs={"focus_concept": None}, terminated=True, termination_reason=state.termination_reason)

        state.turn_count += 1
        outputs: Dict[str, object] = {}

        if action == SessionAction.START_CONCEPT:
            # Ensure we have a valid current concept; do not advance index here.
            current = state.current_concept_id
            outputs["focus_concept"] = current

        elif action == SessionAction.ADVANCE_CONCEPT:
            # Move to the next concept if available; otherwise terminate.
            if state.session_plan and state.current_concept_index < len(state.session_plan.entries):
                state.concepts_completed += 1
                state.current_concept_index += 1

            current = state.current_concept_id
            outputs["focus_concept"] = current

            if current is None:
                state.terminated = True
                state.termination_reason = "completed_plan"

        elif action == SessionAction.END_SESSION:
            state.terminated = True
            state.termination_reason = "session_end"
            outputs["focus_concept"] = None

        else:
            # Unknown action: no-op but keep determinism.
            outputs["focus_concept"] = state.current_concept_id

        outputs["session_complete"] = bool(state.terminated)

        return EnvironmentTransition(
            state=state,
            outputs=outputs,
            terminated=state.terminated,
            termination_reason=state.termination_reason,
        )

    def get_state(self) -> SessionState:
        return self.state

    def is_terminated(self) -> bool:
        return bool(self.state.terminated)
