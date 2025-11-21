from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from .mdp.pedagogical_tutor import (
    PedagogicalTutorState,
    PedagogicalTutorObservation,
    PedagogicalTutorAction,
)


@dataclass
class PedagogicalTutorStepEvent:
    """RL-friendly record for a single pedagogical tutor MDP step."""

    episode_id: str
    session_id: str
    user_id: str
    concept_id: str

    turn_index: int
    plan_index: int
    plan_length: int
    phase: str

    observation: Dict[str, Any] = field(default_factory=dict)
    action: str = ""
    reward: Dict[str, float] = field(default_factory=dict)

    control_type: Optional[str] = None
    mcq_outcome: Optional[Dict[str, Any]] = None

    mastery_before: Optional[float] = None
    mastery_after: Optional[float] = None

    source: str = "pedagogical_mdp"


def build_pedagogical_step_event(
    *,
    state: PedagogicalTutorState,
    observation: PedagogicalTutorObservation,
    action: PedagogicalTutorAction,
    reward: Dict[str, float],
    turn_index: int,
    mcq_outcome: Optional[Dict[str, Any]],
    mastery_before: Optional[float],
    mastery_after: Optional[float],
) -> PedagogicalTutorStepEvent:
    """Construct a PedagogicalTutorStepEvent from MDP artefacts.

    This helper is intentionally pure and does not perform any logging by
    itself. Callers are expected to serialize the event and emit it via the
    standard logging infrastructure.
    """

    obs_dict: Dict[str, Any] = {
        "plan_index": observation.plan_index,
        "plan_length": observation.plan_length,
        "at_end_of_plan": observation.at_end_of_plan,
        "phase": observation.phase,
        "mastery": observation.mastery,
        "target_mastery": observation.target_mastery,
        "mastery_gap": observation.mastery_gap,
        "last_intent": observation.last_intent,
        "last_affect": observation.last_affect,
        "last_control_type": observation.last_control_type,
        "awaiting_mcq_answer": observation.awaiting_mcq_answer,
        "last_mcq_answered": observation.last_mcq_answered,
        "last_quiz_correct": observation.last_quiz_correct,
        "num_explain_steps": observation.num_explain_steps,
        "num_practice_steps": observation.num_practice_steps,
        "num_quiz_steps": observation.num_quiz_steps,
    }

    return PedagogicalTutorStepEvent(
        episode_id=state.episode_id,
        session_id=state.session_id,
        user_id=state.user_id,
        concept_id=state.concept_id,
        turn_index=turn_index,
        plan_index=state.plan_index,
        plan_length=state.plan_length,
        phase=state.phase,
        observation=obs_dict,
        action=action.value,
        reward=dict(reward or {}),
        control_type=state.last_control_type,
        mcq_outcome=mcq_outcome,
        mastery_before=mastery_before,
        mastery_after=mastery_after,
    )


def pedagogical_step_event_to_dict(event: PedagogicalTutorStepEvent) -> Dict[str, Any]:
    """Convert a PedagogicalTutorStepEvent to a JSON-serializable dict."""

    return asdict(event)
