from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import TutorSessionPolicy


@dataclass
class ConceptStepEvent:
    episode_id: str
    session_id: str
    user_id: str
    concept_id: str
    step_index: int
    turn_start_index: int
    turn_end_index: int
    control_type: Optional[str] = None
    control_payload: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    plan_step: Optional[Dict[str, Any]] = None
    pre_mastery: Optional[float] = None
    post_mastery: Optional[float] = None
    quiz_result: Dict[str, Any] = field(default_factory=dict)
    reward: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None


@dataclass
class ConceptEpisode:
    episode_id: str
    session_id: str
    user_id: str
    concept_id: str
    target_mastery: Optional[float] = None
    mastery_start: Optional[float] = None
    mastery_end: Optional[float] = None
    start_turn_index: Optional[int] = None
    end_turn_index: Optional[int] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    termination_reason: Optional[str] = None
    steps: List[ConceptStepEvent] = field(default_factory=list)
    aggregates: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, event: ConceptStepEvent) -> None:
        self.steps.append(event)


@dataclass
class SessionStepEvent:
    episode_id: str
    session_id: str
    user_id: str
    turn_index: int
    plan_index: int
    total_concepts: int
    strategy: Optional[str] = None
    current_concept_id: Optional[str] = None
    action: Optional[str] = None
    terminated: bool = False
    termination_reason: Optional[str] = None
    reward: Dict[str, Any] = field(default_factory=dict)


def start_concept_episode(
    policy_state: "TutorSessionPolicy",
    *,
    session_id: str,
    user_id: str,
    concept_id: str,
    turn_index: int,
    target_mastery: Optional[float],
    mastery_map: Dict[str, Any],
) -> None:
    """Initialize concept-episode tracking on policy_state if needed.

    This is idempotent for the same concept: if an episode is already
    active for concept_id, this is a no-op.
    """

    if not concept_id:
        return

    current_episode_id = getattr(policy_state, "concept_episode_id", None)
    current_concept = getattr(policy_state, "concept_episode_concept", None)
    if current_episode_id and current_concept == concept_id:
        return

    try:
        episode_id = f"ce-{session_id}-{concept_id}-{turn_index}"
    except Exception:
        episode_id = f"ce-{session_id}-{concept_id}"

    # Best-effort mastery snapshot at episode start
    mastery_start: Optional[float] = None
    try:
        raw = (mastery_map.get(concept_id) or {}).get("mastery", 0.0)
        mastery_start = float(raw or 0.0)
    except Exception:
        mastery_start = None

    try:
        policy_state.concept_episode_id = episode_id
        policy_state.concept_episode_concept = concept_id
        policy_state.concept_episode_start_turn = int(turn_index)
        policy_state.concept_episode_mastery_start = mastery_start
        policy_state.concept_episode_step_count = 0
        policy_state.concept_episode_quiz_correct = 0
        policy_state.concept_episode_quiz_wrong = 0
        policy_state.concept_episode_last_control_type = None
    except Exception:
        # Defensive: do not fail the main tutor flow if tracking cannot be set.
        return


def update_concept_episode_counters(
    policy_state: "TutorSessionPolicy",
    *,
    control_type: Optional[str],
    mcq_outcome: Optional[Dict[str, Any]],
) -> None:
    """Update per-episode counters for each concept-level step.

    This is a no-op if no concept episode is currently active.
    """

    if not getattr(policy_state, "concept_episode_id", None):
        return

    try:
        policy_state.concept_episode_step_count += 1
    except Exception:
        try:
            policy_state.concept_episode_step_count = 1
        except Exception:
            pass

    if control_type:
        try:
            policy_state.concept_episode_last_control_type = control_type
        except Exception:
            pass

    if not mcq_outcome:
        return

    answer_correct = mcq_outcome.get("answer_correct")
    if answer_correct is True:
        try:
            policy_state.concept_episode_quiz_correct += 1
        except Exception:
            try:
                policy_state.concept_episode_quiz_correct = 1
            except Exception:
                pass
    elif answer_correct is False:
        try:
            policy_state.concept_episode_quiz_wrong += 1
        except Exception:
            try:
                policy_state.concept_episode_quiz_wrong = 1
            except Exception:
                pass


def finalize_concept_episode(
    policy_state: "TutorSessionPolicy",
    *,
    session_id: str,
    user_id: str,
    concept_id: str,
    turn_index: int,
    mastery_map: Dict[str, Any],
    target_mastery: Optional[float],
    termination_reason: str,
    pre_mastery: Optional[float],
    post_mastery: Optional[float],
) -> Optional[ConceptEpisode]:
    """Build a ConceptEpisode object and clear episode fields.

    Returns None if no episode is active for the given concept.
    """

    if not getattr(policy_state, "concept_episode_id", None):
        return None

    episode_concept = getattr(policy_state, "concept_episode_concept", None) or concept_id
    if not episode_concept or episode_concept != concept_id:
        return None

    episode_id = getattr(policy_state, "concept_episode_id", None)
    if not episode_id:
        try:
            episode_id = f"ce-{session_id}-{episode_concept}"
        except Exception:
            episode_id = f"ce-{session_id}"

    mastery_start = getattr(policy_state, "concept_episode_mastery_start", None)

    # Best-effort mastery_end snapshot
    mastery_end: Optional[float] = None
    try:
        raw = (mastery_map.get(episode_concept) or {}).get("mastery")
        if raw not in (None, ""):
            mastery_end = float(raw)
    except Exception:
        mastery_end = None
    if mastery_end is None:
        mastery_end = post_mastery if post_mastery is not None else mastery_start

    try:
        start_turn = int(getattr(policy_state, "concept_episode_start_turn", 0) or 0)
    except Exception:
        start_turn = 0

    step_count = getattr(policy_state, "concept_episode_step_count", 0) or 0
    quiz_correct = getattr(policy_state, "concept_episode_quiz_correct", 0) or 0
    quiz_wrong = getattr(policy_state, "concept_episode_quiz_wrong", 0) or 0

    aggregates: Dict[str, Any] = {
        "step_count": step_count,
        "quiz_correct": quiz_correct,
        "quiz_wrong": quiz_wrong,
    }
    if mastery_start is not None and mastery_end is not None:
        try:
            aggregates["mastery_delta"] = float(mastery_end) - float(mastery_start)
        except Exception:
            pass

    episode = ConceptEpisode(
        episode_id=episode_id,
        session_id=session_id,
        user_id=user_id,
        concept_id=episode_concept,
        target_mastery=target_mastery,
        mastery_start=mastery_start,
        mastery_end=mastery_end,
        start_turn_index=start_turn,
        end_turn_index=turn_index,
        started_at=None,
        ended_at=None,
        termination_reason=termination_reason,
        steps=[],
        aggregates=aggregates,
    )

    # Clear episode tracking fields so a future concept can start fresh.
    try:
        policy_state.concept_episode_id = None
        policy_state.concept_episode_concept = None
        policy_state.concept_episode_start_turn = 0
        policy_state.concept_episode_mastery_start = None
        policy_state.concept_episode_step_count = 0
        policy_state.concept_episode_quiz_correct = 0
        policy_state.concept_episode_quiz_wrong = 0
        policy_state.concept_episode_last_control_type = None
    except Exception:
        pass

    return episode
