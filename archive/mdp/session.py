from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import os

from .base import MDPState, MDPObservation, StepOutcome
from ..state import TutorSessionPolicy
from ..context_model import TutorContext


@dataclass
class SessionState(MDPState):
    """Internal session-level MDP state for step-by-step mode.

    This captures the session plan, current position in the plan, and
    aggregated mastery/progress statistics. It is derived from
    TutorSessionPolicy + TutorContext + mastery_map.
    """

    episode_id: str
    session_id: str
    user_id: str

    strategy: str
    concept_plan: List[str]
    plan_index: int
    completed_concepts: int
    total_concepts: int

    avg_mastery: Optional[float]
    min_mastery: Optional[float]
    max_mastery: Optional[float]
    remaining_low_mastery_count: int

    current_concept_id: Optional[str]
    current_concept_mastery: Optional[float]
    current_concept_target_mastery: Optional[float]
    current_concept_episode_id: Optional[str]
    current_concept_terminated: bool
    current_concept_termination_reason: Optional[str]

    session_step_count: int
    concept_episodes_completed: int
    time_budget_steps: Optional[int]
    time_used_steps: int

    last_session_action: Optional["SessionMDPAction"]


@dataclass
class SessionObservation(MDPObservation):
    """Filtered observation for the session-level MDP.

    This is what a RL policy or analysis code would typically consume.
    """

    strategy: str
    current_index: int
    total_concepts: int
    position_fraction: float

    current_concept_id: Optional[str]
    current_mastery: Optional[float]
    target_mastery: Optional[float]

    avg_mastery: Optional[float]
    min_mastery: Optional[float]
    remaining_low_mastery_count: int

    session_step_count: int
    concept_episodes_completed: int
    time_used_steps: int

    last_session_action: Optional[str]
    last_concept_termination_reason: Optional[str]


class SessionMDPAction(str, Enum):
    """Canonical action space for the session-level MDP.

    These are macro decisions about which concept to study next and
    when to end or replan the session.
    """

    FOLLOW_PLAN_CONCEPT = "FOLLOW_PLAN_CONCEPT"
    ADVANCE_IN_PLAN = "ADVANCE_IN_PLAN"
    REPLAN_SESSION = "REPLAN_SESSION"
    TERMINATE_SESSION = "TERMINATE_SESSION"


def _safe_mastery_value(raw: Any) -> Optional[float]:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except Exception:
        return None


def build_session_state_from_session(
    *,
    policy_state: TutorSessionPolicy,
    tutor_context: TutorContext,
    mastery_map: Dict[str, Dict[str, Any]],
    episode_id_fallback: Optional[str] = None,
) -> SessionState:
    """Best-effort SessionState builder from session policy + context.

    This mirrors the core data carried in TutorSessionPolicy plus the
    mastery map and session plan. It is safe to call even when
    session_plan is missing; in that case the concept_plan will be
    empty and derived statistics will be minimal.
    """

    episode_id = episode_id_fallback or f"se-{tutor_context.session_id}"

    session_plan: Dict[str, Any] = {}
    try:
        raw_plan = getattr(policy_state, "session_plan", None)
        if isinstance(raw_plan, dict):
            session_plan = raw_plan
    except Exception:
        session_plan = {}

    raw_concept_plan = session_plan.get("concept_plan") if isinstance(session_plan, dict) else None
    concept_plan: List[str] = []
    if isinstance(raw_concept_plan, list):
        concept_plan = [str(c) for c in raw_concept_plan if isinstance(c, str) and c]

    try:
        plan_index = int(getattr(policy_state, "session_plan_index", 0) or 0)
    except Exception:
        plan_index = 0

    total_concepts = len(concept_plan)
    if plan_index < 0:
        plan_index = 0
    if plan_index > total_concepts:
        plan_index = total_concepts

    completed_concepts = max(0, min(plan_index, total_concepts))

    strategy = "learning_path"
    try:
        raw_strategy = getattr(policy_state, "session_strategy", None) or session_plan.get("strategy")
        if isinstance(raw_strategy, str) and raw_strategy.strip():
            strategy = raw_strategy.strip().lower()
    except Exception:
        pass

    # Aggregate mastery statistics across the plan.
    mastery_values: List[float] = []
    for cid in concept_plan:
        mv = _safe_mastery_value((mastery_map.get(cid) or {}).get("mastery"))
        if mv is not None:
            mastery_values.append(mv)

    if mastery_values:
        avg_mastery = float(sum(mastery_values) / len(mastery_values))
        min_mastery = float(min(mastery_values))
        max_mastery = float(max(mastery_values))
    else:
        avg_mastery = None
        min_mastery = None
        max_mastery = None

    # Count remaining low-mastery concepts using the same target as
    # step-by-step mode where possible.
    try:
        target_th = float(os.getenv("TUTOR_STEP_SRL_TARGET_MASTERY", "0.8") or 0.8)
    except Exception:
        target_th = 0.8

    remaining_low_mastery_count = 0
    for cid in concept_plan:
        mv = _safe_mastery_value((mastery_map.get(cid) or {}).get("mastery"))
        if mv is None or mv < target_th:
            remaining_low_mastery_count += 1

    # Current concept slice.
    current_concept_id: Optional[str] = None
    if 0 <= plan_index < total_concepts:
        current_concept_id = concept_plan[plan_index]

    current_concept_mastery: Optional[float] = None
    if current_concept_id:
        current_concept_mastery = _safe_mastery_value(
            (mastery_map.get(current_concept_id) or {}).get("mastery")
        )

    try:
        current_concept_target_mastery = float(
            os.getenv("TUTOR_STEP_SRL_TARGET_MASTERY", "0.8") or 0.8
        )
    except Exception:
        current_concept_target_mastery = None

    current_concept_episode_id = getattr(policy_state, "concept_episode_id", None)

    # Session progress proxies: for now we approximate using the turn
    # index as a step counter.
    try:
        session_step_count = int(getattr(tutor_context, "turn_index", 0) or 0)
    except Exception:
        session_step_count = 0

    concept_episodes_completed = 0
    time_budget_steps: Optional[int] = None
    try:
        raw_budget = os.getenv("TUTOR_SESSION_TIME_BUDGET_STEPS")
        if raw_budget not in (None, ""):
            time_budget_steps = int(raw_budget)
    except Exception:
        time_budget_steps = None

    time_used_steps = session_step_count

    return SessionState(
        episode_id=episode_id,
        session_id=tutor_context.session_id,
        user_id=tutor_context.user_id,
        strategy=strategy,
        concept_plan=concept_plan,
        plan_index=plan_index,
        completed_concepts=completed_concepts,
        total_concepts=total_concepts,
        avg_mastery=avg_mastery,
        min_mastery=min_mastery,
        max_mastery=max_mastery,
        remaining_low_mastery_count=remaining_low_mastery_count,
        current_concept_id=current_concept_id,
        current_concept_mastery=current_concept_mastery,
        current_concept_target_mastery=current_concept_target_mastery,
        current_concept_episode_id=current_concept_episode_id,
        current_concept_terminated=False,
        current_concept_termination_reason=None,
        session_step_count=session_step_count,
        concept_episodes_completed=concept_episodes_completed,
        time_budget_steps=time_budget_steps,
        time_used_steps=time_used_steps,
        last_session_action=None,
    )


def build_session_observation(state: SessionState) -> SessionObservation:
    """Construct a SessionObservation from a SessionState."""

    total = state.total_concepts or 0
    idx = state.plan_index or 0
    if total > 0 and 0 <= idx <= total:
        position_fraction = float(idx) / float(total)
    else:
        position_fraction = 0.0

    last_action_label: Optional[str] = None
    if state.last_session_action is not None:
        try:
            last_action_label = str(state.last_session_action.value)
        except Exception:
            last_action_label = str(state.last_session_action)

    return SessionObservation(
        strategy=state.strategy,
        current_index=idx,
        total_concepts=total,
        position_fraction=position_fraction,
        current_concept_id=state.current_concept_id,
        current_mastery=state.current_concept_mastery,
        target_mastery=state.current_concept_target_mastery,
        avg_mastery=state.avg_mastery,
        min_mastery=state.min_mastery,
        remaining_low_mastery_count=state.remaining_low_mastery_count,
        session_step_count=state.session_step_count,
        concept_episodes_completed=state.concept_episodes_completed,
        time_used_steps=state.time_used_steps,
        last_session_action=last_action_label,
        last_concept_termination_reason=state.current_concept_termination_reason,
    )


def apply_session_transition(
    *,
    prev_state: SessionState,
    mdp_action: SessionMDPAction,
    concept_termination_reason: Optional[str] = None,
) -> StepOutcome[SessionState, SessionObservation, SessionMDPAction]:
    """Apply a single session-level MDP transition.

    For now this uses a simple heuristic policy:

    - FOLLOW_PLAN_CONCEPT: keep current plan_index.
    - ADVANCE_IN_PLAN: increment plan_index if possible; terminate when
      the end of the plan is reached.
    - REPLAN_SESSION: caller is responsible for rebuilding the
      SessionState; this function only records the action.
    - TERMINATE_SESSION: mark the session episode as terminated.

    Reward is left empty for now; FLOW-07 will add reward computation
    and logging.
    """

    state = SessionState(**prev_state.__dict__)
    state.last_session_action = mdp_action

    if concept_termination_reason is not None:
        state.current_concept_terminated = True
        state.current_concept_termination_reason = concept_termination_reason

    terminated = False
    termination_reason: Optional[str] = None

    if mdp_action == SessionMDPAction.ADVANCE_IN_PLAN:
        if state.plan_index < state.total_concepts:
            state.plan_index += 1
        if state.plan_index >= state.total_concepts:
            terminated = True
            termination_reason = "completed_plan"
    elif mdp_action == SessionMDPAction.TERMINATE_SESSION:
        terminated = True
        termination_reason = "session_end"
    elif mdp_action == SessionMDPAction.REPLAN_SESSION:
        # Caller is expected to rebuild SessionState based on the new
        # plan on the next turn. We do not change plan_index here.
        pass
    else:
        # FOLLOW_PLAN_CONCEPT: no index change.
        pass

    obs = build_session_observation(state)

    return StepOutcome(
        state=state,
        observation=obs,
        action=mdp_action,
        reward={},
        terminated=terminated,
        termination_reason=termination_reason,
        info={},
    )
