from __future__ import annotations

from typing import Optional

from ..state import TutorSessionPolicy
from ..mdp.plans import SessionPlan
from ..mdp.adapters import session_plan_from_policy, session_plan_to_policy


def load_session_plan(policy: TutorSessionPolicy) -> Optional[SessionPlan]:
    return session_plan_from_policy(policy)


def save_session_plan(plan: SessionPlan, policy: TutorSessionPolicy) -> None:
    session_plan_to_policy(plan, policy)
