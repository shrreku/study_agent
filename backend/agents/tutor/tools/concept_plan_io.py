from __future__ import annotations

from typing import Optional

from ..state import TutorSessionPolicy
from ..mdp.plans import ConceptPlan, ConceptPlanStep
from ..mdp.adapters import concept_plan_from_policy, concept_plan_to_policy
from ..mdp.concept import ConceptState


def load_concept_plan(policy: TutorSessionPolicy) -> Optional[ConceptPlan]:
    return concept_plan_from_policy(policy)


def save_concept_plan(plan: ConceptPlan, policy: TutorSessionPolicy) -> None:
    concept_plan_to_policy(plan, policy)


def get_current_step(plan: ConceptPlan, state: ConceptState) -> Optional[ConceptPlanStep]:
    try:
        idx = int(getattr(state, "plan_index", 0) or 0)
    except Exception:
        idx = 0
    if idx < 0:
        idx = 0
    if idx >= len(plan.steps):
        return None
    return plan.steps[idx]
