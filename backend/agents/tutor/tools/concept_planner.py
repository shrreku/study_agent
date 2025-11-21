from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm import call_llm_json
from prompts import get as prompt_get, render as prompt_render

from ..context_model import TutorContext
from ..mdp.plans import ConceptPlan


def make_concept_planning_observation(
    tutor_context: TutorContext,
    retrieval_chunks: List[Dict[str, Any]],
    mastery_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    base = tutor_context.to_planning_observation()
    base["chunks"] = list(retrieval_chunks or [])[:5]
    base["mastery_snapshot"] = mastery_map or {}
    if tutor_context.focus_concept and not base.get("focus_concept"):
        base["focus_concept"] = tutor_context.focus_concept
    if tutor_context.concept_level and not base.get("student_level"):
        base["student_level"] = tutor_context.concept_level
    return base


class ConceptPlannerLLM:
    def __init__(
        self,
        template_name: str = "tutor.srl_concept_plan_v1",
        model_hint: Optional[str] = None,
    ) -> None:
        self.template_name = template_name
        self.model_hint = model_hint

    def __call__(
        self,
        *,
        user_id: str,
        session_id: str,
        concept_id: str,
        target_mastery: Optional[float],
        context_obs: Dict[str, Any],
    ) -> ConceptPlan:
        payload: Dict[str, Any] = dict(context_obs or {})
        payload.setdefault("session_id", session_id)
        payload.setdefault("user_id", user_id)
        payload.setdefault("concept_id", concept_id)
        if target_mastery is not None:
            payload.setdefault("target_mastery", target_mastery)

        try:
            template = prompt_get(self.template_name)
            prompt = prompt_render(template, payload)
            result = call_llm_json(prompt, model_hint=self.model_hint)
        except Exception:
            result = {}

        if not isinstance(result, dict):
            result = {}

        if not result.get("concept_id"):
            result["concept_id"] = concept_id
        if target_mastery is not None and result.get("target_mastery") is None:
            result["target_mastery"] = target_mastery
        if not result.get("plan_id"):
            result["plan_id"] = f"cp-{session_id}-{concept_id}"
        if not result.get("source"):
            result["source"] = "concept_planner_v1"

        try:
            plan = ConceptPlan.from_dict(result)
        except Exception:
            plan = ConceptPlan(
                plan_id=str(result.get("plan_id") or f"cp-{session_id}-{concept_id}"),
                concept_id=concept_id,
                steps=[],
                created_at_step=0,
                source=str(result.get("source") or "concept_planner_v1"),
                initial_mastery=None,
                target_mastery=target_mastery,
            )

        if not plan.concept_id:
            plan.concept_id = concept_id
        if plan.target_mastery is None and target_mastery is not None:
            plan.target_mastery = target_mastery

        return plan
