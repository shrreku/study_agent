from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm import call_llm_json
from prompts import get as prompt_get, render as prompt_render

from ..mdp.plans import SessionPlan, SessionPlanEntry


class SessionPlannerLLM:
    def __init__(
        self,
        template_name: str = "tutor.srl_session_plan_v1",
        model_hint: Optional[str] = None,
    ) -> None:
        self.template_name = template_name
        self.model_hint = model_hint

    def __call__(
        self,
        *,
        user_id: str,
        session_id: str,
        strategy: str,
        target_concepts: List[str],
        mastery_map: Dict[str, Dict[str, Any]],
    ) -> SessionPlan:
        try:
            template = prompt_get(self.template_name)
            payload: Dict[str, Any] = {
                "session_id": session_id,
                "user_id": user_id,
                "strategy": strategy,
                "target_concepts": list(target_concepts or []),
                "mastery_map": mastery_map or {},
            }
            prompt = prompt_render(template, payload)
            result = call_llm_json(prompt, model_hint=self.model_hint)
        except Exception:
            result = {}

        if not isinstance(result, dict):
            result = {}

        raw_strategy = result.get("strategy")
        if isinstance(raw_strategy, str) and raw_strategy.strip():
            out_strategy = raw_strategy.strip()
        else:
            out_strategy = (strategy or "learning_path").strip() or "learning_path"
        result["strategy"] = out_strategy

        if not result.get("plan_id"):
            result["plan_id"] = f"sp-{session_id}"
        if not result.get("source"):
            result["source"] = "session_planner_v1"

        try:
            plan = SessionPlan.from_dict(result)
        except Exception:
            plan = SessionPlan(strategy=out_strategy, entries=[], plan_id=str(result.get("plan_id") or ""))

        if not plan.entries:
            entries: List[SessionPlanEntry] = []
            for cid in target_concepts or []:
                if not isinstance(cid, str) or not cid.strip():
                    continue
                entries.append(SessionPlanEntry(concept_id=cid.strip()))
            plan = SessionPlan(
                strategy=out_strategy,
                entries=entries,
                plan_id=plan.plan_id or f"sp-{session_id}",
                created_at_step=plan.created_at_step,
                source=plan.source or "session_planner_v1",
            )

        return plan
