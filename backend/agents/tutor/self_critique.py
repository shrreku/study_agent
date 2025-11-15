from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import logging

from llm import call_llm_json
from prompts import get as prompt_get, render as prompt_render

logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    overall_quality: float
    issues_found: List[str]
    suggestions: List[str]
    should_revise: bool
    reasoning: str


class SelfCritic:
    def critique_response(
        self,
        plan,
        response: str,
        observation: Dict,
    ) -> CritiqueResult:
        template = prompt_get("tutor.self_critique")
        prompt = prompt_render(
            template,
            {
                "plan_thinking": getattr(plan, "thinking", ""),
                "intended_action": getattr(plan, "intended_action", "explain"),
                "plan_rationale": getattr(plan, "action_rationale", ""),
                "generated_response": response,
                "student_level": (observation.get("tutor", {}) or {}).get("concept_level", "beginner"),
            },
        )
        default = {
            "quality": 0.7,
            "issues": [],
            "suggestions": [],
            "should_revise": False,
            "reasoning": "Response seems adequate.",
        }
        try:
            result = call_llm_json(prompt, default=default)
            return CritiqueResult(
                overall_quality=float(result.get("quality", 0.7)),
                issues_found=list(result.get("issues", [])),
                suggestions=list(result.get("suggestions", [])),
                should_revise=bool(result.get("should_revise", False)),
                reasoning=str(result.get("reasoning", "")),
            )
        except Exception:
            logger.exception("self_critique_failed")
            return CritiqueResult(
                overall_quality=0.6,
                issues_found=[],
                suggestions=[],
                should_revise=False,
                reasoning="Critique unavailable",
            )
