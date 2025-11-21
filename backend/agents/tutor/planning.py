from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging
import os

from llm import call_llm_json
from prompts import get as prompt_get, render as prompt_render

logger = logging.getLogger(__name__)


@dataclass
class TutorPlan:
    """Internal reasoning plan before action."""
    thinking: str
    intended_action: str
    action_rationale: str
    retrieval_query: str
    pedagogy_focus: List[str]
    difficulty_cap: str
    confidence: float
    assumptions: List[str]
    risks: List[str]
    steps: List[Dict] = field(default_factory=list)
    target_sequence: List[str] = field(default_factory=list)


class TutorPlanner:
    """Generate internal reasoning plans for tutoring actions."""

    def __init__(self, enable_srl_mode: bool = True):
        self.enable_srl_mode = enable_srl_mode

    def generate_plan(
        self,
        observation: Dict,
        student_state: Dict,
        available_actions: List[str],
    ) -> TutorPlan:
        if not self.enable_srl_mode:
            return self._rule_based_plan(observation, student_state)

        # Build planning prompt
        template = prompt_get("tutor.srl_planning")
        
        # Extract fields (handling both legacy nested and new flat formats for safety)
        student_message = observation.get("student_message") or (observation.get("user", {}) or {}).get("message", "")
        intent = observation.get("intent") or (observation.get("classifier", {}) or {}).get("intent", "unknown")
        affect = observation.get("affect") or (observation.get("classifier", {}) or {}).get("affect", "neutral")
        focus_concept = observation.get("focus_concept") or (observation.get("tutor", {}) or {}).get("focus_concept", "unknown")
        student_level = observation.get("student_level") or (observation.get("tutor", {}) or {}).get("concept_level", "beginner")
        previous_action = observation.get("previous_action") or (observation.get("policy", {}) or {}).get("last_action", "none")
        recent_history = observation.get("recent_history", "")
        
        # Prefer passed-in student_state for mastery/path, fallback to observation
        learning_path_list = (student_state or {}).get("learning_path") or observation.get("learning_path") or []
        learning_path_str = ", ".join(learning_path_list) if isinstance(learning_path_list, list) else str(learning_path_list)
        
        mastery_map = (student_state or {}).get("mastery_map") or observation.get("mastery_snapshot") or {}

        prompt = prompt_render(
            template,
            {
                "student_message": student_message,
                "intent": intent,
                "affect": affect,
                "focus_concept": focus_concept,
                "student_level": student_level,
                "mastery_snapshot": self._format_mastery(mastery_map),
                "learning_path": learning_path_str,
                "available_actions": ", ".join(available_actions or []),
                "previous_action": previous_action,
                "recent_history": recent_history,
            },
        )

        # Extract variables for logic
        # (Already extracted above)

        # Dynamic default plan based on intent
        default_action = "explain"
        default_rationale = "Default to explanation."
        
        if str(intent) == "answer":
            default_action = "reflect"
            default_rationale = "Student answered, should reflect and check understanding."
        elif str(intent) == "question":
            default_action = "explain"
            default_rationale = "Student asked a question, provide explanation."
        elif str(affect) in {"confused", "frustrated"}:
            default_action = "hint"
            default_rationale = "Student seems stuck, offer a hint."

        default_plan = {
            "thinking": f"Student intent is {intent}. I should {default_action}.",
            "intended_action": default_action,
            "action_rationale": default_rationale,
            "retrieval_query": str(focus_concept),
            "pedagogy_focus": ["definition", "explanation"] if default_action == "explain" else ["concept_check"],
            "difficulty_cap": "intermediate",
            "confidence": 0.6,
            "assumptions": ["Student has basic understanding"],
            "risks": ["May be too advanced"],
            "steps": [
                {"action": default_action, "rationale": default_rationale, "pedagogy_focus": ["definition", "explanation"]},
                {"action": "ask", "rationale": "Check fluency", "pedagogy_focus": ["concept_check"]},
            ],
            "target_sequence": [],
        }

        try:
            result = call_llm_json(prompt, default=default_plan)
            return TutorPlan(
                thinking=str(result.get("thinking", default_plan["thinking"])),
                intended_action=str(result.get("intended_action", "explain")),
                action_rationale=str(result.get("action_rationale", "")),
                retrieval_query=str(result.get("retrieval_query", "")),
                pedagogy_focus=list(result.get("pedagogy_focus", ["explanation"])),
                difficulty_cap=str(result.get("difficulty_cap", "intermediate")),
                confidence=float(result.get("confidence", 0.6)),
                assumptions=list(result.get("assumptions", [])),
                risks=list(result.get("risks", [])),
                steps=list(result.get("steps", default_plan["steps"])),
                target_sequence=list(result.get("target_sequence", [])),
            )
        except Exception:
            logger.exception("planning_failed")
            # Optional fallback to rules if enabled
            use_rules = os.getenv("TUTOR_SRL_FALLBACK_TO_RULES", "true").strip().lower() == "true"
            if use_rules:
                return self._rule_based_plan(observation, student_state)
            return TutorPlan(
                thinking="",
                intended_action="explain",
                action_rationale="",
                retrieval_query=str(focus_concept),
                pedagogy_focus=["explanation"],
                difficulty_cap="intermediate",
                confidence=0.5,
                assumptions=[],
                risks=[],
                steps=default_plan["steps"],
                target_sequence=[],
            )

    def _rule_based_plan(self, observation: Dict, student_state: Dict) -> TutorPlan:
        # Support both flat and nested
        affect = str(observation.get("affect") or (observation.get("classifier", {}) or {}).get("affect", "neutral") or "neutral").lower()
        intent = str(observation.get("intent") or (observation.get("classifier", {}) or {}).get("intent", "unknown") or "unknown").lower()
        level = str(observation.get("student_level") or (observation.get("tutor", {}) or {}).get("concept_level", "beginner") or "beginner").lower()
        focus_concept = str(observation.get("focus_concept") or (observation.get("tutor", {}) or {}).get("focus_concept", "") or "")

        if affect in {"confused", "frustrated"}:
            action = "hint"
            rationale = "Student is confused, provide gentle hint"
        elif intent == "answer":
            action = "reflect"
            rationale = "Student answered, prompt reflection"
        else:
            action = "explain"
            rationale = "Default explanation"

        if level == "beginner":
            pedagogy = ["definition", "explanation", "example"]
        elif level == "proficient":
            pedagogy = ["example", "application"]
        else:
            pedagogy = ["derivation", "application"]

        return TutorPlan(
            thinking="[Rule-based fallback]",
            intended_action=action,
            action_rationale=rationale,
            retrieval_query=focus_concept,
            pedagogy_focus=pedagogy,
            difficulty_cap="intermediate" if level == "beginner" else "advanced",
            confidence=0.7,
            assumptions=[],
            risks=[],
            steps=[
                {"action": action, "rationale": rationale, "pedagogy_focus": pedagogy},
                {"action": "ask", "rationale": "Check understanding", "pedagogy_focus": ["concept_check"]},
            ],
            target_sequence=[],
        )

    def _format_mastery(self, mastery_map: Dict) -> str:
        if not mastery_map:
            return "No mastery data"
        lines: List[str] = []
        for concept, data in list(mastery_map.items())[:5]:
            try:
                mastery = float((data or {}).get("mastery", 0.0) or 0.0)
            except Exception:
                mastery = 0.0
            lines.append(f"- {concept}: {mastery:.2f}")
        return "\n".join(lines)
