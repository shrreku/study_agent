from __future__ import annotations

import logging
from typing import Any, Dict, List

from prompts import get as prompt_get, render as prompt_render
from llm import call_llm_json

from .auto_classifiers import ActionSuggestion, PhaseSuggestion, TurnSignals
from .context_model import TutorContext


logger = logging.getLogger(__name__)


def _default_action_payload() -> Dict[str, Any]:
    return {
        "current_action_interpretation": None,
        "next_action": "explain",
        "pedagogy_focus": ["definition", "explanation"],
        "should_use_planning": False,
        "should_use_multi_step": False,
        "desired_state_after": None,
    }


def _parse_action(raw: Dict[str, Any]) -> ActionSuggestion:
    payload = _default_action_payload()
    if isinstance(raw, dict):
        payload.update({k: raw.get(k) for k in payload.keys() if k in raw})

    next_action = str(payload.get("next_action") or "explain").strip().lower()
    if next_action not in {"explain", "ask", "reflect", "hint", "review", "orient", "preview"}:
        next_action = "explain"

    def _bool(name: str) -> bool:
        return bool(payload.get(name, False))

    pedagogy_raw = payload.get("pedagogy_focus")
    pedagogy: List[str] | None = None
    if isinstance(pedagogy_raw, list):
        cleaned: List[str] = []
        for item in pedagogy_raw:
            try:
                v = str(item or "").strip()
            except Exception:
                v = ""
            if v:
                cleaned.append(v)
        pedagogy = cleaned or None

    desired_state_after = payload.get("desired_state_after")
    if desired_state_after is not None:
        try:
            desired_state_after = str(desired_state_after).strip().lower() or None
        except Exception:
            desired_state_after = None

    current_interpretation = payload.get("current_action_interpretation")
    if current_interpretation is not None:
        try:
            current_interpretation = str(current_interpretation).strip().lower() or None
        except Exception:
            current_interpretation = None

    return ActionSuggestion(
        current_action_interpretation=current_interpretation,
        next_action=next_action,
        pedagogy_focus=pedagogy,
        should_use_planning=_bool("should_use_planning"),
        should_use_multi_step=_bool("should_use_multi_step"),
        desired_state_after=desired_state_after,
    )


def classify_action(
    context: TutorContext,
    turn_signals: TurnSignals,
    phase: PhaseSuggestion,
) -> ActionSuggestion:
    """Classify the next pedagogic action for auto/intelligent mode.

    Uses LLM JSON with conservative defaults; never raises.
    """
    try:
        template = prompt_get("tutor.action_classifier")
    except Exception:
        logger.exception("tutor_action_prompt_missing")
        return ActionSuggestion()

    observation = {
        "student_message": context.message,
        "intent": context.intent,
        "affect": context.affect,
        "focus_concept": context.focus_concept or "",
        "current_state": context.current_state.value,
        "suggested_phase": phase.suggested_state,
        "student_confirmation": turn_signals.student_confirmation,
        "meta_intent": turn_signals.meta_intent or "",
    }

    prompt = prompt_render(template, observation)
    default_payload = _default_action_payload()

    try:
        raw = call_llm_json(prompt, default=default_payload)
    except Exception:
        logger.exception("tutor_action_classifier_error")
        return ActionSuggestion()

    try:
        return _parse_action(raw or {})
    except Exception:
        logger.exception("tutor_action_parse_error")
        return ActionSuggestion()
