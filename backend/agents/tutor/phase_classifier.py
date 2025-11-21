from __future__ import annotations

import logging
from typing import Any, Dict

from prompts import get as prompt_get, render as prompt_render
from llm import call_llm_json

from .auto_classifiers import PhaseSuggestion, TurnSignals
from .context_model import TutorContext
from .state_machine import TutorState


logger = logging.getLogger(__name__)


def _default_phase_payload() -> Dict[str, Any]:
    return {
        "suggested_state": "teaching",
        "confidence": 0.5,
        "reason": "default_from_state",
    }


def _parse_phase(raw: Dict[str, Any]) -> PhaseSuggestion:
    payload = _default_phase_payload()
    if isinstance(raw, dict):
        payload.update({k: raw.get(k) for k in payload.keys() if k in raw})

    suggested = str(payload.get("suggested_state") or "teaching").strip().lower()
    if suggested not in {s.value for s in TutorState}:
        suggested = "teaching"

    try:
        confidence = float(payload.get("confidence", 0.5) or 0.5)
    except Exception:
        confidence = 0.5

    reason = str(payload.get("reason") or "").strip()

    return PhaseSuggestion(
        suggested_state=suggested,
        confidence=confidence,
        reason=reason,
    )


def classify_phase(context: TutorContext, turn_signals: TurnSignals) -> PhaseSuggestion:
    """Classify the high-level learning phase/state using LLM JSON.

    Defaults to the current TutorState on failure.
    """
    try:
        template = prompt_get("tutor.phase_classifier")
    except Exception:
        logger.exception("tutor_phase_prompt_missing")
        # Default to current state with moderate confidence
        return PhaseSuggestion(
            suggested_state=context.current_state.value,
            confidence=0.6,
            reason="fallback_current_state",
        )

    observation = {
        "student_message": context.message,
        "intent": context.intent,
        "affect": context.affect,
        "focus_concept": context.focus_concept or "",
        "current_state": context.current_state.value,
        "student_confirmation": turn_signals.student_confirmation,
        "wants_closure": bool(getattr(turn_signals, "wants_closure", False)),
        "wants_orientation": bool(getattr(turn_signals, "wants_orientation", False)),
        "wants_study_plan": bool(getattr(turn_signals, "wants_study_plan", False)),
    }

    prompt = prompt_render(template, observation)
    default_payload = _default_phase_payload()

    try:
        raw = call_llm_json(prompt, default=default_payload)
    except Exception:
        logger.exception("tutor_phase_classifier_error")
        return PhaseSuggestion(
            suggested_state=context.current_state.value,
            confidence=0.6,
            reason="fallback_current_state",
        )

    try:
        return _parse_phase(raw or {})
    except Exception:
        logger.exception("tutor_phase_parse_error")
        return PhaseSuggestion(
            suggested_state=context.current_state.value,
            confidence=0.6,
            reason="fallback_current_state",
        )
