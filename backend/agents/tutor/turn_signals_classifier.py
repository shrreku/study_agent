from __future__ import annotations

import logging
from typing import Any, Dict

from prompts import get as prompt_get, render as prompt_render
from llm import call_llm_json

from .auto_classifiers import TurnSignals
from .context_model import TutorContext


logger = logging.getLogger(__name__)


def _default_turn_signals_payload() -> Dict[str, Any]:
    return {
        "student_confirmation": "not_confirmed",
        "wants_orientation": False,
        "wants_study_plan": False,
        "wants_closure": False,
        "reflection_provided": False,
        "meta_intent": "",
    }


def _parse_turn_signals(raw: Dict[str, Any]) -> TurnSignals:
    payload = _default_turn_signals_payload()
    if isinstance(raw, dict):
        payload.update({k: raw.get(k) for k in payload.keys() if k in raw})

    student_confirmation = str(payload.get("student_confirmation") or "not_confirmed").strip().lower()
    if student_confirmation not in {"confirmed", "uncertain", "not_confirmed"}:
        student_confirmation = "not_confirmed"

    def _bool(name: str) -> bool:
        return bool(payload.get(name, False))

    meta = payload.get("meta_intent")
    meta_intent = str(meta).strip().lower() if meta is not None else None
    if meta_intent == "":
        meta_intent = None

    return TurnSignals(
        student_confirmation=student_confirmation,
        wants_orientation=_bool("wants_orientation"),
        wants_study_plan=_bool("wants_study_plan"),
        wants_closure=_bool("wants_closure"),
        reflection_provided=_bool("reflection_provided"),
        meta_intent=meta_intent,
    )


def classify_turn_signals(context: TutorContext) -> TurnSignals:
    """Classify turn-level signals for auto/intelligent mode using LLM JSON.

    Falls back to a conservative default on any failure.
    """
    try:
        template = prompt_get("tutor.turn_signals")
    except Exception:
        logger.exception("tutor_turn_signals_prompt_missing")
        return TurnSignals()

    recent_turns = []
    try:
        # Recent turns list is usually newest-first; keep a small window.
        for t in (context.recent_turns or [])[:4]:
            role = t.get("role") or "user"
            text = (t.get("text") or t.get("response_text") or "").strip()
            if text:
                recent_turns.append({"role": role, "text": text})
    except Exception:
        recent_turns = []

    observation = {
        "student_message": context.message,
        "intent": context.intent,
        "affect": context.affect,
        "focus_concept": context.focus_concept or "",
        "current_state": context.current_state.value,
        "last_action": "",  # can be threaded later from session policy
        "recent_turns": recent_turns,
    }

    prompt = prompt_render(template, observation)
    default_payload = _default_turn_signals_payload()

    try:
        raw = call_llm_json(prompt, default=default_payload)
    except Exception:
        logger.exception("tutor_turn_signals_classifier_error")
        return TurnSignals()

    try:
        return _parse_turn_signals(raw or {})
    except Exception:
        logger.exception("tutor_turn_signals_parse_error")
        return TurnSignals()
