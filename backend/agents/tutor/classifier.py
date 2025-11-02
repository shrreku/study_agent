from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from prompts import get as prompt_get, render as prompt_render
from llm import call_json_chat

from .constants import ALLOWED_INTENTS, ALLOWED_AFFECTS
from .utils import format_concept_list, clamp_confidence

logger = logging.getLogger(__name__)


def classify_message(
    message: str,
    target_concepts: List[str],
    last_concept: Optional[str],
) -> Dict[str, Any]:
    template = prompt_get("tutor.classify")
    prompt = prompt_render(
        template,
        {
            "student_message": message,
            "target_concepts": format_concept_list(target_concepts),
            "last_concept": last_concept or "",
        },
    )
    default_concept = last_concept or (target_concepts[0] if target_concepts else "")
    default_payload = {
        "intent": "unknown",
        "affect": "neutral",
        "concept": default_concept,
        "confidence": 0.3,
        "needs_escalation": False,
    }
    try:
        result = call_json_chat(prompt, default=default_payload)
    except Exception:
        logger.exception("tutor_classify_call_failed")
        result = default_payload

    intent = str(result.get("intent") or "unknown").lower()
    if intent not in ALLOWED_INTENTS:
        intent = "unknown"
    affect = str(result.get("affect") or "neutral").lower()
    if affect not in ALLOWED_AFFECTS:
        affect = "neutral"
    concept = str(result.get("concept") or "").strip()
    if not concept:
        concept = default_concept or ""
    confidence = clamp_confidence(result.get("confidence"))
    needs_escalation = bool(result.get("needs_escalation", False))

    return {
        "intent": intent,
        "affect": affect,
        "concept": concept,
        "confidence": confidence,
        "needs_escalation": needs_escalation,
    }
