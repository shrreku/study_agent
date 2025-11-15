from __future__ import annotations

from typing import Dict, List

from config.tutor_rl import ValidatorConfig
from .types import ValidatorComponentResult, ValidatorContext


_INTENT_TO_ACTION_PRIORITIES: Dict[str, List[str]] = {
    "question": ["explain", "hint", "worked_example"],
    "answer": ["reflect", "ask", "review"],
    "reflection": ["reflect", "ask", "review"],
    "off_topic": ["review", "ask", "explain"],
    "greeting": ["ask", "explain"],
    "unknown": ["explain", "ask", "review"],
}


def intent_alignment(context: ValidatorContext, _: ValidatorConfig) -> ValidatorComponentResult:
    observation = context.observation
    classifier_block = observation.get("classifier") or {}
    action_block = observation.get("action") or {}

    intent = (classifier_block.get("intent") or "unknown").lower()
    action_type = (action_block.get("type") or "").lower()
    affect = (classifier_block.get("affect") or "neutral").lower()

    allowed_actions = _INTENT_TO_ACTION_PRIORITIES.get(intent) or _INTENT_TO_ACTION_PRIORITIES["unknown"]
    if action_type in allowed_actions[:1]:
        score = 1.0
        band = "preferred"
    elif action_type in allowed_actions[1:2]:
        score = 0.8
        band = "acceptable"
    elif action_type in allowed_actions:
        score = 0.6
        band = "fallback"
    else:
        score = 0.2
        band = "mismatch"

    if affect in {"frustrated", "unsure"} and action_type == "explain":
        # Encourage explanations when student is confused.
        score = max(score, 0.7)
        band = "affect_override"

    flags = []
    if score < 0.6:
        flags.append("intent_action_mismatch")

    details = {
        "intent": intent,
        "affect": affect,
        "action_type": action_type,
        "band": band,
        "allowed_actions": allowed_actions,
    }

    return ValidatorComponentResult(name="intent", score=round(score, 4), details=details, flags=flags)


__all__ = ["intent_alignment"]

