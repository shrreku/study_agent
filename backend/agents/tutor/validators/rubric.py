from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from config.tutor_rl import ValidatorConfig
from .types import ValidatorComponentResult, ValidatorContext


RUBRIC_FEATURE_NAMES = (
    "direct_answer",
    "example",
    "reasoning",
    "formative",
)


def _count_feature(score: float, weight: float = 1.0) -> float:
    return score * weight


def _has_any_marker(text: str, markers: Tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _has_direct_answer(text: str, config: ValidatorConfig, focus_concept: str | None) -> bool:
    lowered = text.lower()
    if focus_concept and focus_concept.lower() in lowered:
        return True
    return _has_any_marker(lowered, tuple(config.direct_answer_markers))


def _has_formative(text: str, config: ValidatorConfig) -> bool:
    if text.strip().endswith("?"):
        return True
    return _has_any_marker(text, tuple(config.suggestion_markers))


def rubric_check(context: ValidatorContext, config: ValidatorConfig) -> ValidatorComponentResult:
    observation = context.observation
    response_text = context.response_text
    tutor_block: Dict[str, Any] = observation.get("tutor") or {}
    focus_concept = tutor_block.get("focus_concept") or tutor_block.get("inference_concept")

    lowered = response_text.lower()

    example_present = _has_any_marker(lowered, tuple(config.example_markers)) or bool(
        re.search(r"\bexample\b", lowered)
    )
    reasoning_present = _has_any_marker(lowered, tuple(config.reasoning_markers))
    formative_present = _has_formative(lowered, config)
    direct_answer_present = _has_direct_answer(lowered, config, focus_concept)

    feature_scores = {
        "direct_answer": 1.0 if direct_answer_present else 0.0,
        "example": 1.0 if example_present else 0.0,
        "reasoning": 1.0 if reasoning_present else 0.0,
        "formative": 1.0 if formative_present else 0.0,
    }

    # Weighted average; formative has slightly lower weight to tolerate some turns.
    weighted_total = sum(
        _count_feature(value, 1.0 if name != "formative" else 0.75)
        for name, value in feature_scores.items()
    )
    weight_sum = 3.75
    score = weighted_total / weight_sum

    details: Dict[str, Any] = {
        "focus_concept": focus_concept,
        "features": feature_scores,
    }

    flags = []
    if score < 0.5:
        flags.append("rubric_incomplete")

    return ValidatorComponentResult(name="rubric", score=round(score, 4), details=details, flags=flags)


__all__ = ["rubric_check"]

