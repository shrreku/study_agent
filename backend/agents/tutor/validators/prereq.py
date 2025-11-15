from __future__ import annotations

from typing import Any, Dict, List

from config.tutor_rl import ValidatorConfig
from .types import ValidatorComponentResult, ValidatorContext


def _concept_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def prereq_gate(context: ValidatorContext, config: ValidatorConfig) -> ValidatorComponentResult:
    observation = context.observation
    response_text = context.response_text.lower()

    tutor_block: Dict[str, Any] = observation.get("tutor") or {}
    focus_concept = (tutor_block.get("focus_concept") or tutor_block.get("inference_concept") or "").strip()
    learning_path = _concept_list(tutor_block.get("learning_path"))

    if focus_concept:
        focus_lower = focus_concept.lower()
    else:
        focus_lower = ""

    score = 1.0
    flags: List[str] = []
    violations: List[str] = []

    if focus_lower and focus_lower not in response_text:
        score -= 0.4
        violations.append("focus_concept_missing")

    if learning_path and focus_concept in learning_path:
        focus_index = learning_path.index(focus_concept)
        advanced_terms = learning_path[focus_index + 1 :]
    else:
        advanced_terms = []

    drifting_terms = [term for term in advanced_terms if term and term.lower() in response_text]
    if drifting_terms:
        score -= min(config.advanced_term_penalty, 0.6)
        violations.append(f"advanced_terms:{','.join(drifting_terms)}")
        flags.append("advanced_concept_drift")

    score = max(score, 0.0)

    if score < 0.5:
        flags.append("prereq_gating_failed")

    details = {
        "focus_concept": focus_concept,
        "learning_path": learning_path,
        "advanced_terms_detected": drifting_terms,
        "violations": violations,
    }

    return ValidatorComponentResult(name="gating", score=round(score, 4), details=details, flags=flags)


__all__ = ["prereq_gate"]

