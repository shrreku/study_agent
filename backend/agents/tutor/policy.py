from __future__ import annotations

from typing import Any, Dict, List, Optional

from .constants import LEVEL_BUCKETS
from .state import TutorSessionPolicy


def level_for_mastery(mastery: Optional[Any]) -> str:
    try:
        score = float(mastery)
    except Exception:
        score = None
    if score is None:
        return "beginner"
    for name, lower, upper in LEVEL_BUCKETS:
        if lower <= score < upper:
            return name
    return "beginner"


def needs_cold_start(
    concept: Optional[str],
    mastery_map: Dict[str, Dict[str, Any]],
    policy: TutorSessionPolicy,
) -> bool:
    if not concept:
        return False
    if concept in policy.cold_start_completed:
        return False
    info = mastery_map.get(concept)
    if not info:
        return True
    attempts = info.get("attempts") or 0
    mastery = info.get("mastery") or 0.0
    return attempts < 1 or mastery < 0.15


def select_focus_concept(
    classification: Dict[str, Any],
    learning_path: List[str],
    mastery_map: Dict[str, Dict[str, Any]],
    fallback_concepts: List[str],
) -> Optional[str]:
    primary = (classification.get("concept") or "").strip()
    if primary:
        info = mastery_map.get(primary)
        if not info or (info.get("mastery") or 0.0) < 0.85:
            return primary
    for concept in learning_path:
        info = mastery_map.get(concept)
        mastery_val = info.get("mastery") if info else None
        if mastery_val is None or mastery_val < 0.8:
            return concept
    for concept in fallback_concepts:
        if concept:
            return concept
    return primary or (learning_path[0] if learning_path else None)


def role_sequence_for_level(level: str) -> List[str]:
    if level in {"beginner", "developing"}:
        return ["definition", "explanation", "example"]
    if level == "proficient":
        return ["example", "application", "derivation"]
    return ["derivation", "proof", "application"]
