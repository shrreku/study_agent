from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os

from .constants import LEVEL_BUCKETS
from .state import TutorSessionPolicy
from .tools.prereq_checker import PrerequisiteChecker, PrerequisiteCheckResult


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


def select_focus_concept_with_prereqs(
    classification: Dict[str, Any],
    learning_path: List[str],
    mastery_map: Dict[str, Dict[str, Any]],
    fallback_concepts: List[str],
    user_id: str,
    enable_prereq_check: bool = True,
) -> Tuple[Optional[str], PrerequisiteCheckResult]:
    """Enhanced concept selection with prerequisite validation.

    Returns (focus_concept, prereq_check_result).
    """
    primary = (classification.get("concept") or "").strip()

    # Default ready result when check disabled or not applicable
    default_result = PrerequisiteCheckResult(
        ready=True,
        confidence=1.0,
        missing_prereqs=[],
        weak_prereqs=[],
        recommendation="",
        should_review=False,
    )

    if enable_prereq_check and primary and primary in learning_path:
        # Read thresholds from env, with safe fallbacks
        try:
            mastery_th = float(os.getenv("TUTOR_PREREQ_MASTERY_THRESHOLD", "0.6") or 0.6)
        except Exception:
            mastery_th = 0.6
        try:
            weak_th = float(os.getenv("TUTOR_PREREQ_WEAK_THRESHOLD", "0.4") or 0.4)
        except Exception:
            weak_th = 0.4

        prereq_checker = PrerequisiteChecker(mastery_threshold=mastery_th, weak_threshold=weak_th)
        prereq_result = prereq_checker.check_readiness(
            concept=primary,
            user_id=user_id,
            learning_path=learning_path,
            mastery_map=mastery_map,
        )

        if not prereq_result.ready:
            alternative = prereq_checker.get_next_ready_concept(learning_path, mastery_map, user_id)
            if alternative:
                return alternative, prereq_result
        # Fall through to standard selection with prereq_result attached
    else:
        prereq_result = default_result

    # Fallback to existing logic
    for concept in learning_path:
        info = mastery_map.get(concept)
        mastery_val = info.get("mastery") if info else None
        if mastery_val is None or mastery_val < 0.8:
            return concept, prereq_result

    return primary or (learning_path[0] if learning_path else None), prereq_result
