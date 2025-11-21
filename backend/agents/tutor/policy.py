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

        # CHECK 1: Is the primary concept ALREADY mastered?
        # If so, and the user is just saying "continue" (not asking a specific question),
        # we should probably move to the NEXT concept.
        # We infer "just saying continue" if the intent is not a specific question/confusion.
        # For now, we'll be conservative: if mastery > 0.85, we consider looking ahead.
        primary_info = mastery_map.get(primary)
        primary_mastery = (primary_info or {}).get("mastery", 0.0)
        
        # If primary is mastered, we might want to skip it
        if primary_mastery > 0.85:
             # Find its index
             try:
                 idx = learning_path.index(primary)
                 # Look for next unmastered
                 found_next = None
                 for candidate in learning_path[idx+1:]:
                     c_info = mastery_map.get(candidate)
                     c_mastery = (c_info or {}).get("mastery", 0.0)
                     if c_mastery < 0.8:
                         found_next = candidate
                         break
                 
                 if found_next:
                     # We found a better target forward in the path
                     # But we must ensure we don't skip if the user EXPLICITLY asked about the old one.
                     # This function doesn't see the message text directly, but the caller
                     # passed 'classification'.
                     # If classification confidence is high on 'primary', it implies the user
                     # likely mentioned it. If confidence is low or it was just context carryover,
                     # we can switch.
                     # Simplified heuristic: If we found a next concept, let's check ITs prerequisites.
                     # If it's ready, we suggest it.
                     primary = found_next
             except ValueError:
                 pass

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


def build_session_plan(
    learning_targets: List[str],
    mastery_map: Dict[str, Dict[str, Any]],
    learning_path: List[str],
    strategy: str,
) -> Dict[str, Any]:
    """Build a simple session-level concept plan.

    This helper is used in step-by-step mode to derive an ordered list of
    concepts for the current session from the target concepts, mastery
    map, and prerequisite chain.

    Strategies:
        - "learning_path" (default): follow the prerequisite chain order.
        - "weakest_first": sort by ascending mastery.
        - "custom": preserve the target_concepts order.
    """

    normalized_strategy = (strategy or "learning_path").strip().lower()
    if normalized_strategy not in {"learning_path", "weakest_first", "custom"}:
        normalized_strategy = "learning_path"

    # Deduplicate while preserving order.
    def _dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in seq:
            if not item:
                continue
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    if normalized_strategy == "learning_path":
        base = [c for c in learning_path if isinstance(c, str) and c]
        concept_plan = _dedupe(base)
    elif normalized_strategy == "weakest_first":
        candidates = [c for c in learning_targets if isinstance(c, str) and c]
        if not candidates:
            candidates = [c for c in mastery_map.keys() if isinstance(c, str) and c]

        def _mastery_score(cid: str) -> float:
            info = mastery_map.get(cid) or {}
            try:
                return float(info.get("mastery") or 0.0)
            except Exception:
                return 0.0

        concept_plan = sorted(_dedupe(candidates), key=_mastery_score)
    else:  # "custom": preserve provided targets
        base = [c for c in learning_targets if isinstance(c, str) and c]
        concept_plan = _dedupe(base)

    return {
        "strategy": normalized_strategy,
        "concept_plan": concept_plan,
    }
