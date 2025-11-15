from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PrerequisiteCheckResult:
    ready: bool
    confidence: float
    missing_prereqs: List[str]
    weak_prereqs: List[str]
    recommendation: str
    should_review: bool


class PrerequisiteChecker:
    def __init__(self, mastery_threshold: float = 0.6, weak_threshold: float = 0.4) -> None:
        self.mastery_threshold = float(mastery_threshold)
        self.weak_threshold = float(weak_threshold)

    def check_readiness(
        self,
        concept: str,
        user_id: str,
        learning_path: List[str],
        mastery_map: Dict[str, Dict],
    ) -> PrerequisiteCheckResult:
        missing: List[str] = []
        weak: List[str] = []

        try:
            concept_idx = learning_path.index(concept)
            prereqs = learning_path[:concept_idx]
        except ValueError:
            prereqs = []

        for prereq in prereqs:
            mastery_score = float(mastery_map.get(prereq, {}).get("mastery", 0.0) or 0.0)
            if mastery_score == 0.0:
                missing.append(prereq)
            elif mastery_score < self.weak_threshold:
                weak.append(prereq)
            elif mastery_score < self.mastery_threshold:
                weak.append(prereq)

        if missing:
            ready = False
            confidence = 0.0
            recommendation = f"Review prerequisites first: {', '.join(missing[:2])}"
            should_review = True
        elif len(weak) > 2:
            ready = False
            confidence = 0.3
            recommendation = f"Strengthen understanding of: {', '.join(weak[:2])}"
            should_review = True
        elif weak:
            ready = True
            confidence = 0.7
            recommendation = f"Proceed with caution; review {weak[0]} if needed"
            should_review = False
        else:
            ready = True
            confidence = 1.0
            recommendation = "Student is ready for this concept"
            should_review = False

        return PrerequisiteCheckResult(
            ready=ready,
            confidence=confidence,
            missing_prereqs=missing,
            weak_prereqs=weak,
            recommendation=recommendation,
            should_review=should_review,
        )

    def get_next_ready_concept(
        self,
        learning_path: List[str],
        mastery_map: Dict[str, Dict],
        user_id: str,
    ) -> Optional[str]:
        for concept in learning_path:
            mastery = float(mastery_map.get(concept, {}).get("mastery", 0.0) or 0.0)
            if mastery > 0.8:
                continue
            result = self.check_readiness(concept, user_id, learning_path, mastery_map)
            if result.ready:
                return concept
        for concept in learning_path:
            if float(mastery_map.get(concept, {}).get("mastery", 0.0) or 0.0) < 0.8:
                return concept
        return None

    def suggest_review_path(
        self,
        target_concept: str,
        learning_path: List[str],
        mastery_map: Dict[str, Dict],
    ) -> List[str]:
        try:
            concept_idx = learning_path.index(target_concept)
            prereqs = learning_path[:concept_idx]
        except ValueError:
            return []
        review_path: List[str] = []
        for prereq in prereqs:
            mastery = float(mastery_map.get(prereq, {}).get("mastery", 0.0) or 0.0)
            if mastery < self.mastery_threshold:
                review_path.append(prereq)
        return review_path
