from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

from ingestion import embed as embed_service


class RelationshipValidator:
    """Validate relationship quality using semantic embeddings and heuristics."""

    def __init__(self, min_similarity: float = None):
        try:
            env_min = float(__import__("os").getenv("KG_MIN_SEMANTIC_SIMILARITY", "0.30"))
        except Exception:
            env_min = 0.30
        self.min_similarity = float(min_similarity) if min_similarity is not None else env_min

    def _similarity(self, a: str, b: str) -> float:
        try:
            va = np.array(embed_service.embed_text(a) or [], dtype=float)
            vb = np.array(embed_service.embed_text(b) or [], dtype=float)
            if va.size == 0 or vb.size == 0:
                return 0.0
            denom = (np.linalg.norm(va) * np.linalg.norm(vb))
            if denom == 0.0:
                return 0.0
            return float(np.dot(va, vb) / denom)
        except Exception:
            return 0.0

    def _estimate_complexity(self, text: str) -> int:
        score = 0
        words = (text or "").split()
        score += min(len(words), 5)
        technical_indicators = [
            "law", "theorem", "equation", "theory",
            "coefficient", "parameter", "variable",
            "differential", "integral", "partial",
        ]
        tl = (text or "").lower()
        score += sum(1 for term in technical_indicators if term in tl)
        if any(c in text for c in ["∂", "∫", "∑", "α", "β", "γ"]):
            score += 2
        return min(score, 10)

    def validate_prerequisite(
        self,
        source: str,
        target: str,
        confidence: float,
    ) -> Tuple[bool, float, str]:
        sim = self._similarity(source, target)
        if sim > 0.9:
            return False, 0.0, "concepts_too_similar"
        if sim < self.min_similarity:
            return False, 0.0, "concepts_unrelated"
        # Complexity heuristic: prereq should be simpler
        sc = self._estimate_complexity(source)
        tc = self._estimate_complexity(target)
        if sc > tc + 2:
            return True, float(confidence) * 0.5, "complexity_mismatch"
        return True, float(confidence), "valid"

    def validate_applies_to(
        self,
        source: str,
        target: str,
        confidence: float,
    ) -> Tuple[bool, float, str]:
        sim = self._similarity(source, target)
        if sim < (self.min_similarity * 0.83):
            return False, 0.0, "unrelated_domains"
        return True, float(confidence), "valid"
