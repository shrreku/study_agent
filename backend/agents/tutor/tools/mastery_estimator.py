from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .mastery_updater import MasteryUpdater


class HeuristicMasteryEstimator:
    def __init__(
        self,
        learning_rate: float = 0.1,
        decay_factor: float = 0.95,
        min_update: float = 0.02,
        max_update: float = 0.3,
    ) -> None:
        self._updater = MasteryUpdater(
            learning_rate=learning_rate,
            decay_factor=decay_factor,
            min_update=min_update,
            max_update=max_update,
        )

    def __call__(
        self,
        *,
        concept_id: str,
        mastery_before: Optional[float],
        recent_interactions: List[Dict[str, Any]],
    ) -> Tuple[Optional[float], Optional[float]]:
        if not recent_interactions:
            return mastery_before, None

        signals = recent_interactions[-1] or {}

        try:
            current_mastery = float(mastery_before) if mastery_before is not None else 0.0
        except Exception:
            current_mastery = 0.0

        update = self._updater.compute_mastery_delta(
            concept=concept_id,
            user_id="unknown",
            interaction_signals=signals,
            current_mastery=current_mastery,
        )

        delta = update.delta
        if delta == 0.0 and mastery_before is None:
            return None, None

        post = current_mastery + delta
        if post < 0.0:
            post = 0.0
        if post > 1.0:
            post = 1.0

        if mastery_before is None:
            return post, delta if delta != 0.0 else None

        return post, delta if delta != 0.0 else None
