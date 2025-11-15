from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class MasteryUpdate:
    """Mastery update event."""
    concept: str
    delta: float  # Change in mastery (-1.0 to +1.0)
    reason: str  # Why update occurred
    confidence: float
    timestamp: datetime


class MasteryUpdater:
    """Update student mastery based on interaction signals."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        decay_factor: float = 0.95,
        min_update: float = 0.05,
        max_update: float = 0.3,
    ) -> None:
        """
        Args:
            learning_rate: How quickly mastery changes
            decay_factor: Decay for repeated successful demonstrations
            min_update: Minimum change threshold
            max_update: Maximum single update
        """
        self.learning_rate = float(learning_rate)
        self.decay_factor = float(decay_factor)
        self.min_update = float(min_update)
        self.max_update = float(max_update)

    def compute_mastery_delta(
        self,
        concept: str,
        user_id: str,
        interaction_signals: Dict[str, Any],
        current_mastery: float,
    ) -> MasteryUpdate:
        """
        Compute mastery change based on interaction.

        Signals considered:
        - Student affect (engaged=positive, confused=negative)
        - Response correctness (if answering question)
        - Explanation quality (if explaining back to tutor)
        - Prerequisite mastery
        """
        delta = 0.0
        reasons = []

        # Signal 1: Affect
        affect = (interaction_signals.get("affect") or "neutral").strip()
        if affect == "engaged":
            delta += 0.1
            reasons.append("engaged_affect")
        elif affect in {"confused", "frustrated"}:
            delta -= 0.05
            reasons.append(f"{affect}_affect")

        # Signal 2: Intent + correctness
        intent = (interaction_signals.get("intent") or "unknown").strip()
        if intent == "answer":
            answer_correct = interaction_signals.get("answer_correct")
            if answer_correct is True:
                delta += 0.15
                reasons.append("correct_answer")
            elif answer_correct is False:
                delta -= 0.1
                reasons.append("incorrect_answer")

        # Signal 3: Student explanation quality
        if intent == "explanation":
            try:
                quality = float(interaction_signals.get("explanation_quality", 0) or 0)
            except Exception:
                quality = 0.0
            if quality > 0.7:
                delta += 0.2
                reasons.append("quality_explanation")

        # Signal 4: Decay for high mastery (harder to improve at top)
        try:
            cm = float(current_mastery or 0.0)
        except Exception:
            cm = 0.0
        if cm > 0.7:
            delta *= self.decay_factor

        # Apply learning rate and clamp
        delta *= self.learning_rate
        if delta > self.max_update:
            delta = self.max_update
        if delta < -self.max_update:
            delta = -self.max_update

        # Ignore tiny updates
        if abs(delta) < self.min_update:
            delta = 0.0

        confidence = self._compute_confidence(interaction_signals)

        return MasteryUpdate(
            concept=concept,
            delta=delta,
            reason=", ".join(reasons) if reasons else "no_signal",
            confidence=confidence,
            timestamp=datetime.utcnow(),
        )

    def apply_update(
        self,
        user_id: str,
        update: MasteryUpdate,
        db_cursor,
    ) -> float:
        """
        Apply mastery update to database. Returns new mastery score.
        """
        if update.delta == 0.0:
            return self._get_current_mastery(user_id, update.concept, db_cursor)

        current = self._get_current_mastery(user_id, update.concept, db_cursor)
        # For new concepts, seed with positive delta (if any)
        initial_mastery = update.delta if update.delta > 0 else 0.0
        is_correct = 1 if ("correct_answer" in (update.reason or "")) else 0

        # Persist using the same pattern as quiz updates
        db_cursor.execute(
            """
            INSERT INTO user_concept_mastery (user_id, concept, mastery, last_seen, attempts, correct)
            VALUES (%s::uuid, %s, %s, now(), %s, %s)
            ON CONFLICT (user_id, concept) DO UPDATE
              SET attempts = user_concept_mastery.attempts + 1,
                  correct = user_concept_mastery.correct + EXCLUDED.correct,
                  last_seen = now(),
                  mastery = LEAST(1.0, GREATEST(0.0, user_concept_mastery.mastery + %s))
            RETURNING user_concept_mastery.mastery
            """,
            (user_id, update.concept, initial_mastery, 1, is_correct, update.delta),
        )
        row = db_cursor.fetchone()
        new_mastery = float(row[0]) if row and row[0] is not None else current

        try:
            logger.info(
                "tutor_mastery_updated",
                extra={
                    "user_id": user_id,
                    "concept": update.concept,
                    "old_mastery": current,
                    "new_mastery": new_mastery,
                    "delta": update.delta,
                    "reason": update.reason,
                    "confidence": update.confidence,
                },
            )
        except Exception:
            # Avoid breaking flow on logging issues
            pass

        return new_mastery

    def _get_current_mastery(self, user_id: str, concept: str, db_cursor) -> float:
        db_cursor.execute(
            "SELECT mastery FROM user_concept_mastery WHERE user_id = %s::uuid AND concept = %s",
            (user_id, concept),
        )
        row = db_cursor.fetchone()
        try:
            return float(row[0]) if row and row[0] is not None else 0.0
        except Exception:
            return 0.0

    def _compute_confidence(self, signals: Dict[str, Any]) -> float:
        """Compute confidence in mastery update based on signal strength."""
        confidence = 0.5  # Base

        if signals.get("answer_correct") is not None:
            confidence += 0.3  # Explicit assessment is high confidence

        affect = signals.get("affect")
        if affect in {"engaged", "confused", "frustrated"}:
            confidence += 0.1  # Clear affect signal

        try:
            if float(signals.get("classification_confidence", 0) or 0) > 0.8:
                confidence += 0.1  # High classification confidence
        except Exception:
            pass

        if confidence > 1.0:
            confidence = 1.0
        return confidence
