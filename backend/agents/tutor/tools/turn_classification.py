from __future__ import annotations

from typing import Any, Dict

from ..classifier import classify_message
from ..runtime.utils import normalize_single_concept_label
from ..runtime.context import ClassificationContext, TurnContext
from .turn_controls import ParsedTurnControls


class TurnClassifier:
    def classify(
        self,
        ctx: TurnContext,
        session_state: Dict[str, Any],
        controls: ParsedTurnControls,
    ) -> ClassificationContext:
        if controls.is_control_turn:
            last_concept = session_state.get("last_concept")
            classification_raw: Dict[str, Any] = {
                "intent": "control",
                "affect": "neutral",
                "concept": last_concept,
                "confidence": None,
            }
        else:
            targets = ctx.target_concepts or session_state.get("target_concepts", [])
            classification_raw = classify_message(
                controls.message_for_classification,
                targets,
                session_state.get("last_concept"),
            )

        try:
            raw_cls_concept = classification_raw.get("concept")
        except Exception:
            raw_cls_concept = None
        norm_cls_concept = normalize_single_concept_label(raw_cls_concept)
        if norm_cls_concept:
            classification_raw["concept"] = norm_cls_concept

        classification = ClassificationContext(
            intent=classification_raw.get("intent", "unknown"),
            affect=classification_raw.get("affect", "neutral"),
            concept=classification_raw.get("concept"),
            confidence=classification_raw.get("confidence"),
        )

        if controls.mcq_answer is not None:
            classification.intent = "answer"

        return classification
