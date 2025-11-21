from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .context_model import TutorContext
    from .auto_classifiers import (
        TurnSignals,
        PhaseSuggestion,
        ActionSuggestion,
        RetrievalSuggestion,
    )


class TurnSignalsClassifier(Protocol):
    def classify(self, context: "TutorContext") -> "TurnSignals":
        ...


class PhaseClassifier(Protocol):
    def classify(
        self,
        context: "TutorContext",
        signals: "TurnSignals",
    ) -> "PhaseSuggestion":
        ...


class ActionClassifier(Protocol):
    def classify(
        self,
        context: "TutorContext",
        signals: "TurnSignals",
        phase: "PhaseSuggestion",
    ) -> "ActionSuggestion":
        ...


class RetrievalClassifier(Protocol):
    def classify(
        self,
        context: "TutorContext",
        action: "ActionSuggestion",
        signals: "TurnSignals",
    ) -> "RetrievalSuggestion":
        ...
