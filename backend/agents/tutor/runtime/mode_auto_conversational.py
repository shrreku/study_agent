from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..state_machine import TutorState, TutorStateManager
from ..decision_engine import TutorDecisionEngine, ActionDecision
from ..context_model import TutorContext
from ..auto_classifiers import (
    TurnSignals,
    PhaseSuggestion,
    ActionSuggestion,
    RetrievalSuggestion,
)
from ..classifiers import (
    TurnSignalsClassifier,
    PhaseClassifier,
    ActionClassifier,
    RetrievalClassifier,
)
from ..turn_signals_classifier import classify_turn_signals
from ..phase_classifier import classify_phase
from ..action_classifier import classify_action
from ..retrieval_classifier import classify_retrieval
from .utils import looks_like_study_plan_request, looks_like_confirmation


@dataclass
class AutoConversationalRuntime:
    """Auto/intelligent conversational runtime.

    This runtime is responsible for building per-turn classifier-style signals
    (currently via lightweight heuristics) and delegating to the unified
    TutorDecisionEngine for the final ActionDecision.

    In the V3 architecture this class is an internal implementation detail of
    the StepEngine auto path. Callers should route decisions through
    ``StepEngine.decide_step(control_mode="auto", ...)`` instead of invoking
    this runtime directly from the orchestrator or API entrypoints.
    """

    mode: str
    decision_engine: TutorDecisionEngine
    turn_signals_classifier: Optional[TurnSignalsClassifier] = None
    phase_classifier: Optional[PhaseClassifier] = None
    action_classifier: Optional[ActionClassifier] = None
    retrieval_classifier: Optional[RetrievalClassifier] = None

    def __post_init__(self) -> None:
        if self.turn_signals_classifier is None:
            self.turn_signals_classifier = _DefaultTurnSignalsClassifier()
        if self.phase_classifier is None:
            self.phase_classifier = _DefaultPhaseClassifier()
        if self.action_classifier is None:
            self.action_classifier = _DefaultActionClassifier()
        if self.retrieval_classifier is None:
            self.retrieval_classifier = _DefaultRetrievalClassifier()

    def decide_action(self, context: TutorContext, state_manager: TutorStateManager) -> ActionDecision:
        """Produce an ActionDecision for the current turn.

        For intelligent/debug modes, we populate classifier-style fields on the
        TutorContext so downstream components can reason about them. For other
        modes we simply delegate to the decision engine.
        """

        mode_label = (self.mode or "").strip().lower()
        if mode_label in {"intelligent", "debug"}:
            try:
                turn_signals = self.turn_signals_classifier.classify(context)  # type: ignore[union-attr]
            except Exception:
                turn_signals = self._build_turn_signals(context)

            try:
                phase_suggestion = self.phase_classifier.classify(  # type: ignore[union-attr]
                    context,
                    turn_signals,
                )
            except Exception:
                phase_suggestion = self._build_phase_suggestion(context, turn_signals)

            try:
                action_suggestion = self.action_classifier.classify(  # type: ignore[union-attr]
                    context,
                    turn_signals,
                    phase_suggestion,
                )
            except Exception:
                action_suggestion = self._build_action_suggestion(context, turn_signals, phase_suggestion)

            try:
                retrieval_suggestion = self.retrieval_classifier.classify(  # type: ignore[union-attr]
                    context,
                    action_suggestion,
                    turn_signals,
                )
            except Exception:
                retrieval_suggestion = self._build_retrieval_suggestion(
                    context,
                    turn_signals,
                    phase_suggestion,
                    action_suggestion,
                )

            # Attach to context for downstream use.
            context.turn_signals = turn_signals
            context.phase_suggestion = phase_suggestion
            context.action_suggestion = action_suggestion
            context.retrieval_suggestion = retrieval_suggestion

        # Delegate the actual pedagogic decision to the unified engine.
        return self.decision_engine.decide_action(context, state_manager)

    # --- Internal helpers -------------------------------------------------

    def _build_turn_signals(self, context: TutorContext) -> TurnSignals:
        text = (context.message or "").strip().lower()

        # Confirmation heuristic
        if looks_like_confirmation(text):
            student_confirmation = "confirmed"
        elif any(p in text for p in ["maybe", "not sure", "not certain"]):
            student_confirmation = "uncertain"
        else:
            student_confirmation = "not_confirmed"

        # Study plan and orientation signals
        wants_study_plan = looks_like_study_plan_request(text)
        wants_orientation = context.current_state == TutorState.ORIENTATION and not context.focus_concept

        # Closure heuristic (mirror state_machine "student_signals_done" words)
        done_phrases = [
            "done",
            "finish",
            "end session",
            "that's all",
            "no more",
            "i'm done",
        ]
        wants_closure = any(p in text for p in done_phrases)

        # Reflection heuristic
        reflection_words = ["think", "understand", "confused", "clear", "get it"]
        reflection_provided = any(w in text for w in reflection_words) and len(text) > 50

        # Map existing intent to a meta-intent string
        intent = (context.intent or "").strip().lower()
        if intent in {"question"}:
            meta_intent = "ask_question"
        elif intent in {"answer", "reflection"}:
            meta_intent = "answer"
        elif intent in {"meta", "meta_talk"}:
            meta_intent = "meta_talk"
        else:
            meta_intent = None

        return TurnSignals(
            student_confirmation=student_confirmation,
            wants_orientation=wants_orientation,
            wants_study_plan=wants_study_plan,
            wants_closure=wants_closure,
            reflection_provided=reflection_provided,
            meta_intent=meta_intent,
        )

    def _build_phase_suggestion(
        self,
        context: TutorContext,
        turn_signals: TurnSignals,
    ) -> PhaseSuggestion:
        # Start from current state as default.
        suggested = context.current_state.value
        confidence = 0.5
        reason = "default_from_current_state"

        if turn_signals.wants_closure:
            suggested = "closure"
            confidence = 0.9
            reason = "student_signals_done"
        elif context.current_state == TutorState.ORIENTATION and context.focus_concept is None:
            suggested = "orientation"
            confidence = 0.9
            reason = "no_focus_concept_yet"
        elif context.current_state == TutorState.ASSESSMENT:
            suggested = "assessment"
            confidence = 0.8
            reason = "in_assessment_state"
        elif context.current_state == TutorState.REVIEW:
            suggested = "review"
            confidence = 0.8
            reason = "in_review_state"
        else:
            suggested = "teaching"
            confidence = 0.6
            reason = "teaching_default"

        return PhaseSuggestion(
            suggested_state=suggested,
            confidence=confidence,
            reason=reason,
        )

    def _build_action_suggestion(
        self,
        context: TutorContext,
        turn_signals: TurnSignals,
        phase_suggestion: PhaseSuggestion,
    ) -> ActionSuggestion:
        intent = (context.intent or "").strip().lower()
        state = context.current_state

        next_action = "explain"
        pedagogy_focus = None
        should_use_planning = False
        should_use_multi_step = False

        if phase_suggestion.suggested_state == "orientation":
            if context.focus_concept is None or turn_signals.wants_orientation:
                next_action = "orient"
                pedagogy_focus = ["orientation"]
            else:
                next_action = "ask"
                pedagogy_focus = ["prior_knowledge_activation"]
        elif phase_suggestion.suggested_state == "assessment":
            if intent == "answer":
                next_action = "reflect"
                pedagogy_focus = ["feedback", "metacognition"]
            else:
                next_action = "ask"
                pedagogy_focus = ["concept_check"]
            should_use_planning = True
        elif phase_suggestion.suggested_state == "review":
            next_action = "explain"
            pedagogy_focus = ["prerequisite_review", "example"]
            should_use_planning = True
        elif phase_suggestion.suggested_state == "closure":
            next_action = "preview"
            pedagogy_focus = ["summary"]
        else:  # teaching
            if intent in {"answer", "reflection"}:
                next_action = "reflect"
                pedagogy_focus = ["feedback"]
            else:
                next_action = "explain"
                pedagogy_focus = ["definition", "explanation"]
            should_use_planning = True

        # In auto/intelligent mode, multi-step planning is optional; keep it off by default here.
        return ActionSuggestion(
            current_action_interpretation=None,
            next_action=next_action,
            pedagogy_focus=pedagogy_focus,
            should_use_planning=should_use_planning,
            should_use_multi_step=should_use_multi_step,
            desired_state_after=None,
        )

    def _build_retrieval_suggestion(
        self,
        context: TutorContext,
        turn_signals: TurnSignals,
        phase_suggestion: PhaseSuggestion,
        action_suggestion: ActionSuggestion,
    ) -> RetrievalSuggestion:
        # Default: fresh retrieval on current focus concept or message.
        strategy = "fresh_retrieval"
        query: Optional[str] = None
        max_chunks = 8
        prefer_sections = None

        if context.focus_concept:
            query = context.focus_concept
        else:
            query = (context.message or "").strip()

        # If the student is just confirming and we already have retrieval chunks,
        # prefer to reuse them.
        if (
            turn_signals.student_confirmation == "confirmed"
            and context.retrieval_chunks
            and action_suggestion.next_action in {"explain", "ask", "reflect"}
        ):
            strategy = "reuse_last_chunks"

        # For review/assessment, prefer examples/applications.
        if phase_suggestion.suggested_state in {"review", "assessment"}:
            prefer_sections = ["example", "application"]

        return RetrievalSuggestion(
            strategy=strategy,
            retrieval_query=query,
            max_chunks=max_chunks,
            prefer_sections=prefer_sections,
        )


class _DefaultTurnSignalsClassifier:
    def classify(self, context: TutorContext) -> TurnSignals:
        return classify_turn_signals(context)


class _DefaultPhaseClassifier:
    def classify(
        self,
        context: TutorContext,
        signals: TurnSignals,
    ) -> PhaseSuggestion:
        return classify_phase(context, signals)


class _DefaultActionClassifier:
    def classify(
        self,
        context: TutorContext,
        signals: TurnSignals,
        phase: PhaseSuggestion,
    ) -> ActionSuggestion:
        return classify_action(context, signals, phase)


class _DefaultRetrievalClassifier:
    def classify(
        self,
        context: TutorContext,
        action: ActionSuggestion,
        signals: TurnSignals,
    ) -> RetrievalSuggestion:
        phase = PhaseSuggestion(suggested_state=context.current_state.value)
        return classify_retrieval(context, signals, phase, action)
