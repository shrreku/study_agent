"""
Tutor State Machine: Explicit state management with transitions.

Implements a clear learning state progression:
ORIENTATION -> TEACHING -> ASSESSMENT -> REVIEW -> CLOSURE

Each state has well-defined entry/exit conditions and counter tracking.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from .auto_classifiers import TurnSignals, PhaseSuggestion


# Ensure logging is configured even if module is loaded dynamically
logger = logging.getLogger("tutor.state_machine")


class TutorState(Enum):
    """Explicit tutor session states."""

    ORIENTATION = "orientation"  # Initial grounding, concept selection
    TEACHING = "teaching"  # Main instruction phase
    ASSESSMENT = "assessment"  # Checking student understanding
    REVIEW = "review"  # Revisiting weak areas, prerequisites
    CLOSURE = "closure"  # Session wrap-up


@dataclass
class StateTransition:
    """A state transition rule with conditions and priority."""

    from_state: Optional[TutorState]  # None means "from any state"
    to_state: TutorState
    condition: str  # Semantic condition name
    priority: int = 0  # Higher priority evaluated first

    def __post_init__(self) -> None:
        """Validate transition."""
        if self.to_state is None:
            raise ValueError("to_state cannot be None")


@dataclass
class StateContext:
    """Context needed for state transitions."""

    focus_concept: Optional[str] = None
    answer_correct: Optional[bool] = None
    answer_quality: Optional[str] = None  # "correct", "partial", "incorrect"
    student_message: str = ""
    last_action: Optional[str] = None
    mastery_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    session_turn_count: int = 0
    # Phase 5: Mastery state tracking
    mastery_delta: Optional[float] = None  # Mastery change this turn
    mastery_reason: Optional[str] = None  # Why mastery changed
    turn_signals: Optional[TurnSignals] = None
    phase_suggestion: Optional[PhaseSuggestion] = None

    @classmethod
    def from_decision_context(cls, context: Dict[str, Any]) -> StateContext:
        """Build StateContext from decision context."""
        return cls(
            focus_concept=context.get("focus_concept"),
            answer_correct=context.get("answer_correct"),
            answer_quality=context.get("answer_quality"),
            student_message=context.get("message", ""),
            last_action=context.get("last_action"),
            mastery_map=context.get("mastery_map", {}),
            session_turn_count=context.get("turn_index", 0),
            mastery_delta=context.get("mastery_delta"),
            mastery_reason=context.get("mastery_reason"),
        )


@dataclass
class StateSnapshot:
    """Snapshot of state at a point in time."""

    current_state: TutorState
    state_history: List[TutorState]
    state_counters: Dict[str, int]
    turn_updated: int


class TutorStateManager:
    """Manages explicit state transitions with clear rules."""

    # Define all possible transitions with priorities
    TRANSITIONS = [
        # ORIENTATION exits
        StateTransition(
            from_state=TutorState.ORIENTATION,
            to_state=TutorState.TEACHING,
            condition="concept_selected",
            priority=10,
        ),
        # TEACHING -> ASSESSMENT (after sufficient teaching)
        StateTransition(
            from_state=TutorState.TEACHING,
            to_state=TutorState.ASSESSMENT,
            condition="3_consecutive_explains",
            priority=8,
        ),
        StateTransition(
            from_state=TutorState.TEACHING,
            to_state=TutorState.ASSESSMENT,
            condition="student_reflection_complete",
            priority=5,
        ),
        # ASSESSMENT -> TEACHING (student got it right)
        StateTransition(
            from_state=TutorState.ASSESSMENT,
            to_state=TutorState.TEACHING,
            condition="answer_correct",
            priority=10,
        ),
        # ASSESSMENT -> REVIEW (student struggled)
        StateTransition(
            from_state=TutorState.ASSESSMENT,
            to_state=TutorState.REVIEW,
            condition="answer_incorrect_twice",
            priority=9,
        ),
        # REVIEW -> TEACHING (prerequisite mastered)
        StateTransition(
            from_state=TutorState.REVIEW,
            to_state=TutorState.TEACHING,
            condition="prerequisite_mastered",
            priority=7,
        ),
        # Any state -> CLOSURE (highest priority)
        StateTransition(
            from_state=None,
            to_state=TutorState.CLOSURE,
            condition="student_signals_done",
            priority=100,
        ),
    ]

    def __init__(
        self, initial_state: TutorState = TutorState.ORIENTATION
    ) -> None:
        """Initialize state manager."""
        self.current_state = initial_state
        self.state_history: List[TutorState] = [initial_state]
        self.state_counters: Dict[str, int] = {
            "consecutive_explains": 0,
            "consecutive_asks": 0,
            "incorrect_answers_in_assessment": 0,
            "turns_in_teaching": 0,
            "turns_in_assessment": 0,
            "turns_in_review": 0,
        }

    def check_transitions(self, context: StateContext) -> Optional[TutorState]:
        """
        Check if any transition conditions are met.

        Returns the new state if a transition should occur, None otherwise.
        """
        # Filter applicable transitions
        applicable = [
            t
            for t in self.TRANSITIONS
            if t.from_state is None or t.from_state == self.current_state
        ]

        # Sort by priority (highest first)
        applicable.sort(key=lambda t: t.priority, reverse=True)

        # Check conditions in priority order
        for transition in applicable:
            if self._evaluate_condition(transition.condition, context):
                logger.info(
                    f"tutor_state_transition_ready "
                    f"current={self.current_state.value} "
                    f"next={transition.to_state.value} "
                    f"condition={transition.condition}"
                )
                return transition.to_state

        return None

    def transition_to(
        self, new_state: TutorState, reason: str, context: Optional[StateContext] = None
    ) -> None:
        """Execute a state transition."""
        old_state = self.current_state
        self.current_state = new_state
        self.state_history.append(new_state)

        # Reset counters on certain transitions
        if old_state == TutorState.TEACHING and new_state == TutorState.ASSESSMENT:
            self.state_counters["consecutive_explains"] = 0
            self.state_counters["consecutive_asks"] = 0

        if old_state == TutorState.ASSESSMENT and new_state == TutorState.TEACHING:
            self.state_counters["incorrect_answers_in_assessment"] = 0

        logger.info(
            f"tutor_state_transition "
            f"{old_state.value} -> {new_state.value} "
            f"reason={reason}"
        )

    def update_action(self, action: str) -> None:
        """Update counters based on action."""
        if action == "explain":
            if self.state_counters.get("last_action") == "explain":
                self.state_counters["consecutive_explains"] = (
                    self.state_counters.get("consecutive_explains", 0) + 1
                )
            else:
                self.state_counters["consecutive_explains"] = 1
            self.state_counters["consecutive_asks"] = 0

        elif action == "ask":
            self.state_counters["consecutive_explains"] = 0
            self.state_counters["consecutive_asks"] = (
                self.state_counters.get("consecutive_asks", 0) + 1
            )

        else:
            self.state_counters["consecutive_explains"] = 0
            self.state_counters["consecutive_asks"] = 0

        self.state_counters["last_action"] = action

    def record_incorrect_answer(self) -> None:
        """Record incorrect answer for tracking."""
        self.state_counters["incorrect_answers_in_assessment"] = (
            self.state_counters.get("incorrect_answers_in_assessment", 0) + 1
        )

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Get state data for persistence."""
        return {
            "current_state": self.current_state.value,
            "state_history": [s.value for s in self.state_history],
            "state_counters": dict(self.state_counters),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> TutorStateManager:
        """Restore state manager from persisted data."""
        if not data:
            return cls()

        manager = cls()

        # Restore current state
        state_str = data.get("current_state", "orientation")
        try:
            manager.current_state = TutorState(state_str)
        except ValueError:
            logger.warning(f"Invalid state string: {state_str}, using ORIENTATION")
            manager.current_state = TutorState.ORIENTATION

        # Restore history
        history_strs = data.get("state_history", ["orientation"])
        manager.state_history = []
        for state_str in history_strs:
            try:
                manager.state_history.append(TutorState(state_str))
            except ValueError:
                logger.warning(f"Skipping invalid history state: {state_str}")

        # Ensure current state is in history
        if manager.current_state not in manager.state_history:
            manager.state_history.append(manager.current_state)

        # Restore counters
        manager.state_counters = data.get("state_counters", manager.state_counters)

        return manager

    def _evaluate_condition(self, condition: str, context: StateContext) -> bool:
        """Evaluate transition condition."""
        # Highest priority: student signals done
        if condition == "student_signals_done":
            ts = getattr(context, "turn_signals", None)
            if ts and getattr(ts, "wants_closure", False):
                return True
            message = context.student_message.lower()
            done_phrases = [
                "done",
                "finish",
                "end session",
                "that's all",
                "no more",
                "i'm done",
            ]
            return any(phrase in message for phrase in done_phrases)

        # ORIENTATION -> TEACHING
        if condition == "concept_selected":
            return context.focus_concept is not None

        # TEACHING -> ASSESSMENT (after 3 consecutive explains)
        if condition == "3_consecutive_explains":
            return self.state_counters.get("consecutive_explains", 0) >= 3

        # TEACHING -> ASSESSMENT (student provided reflection)
        if condition == "student_reflection_complete":
            ts = getattr(context, "turn_signals", None)
            if ts and getattr(ts, "reflection_provided", False):
                return True
            reflection_words = ["think", "understand", "confused", "clear", "get it"]
            message_lower = context.student_message.lower()
            has_reflection = any(word in message_lower for word in reflection_words)
            is_long = len(context.student_message) > 50
            return has_reflection and is_long

        # ASSESSMENT -> TEACHING (student answered correctly)
        if condition == "answer_correct":
            return context.answer_correct is True

        # ASSESSMENT -> REVIEW (student struggled - 2 incorrect)
        if condition == "answer_incorrect_twice":
            return (
                self.state_counters.get("incorrect_answers_in_assessment", 0) >= 2
            )

        # REVIEW -> TEACHING (prerequisite understood)
        if condition == "prerequisite_mastered":
            # Check if prerequisite concept has decent mastery
            # This is a simplification; real impl would check mastery_map
            return self.state_counters.get("turns_in_review", 0) >= 2

        return False

    def snapshot(self) -> StateSnapshot:
        """Create a snapshot of current state."""
        return StateSnapshot(
            current_state=self.current_state,
            state_history=list(self.state_history),
            state_counters=dict(self.state_counters),
            turn_updated=0,  # Will be set by caller
        )

