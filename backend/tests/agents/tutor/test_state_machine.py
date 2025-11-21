"""
Tests for TutorStateManager state machine logic.
"""

import pytest
from backend.agents.tutor.state_machine import (
    TutorState,
    TutorStateManager,
    StateContext,
    StateTransition,
)


class TestTutorStateManager:
    """Test state manager initialization and basic operations."""

    def test_init_default_state(self):
        """Test initialization with default ORIENTATION state."""
        manager = TutorStateManager()
        assert manager.current_state == TutorState.ORIENTATION
        assert len(manager.state_history) == 1
        assert manager.state_history[0] == TutorState.ORIENTATION

    def test_init_custom_state(self):
        """Test initialization with custom initial state."""
        manager = TutorStateManager(initial_state=TutorState.TEACHING)
        assert manager.current_state == TutorState.TEACHING
        assert manager.state_history[0] == TutorState.TEACHING

    def test_state_counters_initialized(self):
        """Test that state counters are initialized."""
        manager = TutorStateManager()
        assert "consecutive_explains" in manager.state_counters
        assert "consecutive_asks" in manager.state_counters
        assert "incorrect_answers_in_assessment" in manager.state_counters
        assert manager.state_counters["consecutive_explains"] == 0


class TestStateTransitions:
    """Test state transition logic."""

    def test_orientation_to_teaching_with_concept(self):
        """Test ORIENTATION -> TEACHING when concept is selected."""
        manager = TutorStateManager(initial_state=TutorState.ORIENTATION)
        context = StateContext(focus_concept="Heat Transfer")

        new_state = manager.check_transitions(context)
        assert new_state == TutorState.TEACHING

    def test_orientation_to_teaching_without_concept(self):
        """Test ORIENTATION -> TEACHING does not trigger without concept."""
        manager = TutorStateManager(initial_state=TutorState.ORIENTATION)
        context = StateContext(focus_concept=None)

        new_state = manager.check_transitions(context)
        assert new_state is None

    def test_teaching_to_assessment_after_3_explains(self):
        """Test TEACHING -> ASSESSMENT after 3 consecutive explains."""
        manager = TutorStateManager(initial_state=TutorState.TEACHING)
        manager.state_counters["consecutive_explains"] = 3
        context = StateContext()

        new_state = manager.check_transitions(context)
        assert new_state == TutorState.ASSESSMENT

    def test_teaching_to_assessment_threshold(self):
        """Test TEACHING -> ASSESSMENT only at threshold."""
        manager = TutorStateManager(initial_state=TutorState.TEACHING)
        manager.state_counters["consecutive_explains"] = 2
        context = StateContext()

        new_state = manager.check_transitions(context)
        assert new_state is None

    def test_assessment_to_teaching_on_correct_answer(self):
        """Test ASSESSMENT -> TEACHING when answer is correct."""
        manager = TutorStateManager(initial_state=TutorState.ASSESSMENT)
        context = StateContext(answer_correct=True)

        new_state = manager.check_transitions(context)
        assert new_state == TutorState.TEACHING

    def test_assessment_to_review_on_incorrect_twice(self):
        """Test ASSESSMENT -> REVIEW after 2 incorrect answers."""
        manager = TutorStateManager(initial_state=TutorState.ASSESSMENT)
        manager.state_counters["incorrect_answers_in_assessment"] = 2
        context = StateContext()

        new_state = manager.check_transitions(context)
        assert new_state == TutorState.REVIEW

    def test_assessment_to_review_single_incorrect(self):
        """Test ASSESSMENT does not transition to REVIEW on single incorrect."""
        manager = TutorStateManager(initial_state=TutorState.ASSESSMENT)
        manager.state_counters["incorrect_answers_in_assessment"] = 1
        context = StateContext()

        new_state = manager.check_transitions(context)
        assert new_state is None

    def test_any_state_to_closure_on_done_signal(self):
        """Test any state -> CLOSURE when student signals done."""
        for state in [TutorState.TEACHING, TutorState.ASSESSMENT, TutorState.REVIEW]:
            manager = TutorStateManager(initial_state=state)
            context = StateContext(student_message="I'm done with this topic")

            new_state = manager.check_transitions(context)
            assert (
                new_state == TutorState.CLOSURE
            ), f"Failed for state {state.value}"

    def test_closure_priority_over_others(self):
        """Test that closure signal takes priority over other transitions."""
        manager = TutorStateManager(initial_state=TutorState.TEACHING)
        manager.state_counters["consecutive_explains"] = 3  # Would trigger ASSESSMENT
        context = StateContext(
            student_message="I'm done",  # But closure has higher priority
            focus_concept="Heat Transfer",
        )

        new_state = manager.check_transitions(context)
        assert (
            new_state == TutorState.CLOSURE
        ), "Closure should have priority over ASSESSMENT transition"


class TestActionTracking:
    """Test action tracking and counter updates."""

    def test_consecutive_explains_tracking(self):
        """Test tracking of consecutive explains."""
        manager = TutorStateManager()
        
        manager.update_action("explain")
        assert manager.state_counters["consecutive_explains"] == 1
        
        manager.update_action("explain")
        assert manager.state_counters["consecutive_explains"] == 2
        
        manager.update_action("ask")
        assert manager.state_counters["consecutive_explains"] == 0

    def test_consecutive_asks_tracking(self):
        """Test tracking of consecutive asks."""
        manager = TutorStateManager()
        
        manager.update_action("ask")
        assert manager.state_counters["consecutive_asks"] == 1
        
        manager.update_action("ask")
        assert manager.state_counters["consecutive_asks"] == 2
        
        manager.update_action("explain")
        assert manager.state_counters["consecutive_asks"] == 0

    def test_record_incorrect_answer(self):
        """Test recording incorrect answers."""
        manager = TutorStateManager()
        
        assert manager.state_counters["incorrect_answers_in_assessment"] == 0
        
        manager.record_incorrect_answer()
        assert manager.state_counters["incorrect_answers_in_assessment"] == 1
        
        manager.record_incorrect_answer()
        assert manager.state_counters["incorrect_answers_in_assessment"] == 2


class TestStateTransitionExecution:
    """Test actual state transition execution."""

    def test_transition_execution(self):
        """Test state transition is executed properly."""
        manager = TutorStateManager(initial_state=TutorState.ORIENTATION)
        
        manager.transition_to(TutorState.TEACHING, reason="concept_selected")
        
        assert manager.current_state == TutorState.TEACHING
        assert manager.state_history == [TutorState.ORIENTATION, TutorState.TEACHING]

    def test_transition_resets_counters(self):
        """Test that transitions reset appropriate counters."""
        manager = TutorStateManager(initial_state=TutorState.TEACHING)
        manager.state_counters["consecutive_explains"] = 5
        manager.state_counters["consecutive_asks"] = 2
        
        # TEACHING -> ASSESSMENT should reset these
        manager.transition_to(TutorState.ASSESSMENT, reason="3_consecutive_explains")
        
        assert manager.state_counters["consecutive_explains"] == 0
        assert manager.state_counters["consecutive_asks"] == 0

    def test_transition_history_tracking(self):
        """Test that state history is properly tracked."""
        manager = TutorStateManager(initial_state=TutorState.ORIENTATION)
        
        manager.transition_to(TutorState.TEACHING, reason="concept_selected")
        manager.transition_to(TutorState.ASSESSMENT, reason="3_explains")
        manager.transition_to(TutorState.REVIEW, reason="incorrect_twice")
        
        expected = [
            TutorState.ORIENTATION,
            TutorState.TEACHING,
            TutorState.ASSESSMENT,
            TutorState.REVIEW,
        ]
        assert manager.state_history == expected


class TestPersistence:
    """Test state persistence and restoration."""

    def test_get_state_for_persistence(self):
        """Test getting state data for persistence."""
        manager = TutorStateManager(initial_state=TutorState.TEACHING)
        manager.state_counters["consecutive_explains"] = 2
        manager.state_counters["last_action"] = "explain"
        
        data = manager.get_state_for_persistence()
        
        assert data["current_state"] == "teaching"
        assert data["state_counters"]["consecutive_explains"] == 2
        assert data["state_counters"]["last_action"] == "explain"

    def test_from_dict_restore_state(self):
        """Test restoring state from persisted data."""
        original = TutorStateManager(initial_state=TutorState.TEACHING)
        original.state_counters["consecutive_explains"] = 2
        original.transition_to(TutorState.ASSESSMENT, reason="test")
        
        data = original.get_state_for_persistence()
        restored = TutorStateManager.from_dict(data)
        
        assert restored.current_state == original.current_state
        assert restored.state_history == original.state_history
        assert (
            restored.state_counters["consecutive_explains"]
            == original.state_counters["consecutive_explains"]
        )

    def test_from_dict_empty(self):
        """Test from_dict with empty data."""
        manager = TutorStateManager.from_dict(None)
        assert manager.current_state == TutorState.ORIENTATION

    def test_from_dict_invalid_state(self):
        """Test from_dict with invalid state string."""
        data = {"current_state": "invalid_state"}
        manager = TutorStateManager.from_dict(data)
        assert manager.current_state == TutorState.ORIENTATION


class TestStateSnapshot:
    """Test snapshot creation."""

    def test_snapshot_captures_state(self):
        """Test that snapshot captures current state."""
        manager = TutorStateManager(initial_state=TutorState.TEACHING)
        manager.state_counters["consecutive_explains"] = 2
        
        snapshot = manager.snapshot()
        
        assert snapshot.current_state == TutorState.TEACHING
        assert snapshot.state_counters["consecutive_explains"] == 2
        assert TutorState.TEACHING in snapshot.state_history


class TestReflectionCondition:
    """Test reflection-based transition condition."""

    def test_reflection_condition_with_reflection_words(self):
        """Test TEACHING -> ASSESSMENT on student reflection."""
        manager = TutorStateManager(initial_state=TutorState.TEACHING)
        context = StateContext(
            student_message="I think I understand now. The concept of heat transfer is becoming clearer."
        )

        new_state = manager.check_transitions(context)
        assert new_state == TutorState.ASSESSMENT

    def test_reflection_condition_without_reflection_words(self):
        """Test reflection condition without reflection words."""
        manager = TutorStateManager(initial_state=TutorState.TEACHING)
        context = StateContext(student_message="ok")

        new_state = manager.check_transitions(context)
        assert new_state is None

    def test_reflection_condition_short_message(self):
        """Test reflection condition ignores short messages."""
        manager = TutorStateManager(initial_state=TutorState.TEACHING)
        context = StateContext(student_message="I think so.")

        new_state = manager.check_transitions(context)
        assert new_state is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

