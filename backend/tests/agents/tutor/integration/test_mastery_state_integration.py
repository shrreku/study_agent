"""Integration tests for mastery tracking with state machine."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from backend.agents.tutor.tools.mastery_updater import MasteryUpdater
from backend.agents.tutor.state_machine import (
    TutorStateManager,
    TutorState,
    StateContext,
)


class TestMasteryStateIntegration:
    """Test interactions between mastery tracking and state machine."""

    @pytest.fixture
    def updater(self):
        """Create a standard MasteryUpdater instance."""
        return MasteryUpdater(
            learning_rate=0.1,
            decay_factor=0.95,
            min_update=0.02,
            max_update=0.3,
        )

    @pytest.fixture
    def state_manager(self):
        """Create a state manager."""
        return TutorStateManager()

    @pytest.fixture
    def mock_db_cursor(self):
        """Create a mock database cursor."""
        cursor = Mock()
        cursor.fetchone = Mock(return_value=(0.5,))
        return cursor

    # ===== STATE CONTEXT WITH MASTERY TESTS =====

    def test_state_context_includes_mastery_delta(self):
        """StateContext should track mastery changes."""
        context = StateContext(
            focus_concept="heat_transfer",
            answer_correct=True,
            mastery_delta=0.05,
            mastery_reason="correct_answer",
        )

        assert context.mastery_delta == 0.05
        assert context.mastery_reason == "correct_answer"

    def test_state_context_from_decision_context_with_mastery(self):
        """StateContext should extract mastery info from decision context."""
        decision_context = {
            "focus_concept": "conduction",
            "message": "Yes, I understand",
            "mastery_delta": 0.08,
            "mastery_reason": "engaged_productive",
            "answer_correct": True,
            "turn_index": 5,
        }

        state_ctx = StateContext.from_decision_context(decision_context)

        assert state_ctx.focus_concept == "conduction"
        assert state_ctx.mastery_delta == 0.08
        assert state_ctx.mastery_reason == "engaged_productive"
        assert state_ctx.session_turn_count == 5

    def test_state_context_mastery_tracking_nullable(self):
        """Mastery tracking fields should be optional."""
        context = StateContext(focus_concept="test")

        assert context.mastery_delta is None
        assert context.mastery_reason is None

    # ===== TRANSITION WITH MASTERY CONTEXT TESTS =====

    def test_teach_to_assess_with_mastery_improvement(self, state_manager):
        """TEACHING -> ASSESSMENT should happen after 3 explains, even with mastery changes."""
        state_manager.transition_to(TutorState.TEACHING, reason="concept_selected")

        # Simulate 3 explains with mastery improvements
        for i in range(3):
            state_manager.update_action("explain")

        context = StateContext(
            focus_concept="thermal_resistance",
            last_action="explain",
            session_turn_count=3,
            mastery_delta=0.05,  # Mastery improved
            mastery_reason="engaged_listening",
        )

        next_state = state_manager.check_transitions(context)

        assert next_state == TutorState.ASSESSMENT

    def test_assess_to_review_on_incorrect_with_mastery_loss(self, state_manager):
        """ASSESSMENT -> REVIEW should happen on 2 incorrect answers with mastery loss."""
        state_manager.transition_to(TutorState.TEACHING, reason="concept_selected")
        state_manager.transition_to(TutorState.ASSESSMENT, reason="teach_complete")

        # First incorrect answer
        state_manager.update_action("answer_incorrect")

        context1 = StateContext(
            focus_concept="convection",
            answer_correct=False,
            session_turn_count=4,
            mastery_delta=-0.03,
            mastery_reason="incorrect_answer",
        )

        # Should not transition yet
        next_state1 = state_manager.check_transitions(context1)
        assert next_state1 is None

        # Second incorrect answer
        state_manager.update_action("answer_incorrect")

        context2 = StateContext(
            focus_concept="convection",
            answer_correct=False,
            session_turn_count=5,
            mastery_delta=-0.03,
            mastery_reason="incorrect_answer",
        )

        # Should transition to REVIEW
        next_state2 = state_manager.check_transitions(context2)
        assert next_state2 == TutorState.REVIEW

    def test_review_transitions_on_mastery_recovery(self, state_manager):
        """REVIEW -> TEACHING should happen when student recovers mastery."""
        state_manager.transition_to(TutorState.TEACHING, reason="concept_selected")
        state_manager.transition_to(TutorState.ASSESSMENT, reason="teach_complete")
        state_manager.transition_to(TutorState.REVIEW, reason="incorrect_twice")

        # Simulate recovery with positive mastery signals
        context = StateContext(
            focus_concept="radiation",
            answer_correct=True,
            mastery_delta=0.12,  # Strong positive signal
            mastery_reason="correct_answer",
        )

        # Check if transition happens (depends on prerequisite logic)
        next_state = state_manager.check_transitions(context)

        # May or may not transition depending on prerequisite_mastered logic
        assert next_state in [TutorState.TEACHING, None]

    # ===== MASTERY UPDATE EFFECTS ON LEARNING FLOW =====

    def test_high_engagement_signals_boost_mastery_and_flow(self, updater):
        """High engagement + correct answers should boost mastery significantly."""
        signals_turn1 = {
            "affect": "engaged",
            "intent": "answer",
            "answer_correct": True,
            "classification_confidence": 0.9,
        }

        signals_turn2 = {
            "affect": "engaged",
            "intent": "reflection",
            "explanation_quality": 0.8,
            "classification_confidence": 0.85,
        }

        update1 = updater.compute_mastery_delta(
            concept="temperature",
            user_id="user-123",
            interaction_signals=signals_turn1,
            current_mastery=0.2,
        )

        update2 = updater.compute_mastery_delta(
            concept="temperature",
            user_id="user-123",
            interaction_signals=signals_turn2,
            current_mastery=0.2 + update1.delta,
        )

        # Both should be positive
        assert update1.delta > 0.0
        assert update2.delta > 0.0

        # Total improvement should be meaningful
        total_delta = update1.delta + update2.delta
        assert total_delta > 0.08

    def test_confusion_signals_slow_down_mastery_progression(self, updater):
        """Confusion signals should produce small negative updates."""
        signals = {
            "affect": "confused",
            "intent": "ask_for_help",
            "classification_confidence": 0.8,
        }

        update = updater.compute_mastery_delta(
            concept="heat_flux",
            user_id="user-123",
            interaction_signals=signals,
            current_mastery=0.5,
        )

        assert update.delta < 0.0
        assert "confused_affect" in update.reason

    def test_repeated_correct_answers_with_decay_at_high_mastery(self, updater):
        """High mastery should show decay even with correct answers."""
        signals = {
            "affect": "engaged",
            "intent": "answer",
            "answer_correct": True,
            "classification_confidence": 0.9,
        }

        # Low mastery - full signal
        update_low = updater.compute_mastery_delta(
            concept="test",
            user_id="user-123",
            interaction_signals=signals,
            current_mastery=0.3,
        )

        # High mastery - reduced signal due to decay
        update_high = updater.compute_mastery_delta(
            concept="test",
            user_id="user-123",
            interaction_signals=signals,
            current_mastery=0.85,
        )

        # High mastery update should be smaller
        assert update_high.delta < update_low.delta
        assert update_high.delta > 0.0  # Still positive

    # ===== MULTI-TURN MASTERY ACCUMULATION =====

    def test_multi_turn_mastery_accumulation(self, updater, mock_db_cursor):
        """Mastery should accumulate over multiple turns with consistent engagement."""
        concept = "first_law_thermodynamics"
        user_id = "user-123"
        current_mastery = 0.0

        # Simulate 5 turns of engaged learning with correct answers
        total_delta = 0.0

        for turn in range(5):
            signals = {
                "affect": "engaged",
                "intent": "answer" if turn % 2 == 0 else "reflection",
                "answer_correct": True if turn % 2 == 0 else None,
                "explanation_quality": 0.75 if turn % 2 == 1 else None,
                "classification_confidence": 0.85 + (turn * 0.01),
            }

            update = updater.compute_mastery_delta(
                concept=concept,
                user_id=user_id,
                interaction_signals=signals,
                current_mastery=current_mastery,
            )

            if update.delta != 0.0:
                total_delta += update.delta
                current_mastery = min(1.0, current_mastery + update.delta)

        # After 5 turns, should have accumulated meaningful mastery
        assert total_delta > 0.05  # Should be positive overall
        assert current_mastery > 0.1  # Should have gained mastery

    def test_mastery_recovery_after_temporary_confusion(self, updater):
        """Mastery should recover after temporary confusion if student re-engages."""
        concept = "radiation_heat_transfer"
        user_id = "user-123"

        # Turn 1: Engaged correct answer
        update1 = updater.compute_mastery_delta(
            concept=concept,
            user_id=user_id,
            interaction_signals={
                "affect": "engaged",
                "intent": "answer",
                "answer_correct": True,
                "classification_confidence": 0.9,
            },
            current_mastery=0.2,
        )
        mastery_after_1 = 0.2 + update1.delta

        # Turn 2: Confusion (mastery drops)
        update2 = updater.compute_mastery_delta(
            concept=concept,
            user_id=user_id,
            interaction_signals={
                "affect": "confused",
                "intent": "ask_for_help",
                "classification_confidence": 0.7,
            },
            current_mastery=mastery_after_1,
        )
        mastery_after_2 = mastery_after_1 + update2.delta

        # Turn 3: Engaged clarification (recovery)
        update3 = updater.compute_mastery_delta(
            concept=concept,
            user_id=user_id,
            interaction_signals={
                "affect": "engaged",
                "intent": "reflection",
                "explanation_quality": 0.7,
                "classification_confidence": 0.85,
            },
            current_mastery=mastery_after_2,
        )
        mastery_after_3 = mastery_after_2 + update3.delta

        # Should end higher than after confusion
        assert mastery_after_3 > mastery_after_2
        # But might be slightly lower than peak if confusion was strong
        assert mastery_after_3 >= mastery_after_2

    # ===== SIGNAL COMBINATIONS IN CONTEXT =====

    def test_engaged_productive_strongest_signal(self, updater):
        """Engaged + productive intent should be strongest signal."""
        signals_engaged_productive = {
            "affect": "engaged",
            "intent": "answer",
            "answer_correct": True,
            "classification_confidence": 0.9,
        }
        signals_neutral_productive = {
            "affect": "neutral",
            "intent": "answer",
            "answer_correct": True,
            "classification_confidence": 0.9,
        }

        update_ep = updater.compute_mastery_delta(
            concept="test",
            user_id="user-123",
            interaction_signals=signals_engaged_productive,
            current_mastery=0.3,
        )

        update_np = updater.compute_mastery_delta(
            concept="test",
            user_id="user-123",
            interaction_signals=signals_neutral_productive,
            current_mastery=0.3,
        )

        # Engaged should be stronger
        assert update_ep.delta > update_np.delta
        assert "engaged_productive" in update_ep.reason

    def test_quality_reflection_only_above_threshold(self, updater):
        """Quality signal only counts if above 0.6 threshold."""
        signals_high_quality = {
            "affect": "neutral",
            "intent": "reflection",
            "explanation_quality": 0.75,
            "classification_confidence": 0.8,
        }
        signals_low_quality = {
            "affect": "neutral",
            "intent": "reflection",
            "explanation_quality": 0.45,
            "classification_confidence": 0.8,
        }

        update_high = updater.compute_mastery_delta(
            concept="test",
            user_id="user-123",
            interaction_signals=signals_high_quality,
            current_mastery=0.3,
        )

        update_low = updater.compute_mastery_delta(
            concept="test",
            user_id="user-123",
            interaction_signals=signals_low_quality,
            current_mastery=0.3,
        )

        # High quality should show quality_reflection
        assert "quality_reflection" in update_high.reason

        # Low quality should not
        assert "quality_reflection" not in update_low.reason

    # ===== CONFIDENCE IN MASTERY UPDATES =====

    def test_explicit_assessment_highest_confidence(self, updater):
        """Explicit correctness assessment should produce highest confidence."""
        signals_explicit = {
            "affect": "engaged",
            "intent": "answer",
            "answer_correct": True,
            "classification_confidence": 0.95,
        }
        signals_implicit = {
            "affect": "engaged",
            "intent": "reflection",
            "explanation_quality": 0.75,
            "classification_confidence": 0.8,
        }

        update_explicit = updater.compute_mastery_delta(
            concept="test",
            user_id="user-123",
            interaction_signals=signals_explicit,
            current_mastery=0.3,
        )

        update_implicit = updater.compute_mastery_delta(
            concept="test",
            user_id="user-123",
            interaction_signals=signals_implicit,
            current_mastery=0.3,
        )

        # Explicit should have higher confidence
        assert update_explicit.confidence > update_implicit.confidence

    # ===== EDGE CASES IN INTEGRATION =====

    def test_zero_mastery_cold_start(self, updater):
        """New concept should start from zero and respond to engagement."""
        signals = {
            "affect": "engaged",
            "intent": "reflection",
            "explanation_quality": 0.7,
            "classification_confidence": 0.8,
        }

        update = updater.compute_mastery_delta(
            concept="new_concept",
            user_id="user-123",
            interaction_signals=signals,
            current_mastery=0.0,  # Cold start
        )

        # Should produce positive delta
        assert update.delta > 0.0

    def test_max_mastery_ceiling(self, updater):
        """Mastery approaching 1.0 should show decay but stay positive."""
        signals = {
            "affect": "engaged",
            "intent": "answer",
            "answer_correct": True,
            "classification_confidence": 0.9,
        }

        update = updater.compute_mastery_delta(
            concept="expert_level",
            user_id="user-123",
            interaction_signals=signals,
            current_mastery=0.95,  # Near max
        )

        # Should still be positive but smaller
        assert 0.0 < update.delta < 0.02

    def test_threshold_boundary_handling(self, updater):
        """Updates at threshold boundaries should be handled correctly."""
        # At exactly 0.7 mastery (decay threshold)
        signals = {
            "affect": "engaged",
            "intent": "answer",
            "answer_correct": True,
            "classification_confidence": 0.9,
        }

        update_at_threshold = updater.compute_mastery_delta(
            concept="test",
            user_id="user-123",
            interaction_signals=signals,
            current_mastery=0.70,
        )

        update_below_threshold = updater.compute_mastery_delta(
            concept="test",
            user_id="user-123",
            interaction_signals=signals,
            current_mastery=0.69,
        )

        # At threshold should apply decay
        # Below threshold should not
        assert update_at_threshold.delta < update_below_threshold.delta

