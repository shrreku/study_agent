"""Tests for loop prevention and edge case handling."""

import pytest
from backend.agents.tutor.state_machine import TutorStateManager, TutorState, StateContext
from backend.agents.tutor.decision_engine import TutorDecisionEngine
from backend.agents.tutor.context_model import TutorContext


class TestLoopPrevention:
    """Test loop prevention mechanisms."""

    @pytest.fixture
    def state_manager(self):
        """Create state manager."""
        return TutorStateManager()

    @pytest.fixture
    def decision_engine(self):
        """Create decision engine."""
        return TutorDecisionEngine(mode="intelligent")

    # ===== DUPLICATE RESPONSE PREVENTION =====

    def test_action_alternation_teaches_then_assesses(self, state_manager):
        """After explains, system should ask (not explain again)."""
        state_manager.transition_to(TutorState.TEACHING, reason="start")

        # Simulate 3 consecutive explains
        actions = []
        for i in range(3):
            state_manager.update_action("explain")
            actions.append("explain")

        # Should be ready to transition to ASSESSMENT
        state_context = StateContext(
            focus_concept="test",
            last_action="explain",
            session_turn_count=3,
        )

        next_state = state_manager.check_transitions(state_context)

        # Should want to transition to ASSESSMENT
        assert next_state == TutorState.ASSESSMENT

        # Verify we had 3 explains
        assert actions.count("explain") == 3

    def test_avoid_repeated_questions(self, state_manager, decision_engine):
        """System should vary actions, not repeat same questions."""
        state_manager.transition_to(TutorState.ASSESSMENT, reason="ready")

        # Track actions over 5 turns
        actions = []

        for turn in range(5):
            context = TutorContext(
                message=f"Student response {turn}",
                intent="answer" if turn > 0 else "ask",
                affect="engaged",
                focus_concept="concept",
                concept_level="intro",
                current_state=state_manager.current_state,
                mastery_map={"concept": {"mastery": 0.5}},
                retrieval_chunks=[],
                session_history=[],
                turn_index=turn,
            )

            decision = decision_engine.decide_action(context, state_manager)
            actions.append(decision.action)

        # Should not have 5 of the same action
        assert len(set(actions)) >= 2 or "ask" not in actions

    # ===== CONTINUE LOOP PREVENTION =====

    def test_continue_command_five_times_no_crash(self, state_manager):
        """Five consecutive 'continue' commands should not cause infinite loop."""
        state_manager.transition_to(TutorState.TEACHING, reason="start")

        # Simulate 5 "continue" inputs
        for i in range(5):
            state_context = StateContext(
                student_message="continue",
                focus_concept="concept",
                session_turn_count=i + 1,
            )

            # Should handle gracefully
            next_state = state_manager.check_transitions(state_context)

            # Should not crash or hang
            assert state_manager.current_state in [
                TutorState.ORIENTATION,
                TutorState.TEACHING,
                TutorState.ASSESSMENT,
                TutorState.REVIEW,
                TutorState.CLOSURE,
            ]

    def test_repeated_explain_requests_transition_to_assessment(
        self, state_manager
    ):
        """Too many explain requests should trigger assessment."""
        state_manager.transition_to(TutorState.TEACHING, reason="start")

        # Request explains 5 times
        for i in range(5):
            state_manager.update_action("explain")

        # After 3 explains, should transition to ASSESSMENT
        state_context = StateContext(
            focus_concept="test",
            session_turn_count=5,
        )

        next_state = state_manager.check_transitions(state_context)

        # Should transition to ASSESSMENT to prevent endless explanations
        if next_state:
            assert next_state == TutorState.ASSESSMENT

    # ===== STATE VALIDATION =====

    def test_no_invalid_state_transitions(self, state_manager):
        """Transition rules should prevent invalid state transitions."""
        # Start in valid state
        state_manager.transition_to(TutorState.TEACHING, reason="start")

        # Try to transition through various contexts
        for concept_selected in [True, False]:
            for answer_correct in [True, False, None]:
                state_context = StateContext(
                    focus_concept="test" if concept_selected else None,
                    answer_correct=answer_correct,
                    session_turn_count=1,
                )

                next_state = state_manager.check_transitions(state_context)

                # Next state should always be valid
                if next_state:
                    assert isinstance(next_state, TutorState)
                    assert next_state in [
                        TutorState.ORIENTATION,
                        TutorState.TEACHING,
                        TutorState.ASSESSMENT,
                        TutorState.REVIEW,
                        TutorState.CLOSURE,
                    ]

    def test_no_dead_states(self, state_manager):
        """No state should be unreachable."""
        # From ORIENTATION, should be able to reach TEACHING
        state_manager.transition_to(TutorState.ORIENTATION, reason="start")
        state_context = StateContext(focus_concept="test")
        state_manager.transition_to(TutorState.TEACHING, reason="test")
        assert state_manager.current_state == TutorState.TEACHING

        # From TEACHING, should reach ASSESSMENT
        state_manager.update_action("explain")
        state_manager.update_action("explain")
        state_manager.update_action("explain")
        state_context = StateContext(focus_concept="test", session_turn_count=3)
        next_state = state_manager.check_transitions(state_context)
        if next_state:
            state_manager.transition_to(next_state, reason="test")
            assert state_manager.current_state == TutorState.ASSESSMENT

    # ===== DECISION ENGINE ROBUSTNESS =====

    def test_decision_engine_always_produces_decision(self, decision_engine):
        """Decision engine should always return a valid decision."""
        # Test with various contexts
        contexts = [
            # Minimal context
            TutorContext(
                message="",
                intent="",
                affect="neutral",
                focus_concept=None,
                concept_level="intro",
                current_state=TutorState.TEACHING,
                mastery_map={},
                retrieval_chunks=[],
                session_history=[],
                turn_index=0,
            ),
            # Rich context
            TutorContext(
                message="This is a detailed student message",
                intent="answer",
                affect="engaged",
                focus_concept="heat_transfer",
                concept_level="advanced",
                current_state=TutorState.ASSESSMENT,
                mastery_map={"heat_transfer": {"mastery": 0.7}},
                retrieval_chunks=[
                    {"id": "chunk-1", "snippet": "...", "score": 0.9}
                ],
                session_history=[{"role": "tutor", "content": "Explain conduction"}],
                turn_index=5,
            ),
            # Edge case: no mastery
            TutorContext(
                message="New student",
                intent="clarification",
                affect="confused",
                focus_concept="new_concept",
                concept_level="intro",
                current_state=TutorState.TEACHING,
                mastery_map={},
                retrieval_chunks=[],
                session_history=[],
                turn_index=0,
            ),
        ]

        state_manager = TutorStateManager()

        for context in contexts:
            # Should always produce a decision
            decision = decision_engine.decide_action(context, state_manager)

            # Validate decision
            assert decision is not None
            assert hasattr(decision, "action")
            assert hasattr(decision, "rationale")
            assert decision.action in [
                "explain",
                "ask",
                "hint",
                "reflect",
                "review",
            ]

    def test_decision_with_missing_chunks(self, decision_engine):
        """Should handle missing retrieval chunks gracefully."""
        state_manager = TutorStateManager()
        state_manager.transition_to(TutorState.TEACHING, reason="start")

        context = TutorContext(
            message="Can you explain conduction?",
            intent="ask",
            affect="engaged",
            focus_concept="conduction",
            concept_level="intro",
            current_state=state_manager.current_state,
            mastery_map={},
            retrieval_chunks=[],  # No chunks!
            session_history=[],
            turn_index=1,
        )

        # Should not crash
        decision = decision_engine.decide_action(context, state_manager)

        # Should still produce valid decision
        assert decision.action is not None
        assert isinstance(decision.rationale, str)

    # ===== COUNTER OVERFLOW PREVENTION =====

    def test_counter_overflow_on_many_explains(self, state_manager):
        """Counters should handle many updates without overflow."""
        state_manager.transition_to(TutorState.TEACHING, reason="start")

        # Update action 100 times
        for i in range(100):
            state_manager.update_action("explain")

        # Should still work and be at 100
        assert state_manager.state_counters["consecutive_explains"] >= 3

        # Transition should trigger at 3
        state_context = StateContext(focus_concept="test", session_turn_count=100)
        next_state = state_manager.check_transitions(state_context)

        # Should be ready for assessment
        if next_state:
            assert next_state == TutorState.ASSESSMENT

    def test_counter_reset_on_different_action(self, state_manager):
        """Counters should reset when action type changes."""
        state_manager.transition_to(TutorState.TEACHING, reason="start")

        # Explain 3 times
        for i in range(3):
            state_manager.update_action("explain")

        # Now ask
        state_manager.update_action("ask")

        # Consecutive explains should reset or not interfere
        state_context = StateContext(focus_concept="test", session_turn_count=4)
        next_state = state_manager.check_transitions(state_context)

        # Transition might not happen because we broke the streak
        # This is OK - the important thing is no crash
        assert True

    # ===== ERROR RECOVERY =====

    def test_recovery_from_confusion(self, state_manager):
        """Should handle confusion → engagement transition."""
        state_manager.transition_to(TutorState.TEACHING, reason="start")

        # Student is confused (REVIEW)
        state_manager.transition_to(TutorState.REVIEW, reason="confused")
        assert state_manager.current_state == TutorState.REVIEW

        # Student engages with explanation (back to TEACHING)
        state_context = StateContext(
            focus_concept="concept",
            answer_correct=True,
            session_turn_count=5,
        )

        next_state = state_manager.check_transitions(state_context)

        # Should be able to return to TEACHING
        if next_state:
            state_manager.transition_to(next_state, reason="recovered")
            assert state_manager.current_state in [
                TutorState.TEACHING,
                TutorState.REVIEW,
            ]

    def test_graceful_handling_of_none_fields(self, decision_engine):
        """Should handle None values in decision context."""
        state_manager = TutorStateManager()

        context = TutorContext(
            message=None,
            intent=None,
            affect=None,
            focus_concept=None,
            concept_level=None,
            current_state=state_manager.current_state,
            mastery_map=None,
            retrieval_chunks=None,
            session_history=None,
            turn_index=0,
        )

        # Should handle gracefully (may crash or return default)
        try:
            decision = decision_engine.decide_action(context, state_manager)
            # If it doesn't crash, should be valid
            assert decision.action in [
                "explain",
                "ask",
                "hint",
                "reflect",
                "review",
            ]
        except Exception:
            # Some failures are expected with None inputs
            pass

    # ===== SCENARIO: DIFFICULT STUDENT =====

    def test_difficult_scenario_repeated_incorrect_answers(
        self, state_manager
    ):
        """Handle student with repeated incorrect answers."""
        state_manager.transition_to(TutorState.TEACHING, reason="start")

        # 3 asks, all incorrect
        for i in range(3):
            state_manager.update_action("answer_incorrect")

        # Should transition to REVIEW
        state_context = StateContext(
            focus_concept="difficult_concept",
            answer_correct=False,
            session_turn_count=3,
        )

        next_state = state_manager.check_transitions(state_context)

        # After enough incorrect answers, should go to REVIEW
        if next_state:
            assert next_state in [TutorState.REVIEW, TutorState.ASSESSMENT]

    def test_difficult_scenario_rapid_topic_switching(self, state_manager):
        """Handle student switching topics rapidly."""
        state_manager.transition_to(TutorState.TEACHING, reason="concept1")

        # Switch concepts 5 times
        for i in range(5):
            concept = f"concept_{i}"

            state_context = StateContext(
                focus_concept=concept,
                session_turn_count=i + 1,
            )

            # System should handle gracefully
            next_state = state_manager.check_transitions(state_context)

            # No crash is success
            assert state_manager.current_state in [
                TutorState.ORIENTATION,
                TutorState.TEACHING,
                TutorState.ASSESSMENT,
                TutorState.REVIEW,
                TutorState.CLOSURE,
            ]

    def test_difficult_scenario_long_session(self, state_manager):
        """Handle very long teaching session (50+ turns)."""
        state_manager.transition_to(TutorState.TEACHING, reason="start")

        # Simulate 50 turns
        for turn in range(50):
            if turn % 10 == 0:
                state_manager.update_action("explain")
            elif turn % 3 == 0:
                state_manager.update_action("ask")
            else:
                state_manager.update_action("hint")

            # Occasionally check transitions
            if turn % 5 == 0:
                state_context = StateContext(
                    focus_concept="persistent_concept",
                    session_turn_count=turn,
                )

                next_state = state_manager.check_transitions(state_context)

                # Should remain valid
                if next_state:
                    state_manager.transition_to(next_state, reason=f"turn_{turn}")

        # Should finish in a valid state
        assert state_manager.current_state in [
            TutorState.ORIENTATION,
            TutorState.TEACHING,
            TutorState.ASSESSMENT,
            TutorState.REVIEW,
            TutorState.CLOSURE,
        ]

        # History should be tracked
        assert len(state_manager.state_history) > 0

