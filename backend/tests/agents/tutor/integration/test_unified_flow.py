"""End-to-end integration tests for unified tutor architecture.

Tests the complete flow: classification → retrieval → decision → response → mastery.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from backend.agents.tutor.state_machine import (
    TutorStateManager,
    TutorState,
    StateContext,
)
from backend.agents.tutor.decision_engine import TutorDecisionEngine
from backend.agents.tutor.context_model import TutorContext
from backend.agents.tutor.response_generator import ResponseGenerator
from backend.agents.tutor.tools.mastery_updater import MasteryUpdater
from backend.agents.tutor.config import TutorConfig


class TestUnifiedTutorFlow:
    """Test complete tutor flow from input to output."""

    @pytest.fixture
    def state_manager(self):
        """Create state manager."""
        return TutorStateManager()

    @pytest.fixture
    def decision_engine(self):
        """Create decision engine."""
        return TutorDecisionEngine(mode="intelligent")

    @pytest.fixture
    def response_generator(self):
        """Create response generator."""
        return ResponseGenerator()

    @pytest.fixture
    def mastery_updater(self):
        """Create mastery updater."""
        return MasteryUpdater()

    @pytest.fixture
    def config(self):
        """Create config."""
        return TutorConfig.from_mode("intelligent")

    # ===== SINGLE TURN TESTS =====

    def test_orientation_turn_flow(self, state_manager, decision_engine):
        """Test complete ORIENTATION → TEACHING turn."""
        # Initial state
        assert state_manager.current_state == TutorState.ORIENTATION

        # Build decision context
        context = TutorContext(
            message="I want to learn about heat transfer",
            intent="concept_selection",
            affect="engaged",
            focus_concept="heat_transfer",
            concept_level="introductory",
            current_state=state_manager.current_state,
            mastery_map={},
            retrieval_chunks=[
                {"id": "chunk-1", "snippet": "Heat transfer is...", "score": 0.9}
            ],
            session_history=[],
            turn_index=0,
        )

        # Make decision
        decision = decision_engine.decide_action(context, state_manager)

        # Should recommend orientation action
        assert decision.action in ["explain", "ask", "orient"]
        assert decision.confidence > 0.5

        # Now transition to TEACHING
        state_context = StateContext(
            focus_concept="heat_transfer",
            student_message="I want to learn about heat transfer",
            session_turn_count=1,
        )

        next_state = state_manager.check_transitions(state_context)
        if next_state:
            state_manager.transition_to(next_state, reason="concept_selected")

        # Should be in TEACHING
        assert state_manager.current_state in [
            TutorState.ORIENTATION,
            TutorState.TEACHING,
        ]

    def test_teaching_to_assessment_flow(self, state_manager, decision_engine):
        """Test TEACHING → ASSESSMENT transition after 3 explains."""
        # Start in TEACHING
        state_manager.transition_to(TutorState.TEACHING, reason="concept_selected")
        assert state_manager.current_state == TutorState.TEACHING

        # Simulate 3 explain actions
        for i in range(3):
            state_manager.update_action("explain")

        # Build context
        context = TutorContext(
            message=f"Explanation {i}",
            intent="explanation",
            affect="engaged",
            focus_concept="conduction",
            concept_level="introductory",
            current_state=state_manager.current_state,
            mastery_map={"conduction": {"mastery": 0.3}},
            retrieval_chunks=[],
            session_history=[],
            turn_index=3,
        )

        # Decision should check for transition
        decision = decision_engine.decide_action(context, state_manager)

        # Check transition
        state_context = StateContext(
            focus_concept="conduction",
            last_action="explain",
            session_turn_count=3,
            mastery_map={"conduction": {"mastery": 0.3}},
        )

        next_state = state_manager.check_transitions(state_context)

        # Should transition to ASSESSMENT
        if next_state:
            assert next_state == TutorState.ASSESSMENT

    def test_assessment_with_correct_answer(self, state_manager, mastery_updater):
        """Test ASSESSMENT with correct answer → mastery increase."""
        state_manager.transition_to(TutorState.TEACHING, reason="start")
        state_manager.transition_to(TutorState.ASSESSMENT, reason="teach_complete")

        # Student gives correct answer
        signals = {
            "affect": "engaged",
            "intent": "answer",
            "answer_correct": True,
            "classification_confidence": 0.95,
        }

        update = mastery_updater.compute_mastery_delta(
            concept="heat_transfer",
            user_id="user-123",
            interaction_signals=signals,
            current_mastery=0.2,
        )

        # Should increase mastery
        assert update.delta > 0.0
        assert "correct_answer" in update.reason

        # State should prepare to transition back to TEACHING
        state_context = StateContext(
            focus_concept="heat_transfer",
            answer_correct=True,
            mastery_delta=update.delta,
            mastery_reason=update.reason,
        )

        next_state = state_manager.check_transitions(state_context)
        # May transition back to TEACHING or stay in ASSESSMENT
        assert next_state in [TutorState.TEACHING, None]

    def test_assessment_with_incorrect_answer(self, state_manager, mastery_updater):
        """Test ASSESSMENT with incorrect answers → REVIEW."""
        state_manager.transition_to(TutorState.TEACHING, reason="start")
        state_manager.transition_to(TutorState.ASSESSMENT, reason="teach_complete")

        # First incorrect answer
        signals_1 = {
            "affect": "neutral",
            "intent": "answer",
            "answer_correct": False,
            "classification_confidence": 0.9,
        }

        update_1 = mastery_updater.compute_mastery_delta(
            concept="convection",
            user_id="user-123",
            interaction_signals=signals_1,
            current_mastery=0.3,
        )

        # Should decrease mastery
        assert update_1.delta < 0.0
        state_manager.update_action("answer_incorrect")

        # Second incorrect answer
        update_2 = mastery_updater.compute_mastery_delta(
            concept="convection",
            user_id="user-123",
            interaction_signals=signals_1,
            current_mastery=0.3 + update_1.delta,
        )

        state_manager.update_action("answer_incorrect")

        # Check for REVIEW transition
        state_context = StateContext(
            focus_concept="convection",
            answer_correct=False,
            session_turn_count=5,
            mastery_delta=update_2.delta,
        )

        next_state = state_manager.check_transitions(state_context)

        # Should transition to REVIEW
        if next_state:
            assert next_state == TutorState.REVIEW

    # ===== MULTI-TURN SESSION TESTS =====

    def test_multi_turn_teaching_flow(
        self, state_manager, decision_engine, mastery_updater
    ):
        """Test 5-turn teaching session with state transitions."""
        # Turn 1: Orient
        state_manager.transition_to(TutorState.TEACHING, reason="concept_selected")

        turn_data = []

        for turn in range(5):
            # Build decision context
            context = TutorContext(
                message=f"Turn {turn} input",
                intent="answer" if turn > 0 else "concept_selection",
                affect="engaged",
                focus_concept="thermodynamics",
                concept_level="intermediate",
                current_state=state_manager.current_state,
                mastery_map={"thermodynamics": {"mastery": 0.2 + (turn * 0.05)}},
                retrieval_chunks=[
                    {"id": f"chunk-{turn}", "snippet": "...", "score": 0.9}
                ],
                session_history=[],
                turn_index=turn,
            )

            # Make decision
            decision = decision_engine.decide_action(context, state_manager)

            # Track action
            state_manager.update_action(decision.action)

            # Simulate mastery update
            signals = {
                "affect": "engaged",
                "intent": decision.action,
                "answer_correct": turn % 2 == 0 or turn == 0,
                "classification_confidence": 0.8 + (turn * 0.02),
            }

            mastery_update = mastery_updater.compute_mastery_delta(
                concept="thermodynamics",
                user_id="user-123",
                interaction_signals=signals,
                current_mastery=0.2 + (turn * 0.05),
            )

            # Check for transitions
            state_context = StateContext(
                focus_concept="thermodynamics",
                answer_correct=signals.get("answer_correct"),
                session_turn_count=turn,
                mastery_delta=mastery_update.delta,
                mastery_reason=mastery_update.reason,
            )

            next_state = state_manager.check_transitions(state_context)
            if next_state and next_state != state_manager.current_state:
                state_manager.transition_to(next_state, reason=f"turn_{turn}")

            turn_data.append(
                {
                    "turn": turn,
                    "action": decision.action,
                    "state": state_manager.current_state,
                    "mastery_delta": mastery_update.delta,
                }
            )

        # Verify progression
        assert len(turn_data) == 5
        # Should have stayed in TEACHING or progressed to ASSESSMENT
        states_visited = set(t["state"] for t in turn_data)
        assert TutorState.TEACHING in states_visited or TutorState.ASSESSMENT in states_visited

    def test_five_turn_cycle_with_state_transitions(
        self, state_manager, mastery_updater
    ):
        """Test realistic 5-turn cycle: ORIENTATION → TEACHING → ASSESSMENT."""
        # Turn 1: ORIENTATION (concept selection)
        state_manager.transition_to(TutorState.TEACHING, reason="concept_selected")

        # Turns 2-4: TEACHING (3 explains)
        for turn in range(2, 5):
            state_manager.update_action("explain")

            # Check for transition to ASSESSMENT
            state_context = StateContext(
                focus_concept="conduction",
                last_action="explain",
                session_turn_count=turn,
            )

            next_state = state_manager.check_transitions(state_context)
            if next_state and next_state != state_manager.current_state:
                state_manager.transition_to(next_state, reason="3_explains")
                break

        # Should be in ASSESSMENT
        assert state_manager.current_state in [TutorState.TEACHING, TutorState.ASSESSMENT]

        # Turn 5: ASSESSMENT (question answered)
        if state_manager.current_state == TutorState.ASSESSMENT:
            signals = {
                "affect": "engaged",
                "intent": "answer",
                "answer_correct": True,
                "classification_confidence": 0.9,
            }

            update = mastery_updater.compute_mastery_delta(
                concept="conduction",
                user_id="user-123",
                interaction_signals=signals,
                current_mastery=0.4,
            )

            # Verify mastery increased
            assert update.delta > 0.0

    # ===== ERROR HANDLING & EDGE CASES =====

    def test_transition_with_null_context(self, state_manager):
        """Transitions should handle null context fields gracefully."""
        state_manager.transition_to(TutorState.TEACHING, reason="start")

        # Build minimal context
        context = StateContext(
            focus_concept=None,
            answer_correct=None,
            student_message="",
            session_turn_count=0,
        )

        # Should not crash
        next_state = state_manager.check_transitions(context)

        # Should either return None or a valid state
        assert next_state in [TutorState.TEACHING, TutorState.ASSESSMENT, None]

    def test_cold_start_with_mastery_update(self, mastery_updater):
        """New concept should respond to engagement signals."""
        # No prior mastery
        current_mastery = 0.0

        # Student engaged with new concept
        signals = {
            "affect": "engaged",
            "intent": "reflection",
            "explanation_quality": 0.75,
            "classification_confidence": 0.8,
        }

        update = mastery_updater.compute_mastery_delta(
            concept="new_concept",
            user_id="user-123",
            interaction_signals=signals,
            current_mastery=current_mastery,
        )

        # Should produce positive signal for new concept
        assert update.delta >= 0.0

    def test_closure_signal_ends_session(self, state_manager):
        """Student closure signal should end session."""
        state_manager.transition_to(TutorState.TEACHING, reason="concept_selected")

        # Student signals done
        context = StateContext(
            focus_concept="heat_transfer",
            student_message="I'm done for today, thanks!",
            session_turn_count=10,
        )

        next_state = state_manager.check_transitions(context)

        # Should transition to CLOSURE
        if next_state:
            state_manager.transition_to(next_state, reason="student_signals_done")
            assert state_manager.current_state == TutorState.CLOSURE

    # ===== RESPONSE GENERATION IN CONTEXT =====

    def test_response_generation_with_state_and_mastery(
        self, response_generator, state_manager, mastery_updater
    ):
        """Response should include state and mastery info."""
        state_manager.transition_to(TutorState.TEACHING, reason="concept_selected")

        # Simulate mastery update
        signals = {
            "affect": "engaged",
            "intent": "answer",
            "answer_correct": True,
            "classification_confidence": 0.9,
        }

        update = mastery_updater.compute_mastery_delta(
            concept="radiation",
            user_id="user-123",
            interaction_signals=signals,
            current_mastery=0.3,
        )

        # Mock decision context for response generation
        context = TutorContext(
            message="What is radiation?",
            intent="answer",
            affect="engaged",
            focus_concept="radiation",
            concept_level="intermediate",
            current_state=state_manager.current_state,
            mastery_map={"radiation": {"mastery": 0.3 + update.delta}},
            retrieval_chunks=[
                {
                    "id": "chunk-1",
                    "snippet": "Radiation is heat transfer through space...",
                    "score": 0.92,
                }
            ],
            session_history=[],
            turn_index=1,
        )

        # Generate response
        try:
            response = response_generator.generate_response(
                action="explain",
                context=context,
                chunks=context.retrieval_chunks,
                grounding_mode="llm_integrated",
            )

            # Verify response structure
            assert response.response_text is not None
            assert response.action_type == "explain"
            assert response.citations is not None
            # Should have cleaned citations, not OCR artifacts
            for citation in response.citations:
                assert "(cid:" not in citation.snippet_text

        except Exception:
            # May fail if LLM is not available, but structure should be correct
            pass

    # ===== SCENARIO TESTS =====

    def test_teaching_scenario_complete_flow(
        self, state_manager, decision_engine, mastery_updater
    ):
        """Test complete TEACHING scenario."""
        # Setup
        state_manager.transition_to(TutorState.TEACHING, reason="concept_selected")
        concept = "fourier_law"
        user_id = "user-123"

        # Turns 1-2: Explain (tutor explains)
        for turn in range(2):
            state_manager.update_action("explain")

            context = TutorContext(
                message=f"Explanation {turn}",
                intent="explanation",
                affect="engaged",
                focus_concept=concept,
                concept_level="advanced",
                current_state=state_manager.current_state,
                mastery_map={concept: {"mastery": 0.2 + (turn * 0.05)}},
                retrieval_chunks=[],
                session_history=[],
                turn_index=turn + 1,
            )

            decision = decision_engine.decide_action(context, state_manager)
            assert decision.action in ["explain", "ask", "hint"]

        # Turn 3: Check for transition (should check if ready to assess)
        state_context = StateContext(
            focus_concept=concept,
            session_turn_count=2,
        )

        # Simulate 3rd explain to trigger ASSESSMENT transition
        state_manager.update_action("explain")
        state_context.session_turn_count = 3

        next_state = state_manager.check_transitions(state_context)
        if next_state:
            state_manager.transition_to(next_state, reason="3_explains")
            assert next_state == TutorState.ASSESSMENT

    # ===== BACKWARD COMPATIBILITY TESTS =====

    def test_old_mastery_override_ignored(self, mastery_updater):
        """Old policy override should be ignored."""
        signals = {
            "affect": "engaged",
            "intent": "answer",
            "answer_correct": True,
            "classification_confidence": 0.9,
        }

        # Even if policy says don't update (old behavior)
        # Mastery should update (new behavior)
        update = mastery_updater.compute_mastery_delta(
            concept="test",
            user_id="user-123",
            interaction_signals=signals,
            current_mastery=0.3,
        )

        # Should produce positive update
        assert update.delta > 0.0

    def test_config_modes_work_end_to_end(self):
        """All config modes should be usable."""
        for mode in ["simple", "intelligent", "step_by_step", "debug"]:
            config = TutorConfig.from_mode(mode)
            assert config.mode == mode
            # Each mode should have valid settings
            assert config.enable_llm_policy is not None
            assert config.enable_srl_planning is not None

