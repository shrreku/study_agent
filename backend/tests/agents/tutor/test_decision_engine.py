"""
Tests for TutorDecisionEngine decision-making logic.
"""

import pytest
from backend.agents.tutor.decision_engine import (
    TutorDecisionEngine,
    ActionDecision,
)
from backend.agents.tutor.context_model import TutorContext
from backend.agents.tutor.state_machine import (
    TutorStateManager,
    TutorState,
)


class TestDecisionEngineInitialization:
    """Test engine initialization with different modes."""

    def test_init_intelligent_mode(self):
        """Test initialization in intelligent mode."""
        engine = TutorDecisionEngine(mode="intelligent")
        assert engine.mode == "intelligent"
        assert engine.enable_llm_policy is True
        assert engine.enable_srl_planning is True

    def test_init_simple_mode(self):
        """Test initialization in simple mode."""
        engine = TutorDecisionEngine(mode="simple")
        assert engine.mode == "simple"
        assert engine.enable_llm_policy is False
        assert engine.enable_srl_planning is False

    def test_init_step_by_step_mode(self):
        """Test initialization in step-by-step mode."""
        engine = TutorDecisionEngine(mode="step_by_step")
        assert engine.mode == "step_by_step"
        # SRL planning should be enabled
        assert engine.enable_srl_planning is True

    def test_init_case_insensitive(self):
        """Test mode strings are case-insensitive."""
        engine = TutorDecisionEngine(mode="INTELLIGENT")
        assert engine.mode == "intelligent"


class TestStateMandate:
    """Test state-mandated decisions."""

    def test_orientation_mandates_orient_action(self):
        """Test ORIENTATION state mandates orient action."""
        engine = TutorDecisionEngine(mode="simple")
        state_manager = TutorStateManager(initial_state=TutorState.ORIENTATION)

        context = TutorContext(
            message="Hi",
            user_id="user1",
            session_id="sess1",
            turn_index=1,
            intent="greeting",
            affect="neutral",
            inferred_concept=None,
            focus_concept=None,
            concept_level="foundational",
            current_state=TutorState.ORIENTATION,
        )

        decision = engine.decide_action(context, state_manager)

        assert decision.action == "orient"
        assert decision.confidence == 0.95

    def test_closure_mandates_preview_action(self):
        """Test CLOSURE state mandates preview action."""
        engine = TutorDecisionEngine(mode="simple")
        state_manager = TutorStateManager(initial_state=TutorState.CLOSURE)

        context = TutorContext(
            message="Done",
            user_id="user1",
            session_id="sess1",
            turn_index=10,
            intent="closing",
            affect="neutral",
            inferred_concept=None,
            focus_concept="Heat Transfer",
            concept_level="intermediate",
            current_state=TutorState.CLOSURE,
        )

        decision = engine.decide_action(context, state_manager)

        assert decision.action == "preview"
        assert decision.confidence == 0.95


class TestSafetyGates:
    """Test safety gate decisions."""

    def test_confused_student_gets_hint(self):
        """Test confused student receives hint."""
        engine = TutorDecisionEngine(mode="simple")
        state_manager = TutorStateManager(initial_state=TutorState.TEACHING)

        context = TutorContext(
            message="I'm confused",
            user_id="user1",
            session_id="sess1",
            turn_index=5,
            intent="question",
            affect="confused",
            inferred_concept=None,
            focus_concept="Heat Transfer",
            concept_level="intermediate",
            current_state=TutorState.TEACHING,
        )

        decision = engine.decide_action(context, state_manager)

        assert decision.action == "hint"
        assert "confused" in decision.rationale.lower()

    def test_frustrated_student_gets_reflection(self):
        """Test frustrated student gets reflection."""
        engine = TutorDecisionEngine(mode="simple")
        state_manager = TutorStateManager(initial_state=TutorState.TEACHING)

        context = TutorContext(
            message="This is too hard!",
            user_id="user1",
            session_id="sess1",
            turn_index=5,
            intent="complaint",
            affect="frustrated",
            inferred_concept=None,
            focus_concept="Heat Transfer",
            concept_level="advanced",
            current_state=TutorState.TEACHING,
        )

        decision = engine.decide_action(context, state_manager)

        assert decision.action == "reflect"
        assert "frustrated" in decision.rationale.lower()


class TestHeuristicDecisions:
    """Test heuristic-based decisions."""

    def test_teaching_alternates_explain_ask(self):
        """Test TEACHING state alternates between explain and ask."""
        engine = TutorDecisionEngine(mode="simple")
        state_manager = TutorStateManager(initial_state=TutorState.TEACHING)

        context = TutorContext(
            message="OK",
            user_id="user1",
            session_id="sess1",
            turn_index=3,
            intent="acknowledgment",
            affect="neutral",
            inferred_concept=None,
            focus_concept="Heat Transfer",
            concept_level="intermediate",
            current_state=TutorState.TEACHING,
        )

        # After 2 explains, should ask
        state_manager.state_counters["consecutive_explains"] = 2
        decision = engine.decide_action(context, state_manager)
        assert decision.action == "ask"

        # With 0 explains, should explain
        state_manager.state_counters["consecutive_explains"] = 0
        decision = engine.decide_action(context, state_manager)
        assert decision.action == "explain"

    def test_assessment_responds_to_answer(self):
        """Test ASSESSMENT state reacts to student answer."""
        engine = TutorDecisionEngine(mode="simple")
        state_manager = TutorStateManager(initial_state=TutorState.ASSESSMENT)

        # When student provides answer
        context = TutorContext(
            message="I think it's conduction",
            user_id="user1",
            session_id="sess1",
            turn_index=5,
            intent="answer",
            affect="neutral",
            inferred_concept=None,
            focus_concept="Heat Transfer",
            concept_level="intermediate",
            current_state=TutorState.ASSESSMENT,
        )

        decision = engine.decide_action(context, state_manager)
        assert decision.action == "reflect"

        # When student hasn't answered yet
        context.intent = "question"
        decision = engine.decide_action(context, state_manager)
        assert decision.action == "ask"

    def test_review_explains_prerequisites(self):
        """Test REVIEW state explains prerequisites."""
        engine = TutorDecisionEngine(mode="simple")
        state_manager = TutorStateManager(initial_state=TutorState.REVIEW)

        context = TutorContext(
            message="OK",
            user_id="user1",
            session_id="sess1",
            turn_index=6,
            intent="acknowledgment",
            affect="neutral",
            inferred_concept=None,
            focus_concept="Convection",
            concept_level="intermediate",
            current_state=TutorState.REVIEW,
            prerequisites=["Conduction", "Heat Transfer"],
        )

        decision = engine.decide_action(context, state_manager)

        assert decision.action == "explain"
        assert decision.retrieval_query in ["Conduction", "Heat Transfer"]


class TestModeRouting:
    """Test different decision modes."""

    def test_simple_mode_uses_heuristics_only(self):
        """Test simple mode uses only heuristics."""
        engine = TutorDecisionEngine(mode="simple")
        assert engine.enable_llm_policy is False
        assert engine.enable_srl_planning is False

        state_manager = TutorStateManager(initial_state=TutorState.TEACHING)
        context = TutorContext(
            message="Continue",
            user_id="user1",
            session_id="sess1",
            turn_index=3,
            intent="continuation",
            affect="neutral",
            inferred_concept=None,
            focus_concept="Heat Transfer",
            concept_level="intermediate",
            current_state=TutorState.TEACHING,
        )

        decision = engine.decide_action(context, state_manager)
        assert decision is not None
        assert decision.action in ["explain", "ask", "hint"]

    def test_intelligent_mode_available(self):
        """Test intelligent mode initializes with policy."""
        engine = TutorDecisionEngine(mode="intelligent")
        assert engine.enable_llm_policy is True
        assert engine.enable_srl_planning is True

    def test_step_by_step_mode_has_planning(self):
        """Test step-by-step mode has planning enabled."""
        engine = TutorDecisionEngine(mode="step_by_step")
        assert engine.enable_srl_planning is True


class TestTutorContext:
    """Test TutorContext dataclass."""

    def test_context_creation(self):
        """Test creating a TutorContext."""
        context = TutorContext(
            message="Test message",
            user_id="user1",
            session_id="sess1",
            turn_index=5,
            intent="answer",
            affect="neutral",
            inferred_concept="Heat Transfer",
            focus_concept="Conduction",
            concept_level="intermediate",
            current_state=TutorState.TEACHING,
        )

        assert context.message == "Test message"
        assert context.user_id == "user1"
        assert context.current_state == TutorState.TEACHING

    def test_context_with_defaults(self):
        """Test TutorContext with default values."""
        context = TutorContext(
            message="Hi",
            user_id="user1",
            session_id="sess1",
            turn_index=1,
            intent="greeting",
            affect="neutral",
            inferred_concept=None,
            focus_concept=None,
            concept_level="foundational",
            current_state=TutorState.ORIENTATION,
        )

        assert context.mastery_map == {}
        assert context.prerequisites == []
        assert context.retrieval_chunks == []


class TestActionDecision:
    """Test ActionDecision dataclass."""

    def test_action_decision_creation(self):
        """Test creating an ActionDecision."""
        decision = ActionDecision(
            action="explain",
            rationale="Explaining concept",
            retrieval_query="Heat Transfer",
            grounding_mode="llm_integrated",
            confidence=0.85,
        )

        assert decision.action == "explain"
        assert decision.confidence == 0.85

    def test_action_decision_with_state_after(self):
        """Test ActionDecision with expected state transition."""
        decision = ActionDecision(
            action="ask",
            rationale="Testing understanding",
            retrieval_query="Heat Transfer",
            grounding_mode="llm_integrated",
            confidence=0.8,
            state_after=TutorState.ASSESSMENT,
        )

        assert decision.state_after == TutorState.ASSESSMENT


class TestDecisionPriority:
    """Test decision priority ordering."""

    def test_state_mandate_beats_safety_gates(self):
        """Test state mandate has highest priority."""
        engine = TutorDecisionEngine(mode="simple")
        state_manager = TutorStateManager(initial_state=TutorState.ORIENTATION)

        # Even if confused, ORIENTATION state mandates orient
        context = TutorContext(
            message="Hi",
            user_id="user1",
            session_id="sess1",
            turn_index=1,
            intent="greeting",
            affect="confused",  # Normally would trigger hint
            inferred_concept=None,
            focus_concept=None,
            concept_level="foundational",
            current_state=TutorState.ORIENTATION,  # But state mandate wins
        )

        decision = engine.decide_action(context, state_manager)
        assert decision.action == "orient"  # Not "hint"

    def test_safety_gates_beat_heuristics(self):
        """Test safety gates beat normal heuristics."""
        engine = TutorDecisionEngine(mode="simple")
        state_manager = TutorStateManager(initial_state=TutorState.TEACHING)
        state_manager.state_counters["consecutive_explains"] = 0

        # Heuristic would suggest explain, but confused beats it
        context = TutorContext(
            message="I'm confused",
            user_id="user1",
            session_id="sess1",
            turn_index=3,
            intent="question",
            affect="confused",  # Safety gate!
            inferred_concept=None,
            focus_concept="Heat Transfer",
            concept_level="intermediate",
            current_state=TutorState.TEACHING,
        )

        decision = engine.decide_action(context, state_manager)
        assert decision.action == "hint"  # Not "explain" from heuristics


class TestConfidenceScores:
    """Test confidence scores for decisions."""

    def test_state_mandate_high_confidence(self):
        """Test state-mandated decisions have high confidence."""
        engine = TutorDecisionEngine(mode="simple")
        state_manager = TutorStateManager(initial_state=TutorState.ORIENTATION)

        context = TutorContext(
            message="Hi",
            user_id="user1",
            session_id="sess1",
            turn_index=1,
            intent="greeting",
            affect="neutral",
            inferred_concept=None,
            focus_concept=None,
            concept_level="foundational",
            current_state=TutorState.ORIENTATION,
        )

        decision = engine.decide_action(context, state_manager)
        assert decision.confidence == 0.95

    def test_heuristic_moderate_confidence(self):
        """Test heuristic decisions have moderate confidence."""
        engine = TutorDecisionEngine(mode="simple")
        state_manager = TutorStateManager(initial_state=TutorState.TEACHING)

        context = TutorContext(
            message="OK",
            user_id="user1",
            session_id="sess1",
            turn_index=3,
            intent="acknowledgment",
            affect="neutral",
            inferred_concept=None,
            focus_concept="Heat Transfer",
            concept_level="intermediate",
            current_state=TutorState.TEACHING,
        )

        decision = engine.decide_action(context, state_manager)
        assert 0.6 <= decision.confidence <= 0.75


class TestGroundingModes:
    """Test response grounding modes."""

    def test_llm_integrated_grounding(self):
        """Test llm_integrated grounding mode."""
        engine = TutorDecisionEngine(mode="simple")
        state_manager = TutorStateManager(initial_state=TutorState.TEACHING)

        context = TutorContext(
            message="Continue",
            user_id="user1",
            session_id="sess1",
            turn_index=3,
            intent="continuation",
            affect="neutral",
            inferred_concept=None,
            focus_concept="Heat Transfer",
            concept_level="intermediate",
            current_state=TutorState.TEACHING,
        )

        decision = engine.decide_action(context, state_manager)
        assert decision.grounding_mode == "llm_integrated"

    def test_no_grounding_for_reflection(self):
        """Test no grounding for reflection actions."""
        engine = TutorDecisionEngine(mode="simple")
        state_manager = TutorStateManager(initial_state=TutorState.TEACHING)

        context = TutorContext(
            message="This is frustrating",
            user_id="user1",
            session_id="sess1",
            turn_index=5,
            intent="complaint",
            affect="frustrated",
            inferred_concept=None,
            focus_concept="Heat Transfer",
            concept_level="advanced",
            current_state=TutorState.TEACHING,
        )

        decision = engine.decide_action(context, state_manager)
        assert decision.action == "reflect"
        assert decision.grounding_mode == "none"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
