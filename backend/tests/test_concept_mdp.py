import sys
import os
import unittest

# Mock external dependencies that backend imports may expect
sys.modules["psycopg2"] = type("Mock", (), {})()
mock_psycopg2_extras = type("MockPsycopg2Extras", (), {"Json": object})()
sys.modules["psycopg2.extras"] = mock_psycopg2_extras

# Stub core and core.db with a dummy get_db_conn so imports in agents.* succeed
mock_core = type("MockCore", (), {})()
mock_core_db = type("MockDB", (), {"get_db_conn": lambda *args, **kwargs: None})()
sys.modules.setdefault("core", mock_core)
sys.modules.setdefault("core.db", mock_core_db)

# Add backend to path so we can import agents.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from agents.tutor.mdp.concept import (
    ConceptState,
    ConceptObservation,
    ConceptMDPAction,
    apply_concept_transition,
    build_concept_observation,
)
from agents.tutor.mdp.actions import map_step_controls_to_concept_action


class TestConceptMDPTransitions(unittest.TestCase):
    def _make_state(
        self,
        *,
        mastery: float = 0.0,
        target_mastery: float | None = None,
        step_count: int = 0,
        quiz_correct: int = 0,
        quiz_wrong: int = 0,
        last_control_type: str | None = None,
    ) -> ConceptState:
        return ConceptState(
            episode_id="ce-test",
            session_id="s1",
            user_id="u1",
            concept_id="c1",
            target_mastery=target_mastery,
            mastery=mastery,
            mastery_start=mastery,
            plan_index=0,
            quiz_phase="",
            quiz_question_index=0,
            quiz_max_questions=0,
            quiz_correct=quiz_correct,
            quiz_wrong=quiz_wrong,
            step_count=step_count,
            last_control_type=last_control_type,
            last_mdp_action=None,
        )

    def test_basic_step_updates_state_and_observation(self):
        state = self._make_state(mastery=0.1)
        outcome = apply_concept_transition(
            prev_state=state,
            mdp_action=ConceptMDPAction.FOLLOW_PLAN_STEP,
            mastery_delta=0.0,
            quiz_delta=None,
            mcq_outcome=None,
            control_type="continue",
        )

        self.assertIsInstance(outcome.state, ConceptState)
        self.assertIsInstance(outcome.observation, ConceptObservation)
        self.assertEqual(outcome.state.step_count, 1)
        self.assertEqual(outcome.state.last_mdp_action, ConceptMDPAction.FOLLOW_PLAN_STEP)
        # Control should be normalized and threaded into observation
        self.assertEqual(outcome.state.last_control_type, "continue")
        self.assertEqual(outcome.observation.last_control_type, "continue")
        self.assertFalse(outcome.terminated)
        self.assertIsNone(outcome.termination_reason)

    def test_quiz_counters_and_reward(self):
        state = self._make_state(mastery=0.2)
        outcome = apply_concept_transition(
            prev_state=state,
            mdp_action=ConceptMDPAction.FOLLOW_PLAN_STEP,
            mastery_delta=0.1,
            quiz_delta=1.0,
            mcq_outcome={"answer_correct": True},
            control_type="mcq_answer",
        )

        self.assertEqual(outcome.state.quiz_correct, 1)
        self.assertEqual(outcome.state.quiz_wrong, 0)
        self.assertIn("mastery_delta", outcome.reward)
        self.assertIn("quiz_delta", outcome.reward)

    def test_termination_by_mastery_reached(self):
        state = self._make_state(mastery=0.7, target_mastery=0.8)
        outcome = apply_concept_transition(
            prev_state=state,
            mdp_action=ConceptMDPAction.FOLLOW_PLAN_STEP,
            mastery_delta=0.1,
            quiz_delta=None,
            mcq_outcome=None,
            control_type="continue",
            post_mastery=0.8,
        )

        self.assertTrue(outcome.terminated)
        self.assertEqual(outcome.termination_reason, "mastery_reached")

    def test_termination_by_max_steps(self):
        old = os.environ.get("TUTOR_STEP_SRL_MAX_STEPS")
        os.environ["TUTOR_STEP_SRL_MAX_STEPS"] = "1"
        try:
            state = self._make_state(mastery=0.1, step_count=0)
            outcome = apply_concept_transition(
                prev_state=state,
                mdp_action=ConceptMDPAction.FOLLOW_PLAN_STEP,
                mastery_delta=0.0,
                quiz_delta=None,
                mcq_outcome=None,
                control_type="continue",
            )
        finally:
            if old is None:
                os.environ.pop("TUTOR_STEP_SRL_MAX_STEPS", None)
            else:
                os.environ["TUTOR_STEP_SRL_MAX_STEPS"] = old

        self.assertTrue(outcome.terminated)
        self.assertEqual(outcome.termination_reason, "max_steps_reached")

    def test_skip_and_session_end_terminations(self):
        # Skip next concept via override
        state = self._make_state(mastery=0.1)
        skip_outcome = apply_concept_transition(
            prev_state=state,
            mdp_action=ConceptMDPAction.ADVANCE_CONCEPT,
            mastery_delta=0.0,
            quiz_delta=None,
            mcq_outcome=None,
            control_type="skip_to_next_concept",
            requested_override_type="step_next_concept",
        )
        self.assertTrue(skip_outcome.terminated)
        self.assertEqual(skip_outcome.termination_reason, "skip_next_concept")

        # Session end via override and step_control_type
        state2 = self._make_state(mastery=0.1)
        end_outcome = apply_concept_transition(
            prev_state=state2,
            mdp_action=ConceptMDPAction.TERMINATE_CONCEPT,
            mastery_delta=0.0,
            quiz_delta=None,
            mcq_outcome=None,
            control_type="end",
            requested_override_type="session_end",
            step_control_type="end_session",
        )
        self.assertTrue(end_outcome.terminated)
        self.assertEqual(end_outcome.termination_reason, "session_end")

    def test_build_concept_observation_uses_state_fields(self):
        state = self._make_state(mastery=0.5, target_mastery=0.8, quiz_correct=2, quiz_wrong=1)
        obs = build_concept_observation(
            state=state,
            last_intent="question",
            last_affect="confused",
            last_action_type="explain",
            last_control_type="continue",
        )

        self.assertEqual(obs.concept_id, state.concept_id)
        self.assertEqual(obs.mastery, state.mastery)
        self.assertEqual(obs.target_mastery, state.target_mastery)
        self.assertEqual(obs.quiz_correct, state.quiz_correct)
        self.assertEqual(obs.quiz_wrong, state.quiz_wrong)
        self.assertEqual(obs.last_intent, "question")
        self.assertEqual(obs.last_affect, "confused")
        self.assertEqual(obs.last_action_type, "explain")
        self.assertEqual(obs.last_control_type, "continue")

    def test_map_step_controls_to_concept_action(self):
        step_control = type("StepControlMock", (), {"type": "continue"})()
        action_from_control = map_step_controls_to_concept_action(
            step_control_obj=step_control,
            requested_override_type=None,
            control_type_label=None,
        )
        self.assertEqual(action_from_control, ConceptMDPAction.FOLLOW_PLAN_STEP)

        action_from_override = map_step_controls_to_concept_action(
            step_control_obj=None,
            requested_override_type="step_skip_to_assessment",
            control_type_label=None,
        )
        self.assertEqual(action_from_override, ConceptMDPAction.JUMP_TO_ASSESSMENT)

        action_from_label = map_step_controls_to_concept_action(
            step_control_obj=None,
            requested_override_type=None,
            control_type_label="skip_to_next_concept",
        )
        self.assertEqual(action_from_label, ConceptMDPAction.ADVANCE_CONCEPT)


if __name__ == "__main__":
    unittest.main()
