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

from agents.tutor.mdp.session import SessionObservation, SessionMDPAction
from agents.tutor.mdp.concept import ConceptObservation
from agents.tutor.mdp.actions import ConceptMDPAction
from agents.tutor.mdp.policy import (
    DefaultSessionPolicy,
    SRLConceptPolicy,
)
from agents.tutor.mdp.plans import (
    SessionPlan,
    SessionPlanEntry,
    ConceptPlan,
    ConceptPlanStep,
)


class TestSessionPolicy(unittest.TestCase):
    def _make_observation(self, **overrides) -> SessionObservation:
        base = dict(
            strategy="learning_path",
            current_index=0,
            total_concepts=3,
            position_fraction=0.0,
            current_concept_id="c0",
            current_mastery=None,
            target_mastery=None,
            avg_mastery=None,
            min_mastery=None,
            remaining_low_mastery_count=0,
            session_step_count=0,
            concept_episodes_completed=0,
            time_used_steps=0,
            last_session_action=None,
            last_concept_termination_reason=None,
        )
        base.update(overrides)
        return SessionObservation(**base)

    def test_replan_when_no_session_plan_entries(self):
        policy = DefaultSessionPolicy()
        obs = self._make_observation()
        action = policy.decide(observation=obs, session_plan=None)
        self.assertEqual(action, SessionMDPAction.REPLAN_SESSION)

    def test_follow_plan_when_no_termination(self):
        policy = DefaultSessionPolicy()
        obs = self._make_observation()
        plan = SessionPlan(
            strategy="learning_path",
            entries=[SessionPlanEntry(concept_id="c0")],
            plan_id="p1",
        )
        action = policy.decide(observation=obs, session_plan=plan)
        self.assertEqual(action, SessionMDPAction.FOLLOW_PLAN_CONCEPT)

    def test_advance_when_concept_completed(self):
        policy = DefaultSessionPolicy()
        obs = self._make_observation(last_concept_termination_reason="mastery_reached")
        plan = SessionPlan(
            strategy="learning_path",
            entries=[SessionPlanEntry(concept_id="c0")],
            plan_id="p1",
        )
        action = policy.decide(observation=obs, session_plan=plan)
        self.assertEqual(action, SessionMDPAction.ADVANCE_IN_PLAN)

    def test_terminate_session_on_session_end_reason(self):
        policy = DefaultSessionPolicy()
        obs = self._make_observation(last_concept_termination_reason="session_end")
        plan = SessionPlan(
            strategy="learning_path",
            entries=[SessionPlanEntry(concept_id="c0")],
            plan_id="p1",
        )
        action = policy.decide(observation=obs, session_plan=plan)
        self.assertEqual(action, SessionMDPAction.TERMINATE_SESSION)


class TestConceptPolicy(unittest.TestCase):
    def _make_observation(self, **overrides) -> ConceptObservation:
        base = dict(
            concept_id="c1",
            mastery=0.5,
            target_mastery=0.8,
            plan_index=0,
            quiz_phase="",
            quiz_question_index=0,
            quiz_max_questions=0,
            quiz_correct=0,
            quiz_wrong=0,
            last_intent="question",
            last_affect="neutral",
            last_action_type="explain",
            last_control_type=None,
        )
        base.update(overrides)
        return ConceptObservation(**base)

    def test_follow_plan_step_by_default(self):
        policy = SRLConceptPolicy()
        obs = self._make_observation()
        action = policy.decide(observation=obs, concept_plan=None)
        self.assertEqual(action, ConceptMDPAction.REPLAN_CONCEPT)

        # With a non-empty concept plan, we should follow the plan
        cplan = ConceptPlan(
            plan_id="cp1",
            concept_id="c1",
            steps=[
                ConceptPlanStep(
                    step_id="s1",
                    step_type="EXPLAIN",
                    instruction="Explain the basic idea",
                )
            ],
            created_at_step=0,
            source="test",
        )
        action2 = policy.decide(observation=obs, concept_plan=cplan)
        self.assertEqual(action2, ConceptMDPAction.FOLLOW_PLAN_STEP)

    def test_controls_override_plan(self):
        policy = SRLConceptPolicy()
        obs_skip_quiz = self._make_observation(last_control_type="skip_to_quiz")
        action_quiz = policy.decide(observation=obs_skip_quiz, concept_plan=None)
        self.assertEqual(action_quiz, ConceptMDPAction.JUMP_TO_ASSESSMENT)

        obs_skip_next = self._make_observation(last_control_type="skip_to_next_concept")
        action_next = policy.decide(observation=obs_skip_next, concept_plan=None)
        self.assertEqual(action_next, ConceptMDPAction.ADVANCE_CONCEPT)

        obs_end = self._make_observation(last_control_type="end_session")
        action_end = policy.decide(observation=obs_end, concept_plan=None)
        self.assertEqual(action_end, ConceptMDPAction.TERMINATE_CONCEPT)


if __name__ == "__main__":
    unittest.main()
