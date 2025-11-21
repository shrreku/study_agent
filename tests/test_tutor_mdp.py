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

from agents.tutor.mdp.tutor import (  # type: ignore
    TutorStepState,
    TutorStepObservation,
    TutorStepMDPAction,
    build_tutor_step_state,
    build_tutor_step_observation,
    apply_tutor_step_transition,
)
from agents.tutor.mdp.actions import ConceptMDPAction  # type: ignore
from agents.tutor.mdp.plans import ConceptPlan, ConceptPlanStep  # type: ignore
from agents.tutor.mdp.policy import DefaultTutorStepPolicy  # type: ignore
from agents.tutor.state import TutorSessionPolicy  # type: ignore


class TestTutorMDP(unittest.TestCase):
    def _make_concept_plan(self, n_steps: int = 2) -> ConceptPlan:
        steps = [
            ConceptPlanStep(
                step_id=f"s{i}",
                step_type="EXPLAIN",
                subgoal=None,
                instruction="do it",
                tool_call=None,
                params={},
                evaluation_type=None,
                expected_duration_steps=1,
            )
            for i in range(n_steps)
        ]
        return ConceptPlan(
            plan_id="cp-test",
            concept_id="c1",
            steps=steps,
            created_at_step=0,
            source="test",
            initial_mastery=None,
            target_mastery=None,
        )

    def test_build_tutor_state_and_observation(self) -> None:
        policy = TutorSessionPolicy()
        policy.srl_plan_step_index = 1
        plan = self._make_concept_plan(n_steps=3)

        state = build_tutor_step_state(
            policy_state=policy,
            concept_plan=plan,
            concept_action=ConceptMDPAction.FOLLOW_PLAN_STEP,
            session_id="s1",
            user_id="u1",
            concept_id="c1",
            last_intent="question",
            last_affect="neutral",
            last_control_type="continue",
        )

        self.assertIsInstance(state, TutorStepState)
        self.assertEqual(state.session_id, "s1")
        self.assertEqual(state.user_id, "u1")
        self.assertEqual(state.concept_id, "c1")
        self.assertEqual(state.plan_length, 3)
        self.assertEqual(state.plan_index, 1)

        obs = build_tutor_step_observation(state=state)
        self.assertIsInstance(obs, TutorStepObservation)
        self.assertFalse(obs.at_end_of_plan)
        self.assertEqual(obs.plan_index, 1)
        self.assertEqual(obs.plan_length, 3)
        self.assertEqual(obs.last_concept_action, ConceptMDPAction.FOLLOW_PLAN_STEP.value)

    def test_default_policy_requests_replan_when_no_plan(self) -> None:
        policy = DefaultTutorStepPolicy()
        empty_state = TutorStepState(
            episode_id="ce-test",
            session_id="s1",
            user_id="u1",
            concept_id="c1",
            plan_id="cp-test",
            plan_index=0,
            plan_length=0,
            last_concept_action=None,
            last_intent="question",
            last_affect="neutral",
            last_control_type=None,
            last_step_type=None,
            last_step_id=None,
            step_count=0,
            replan_count=0,
        )
        obs = build_tutor_step_observation(state=empty_state)

        act = policy.decide(
            observation=obs,
            concept_action=ConceptMDPAction.FOLLOW_PLAN_STEP,
        )
        self.assertEqual(act, TutorStepMDPAction.REPLAN_CONCEPT)

    def test_default_policy_honors_concept_replan(self) -> None:
        policy = DefaultTutorStepPolicy()
        state = TutorStepState(
            episode_id="ce-test",
            session_id="s1",
            user_id="u1",
            concept_id="c1",
            plan_id="cp-test",
            plan_index=0,
            plan_length=2,
            last_concept_action=None,
            last_intent="question",
            last_affect="neutral",
            last_control_type=None,
            last_step_type=None,
            last_step_id=None,
            step_count=0,
            replan_count=0,
        )
        obs = build_tutor_step_observation(state=state)

        act = policy.decide(
            observation=obs,
            concept_action=ConceptMDPAction.REPLAN_CONCEPT,
        )
        self.assertEqual(act, TutorStepMDPAction.REPLAN_CONCEPT)

    def test_apply_tutor_step_transition_updates_counters(self) -> None:
        state = TutorStepState(
            episode_id="ce-test",
            session_id="s1",
            user_id="u1",
            concept_id="c1",
            plan_id="cp-test",
            plan_index=0,
            plan_length=2,
            last_concept_action=None,
            last_intent="question",
            last_affect="neutral",
            last_control_type=None,
            last_step_type=None,
            last_step_id=None,
            step_count=0,
            replan_count=0,
        )

        outcome = apply_tutor_step_transition(
            prev_state=state,
            mdp_action=TutorStepMDPAction.REPLAN_CONCEPT,
        )
        self.assertIsInstance(outcome.state, TutorStepState)
        self.assertIsInstance(outcome.observation, TutorStepObservation)
        self.assertEqual(outcome.state.step_count, 1)
        self.assertEqual(outcome.state.replan_count, 1)
        self.assertFalse(outcome.terminated)


if __name__ == "__main__":
    unittest.main()
