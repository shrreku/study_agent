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

from agents.tutor.mdp.pedagogical_tutor import (
    PedagogicalTutorState,
    PedagogicalTutorObservation,
    PedagogicalTutorAction,
    build_pedagogical_tutor_state,
    build_pedagogical_tutor_observation,
    apply_pedagogical_transition,
)
from agents.tutor.mdp.policy import DefaultPedagogicalTutorPolicy


class TestPedagogicalTutorMDP(unittest.TestCase):
    def _make_state(
        self,
        *,
        plan_index: int = 0,
        plan_length: int = 3,
        mastery: float = 0.1,
        target_mastery: float = 0.8,
        last_intent: str = "ask_question",
        last_affect: str = "neutral",
    ) -> PedagogicalTutorState:
        return build_pedagogical_tutor_state(
            episode_id="pt-e1",
            session_id="s1",
            user_id="u1",
            concept_id="c1",
            plan_id="plan-1",
            plan_index=plan_index,
            plan_length=plan_length,
            phase="instruction",
            mastery=mastery,
            target_mastery=target_mastery,
            last_intent=last_intent,
            last_affect=last_affect,
            last_control_type=None,
            awaiting_mcq_answer=False,
            last_mcq_answered=False,
            last_quiz_correct=None,
            previous_state=None,
            last_pedagogical_action=None,
        )

    def test_build_observation_from_state(self):
        state = self._make_state(plan_index=1, plan_length=4, mastery=0.5, target_mastery=0.8)
        obs = build_pedagogical_tutor_observation(state=state)

        self.assertEqual(obs.plan_index, 1)
        self.assertEqual(obs.plan_length, 4)
        self.assertFalse(obs.at_end_of_plan)
        self.assertEqual(obs.mastery, 0.5)
        self.assertEqual(obs.target_mastery, 0.8)
        self.assertAlmostEqual(obs.mastery_gap, 0.8 - 0.5)
        self.assertEqual(obs.num_explain_steps, 0)
        self.assertEqual(obs.num_practice_steps, 0)
        self.assertEqual(obs.num_quiz_steps, 0)

    def test_apply_transition_updates_history_and_plan_index(self):
        state = self._make_state(plan_index=0, plan_length=2)
        outcome = apply_pedagogical_transition(
            prev_state=state,
            mdp_action=PedagogicalTutorAction.EXPLAIN,
            mastery_delta=0.05,
            quiz_delta=None,
            last_quiz_correct=None,
            awaiting_mcq_answer=False,
            last_mcq_answered=False,
            last_control_type="continue",
        )

        new_state = outcome.state
        obs = outcome.observation

        self.assertEqual(new_state.step_count, 1)
        self.assertIn(PedagogicalTutorAction.EXPLAIN.value, new_state.ped_action_counts)
        self.assertEqual(new_state.ped_action_counts[PedagogicalTutorAction.EXPLAIN.value], 1)
        self.assertIn(PedagogicalTutorAction.EXPLAIN.value, new_state.recent_ped_actions)
        self.assertEqual(new_state.plan_index, 1)
        self.assertEqual(new_state.last_control_type, "continue")
        self.assertFalse(obs.at_end_of_plan)
        self.assertIn("mastery_delta", outcome.reward)

    def test_default_policy_low_mastery_prefers_explain(self):
        state = self._make_state(plan_index=0, plan_length=3, mastery=0.1, target_mastery=0.8)
        obs = build_pedagogical_tutor_observation(state=state)
        policy = DefaultPedagogicalTutorPolicy()

        action = policy.decide(observation=PedagogicalTutorObservation(**obs.__dict__), concept_plan=None)
        self.assertEqual(action, PedagogicalTutorAction.EXPLAIN)

    def test_default_policy_near_target_end_of_plan_prefers_summary(self):
        state = self._make_state(plan_index=2, plan_length=3, mastery=0.78, target_mastery=0.8)
        obs = build_pedagogical_tutor_observation(state=state)
        policy = DefaultPedagogicalTutorPolicy()

        action = policy.decide(observation=PedagogicalTutorObservation(**obs.__dict__), concept_plan=None)
        self.assertEqual(action, PedagogicalTutorAction.SUMMARY)


if __name__ == "__main__":
    unittest.main()
