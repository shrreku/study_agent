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
)
from agents.tutor.rl_pedagogical_logging import (
    build_pedagogical_step_event,
    pedagogical_step_event_to_dict,
)


class TestPedagogicalRLLogging(unittest.TestCase):
    def _make_state_and_obs(self):
        state = PedagogicalTutorState(
            episode_id="pt-e1",
            session_id="s1",
            user_id="u1",
            concept_id="c1",
            plan_id="plan-1",
            plan_index=1,
            plan_length=3,
            phase="instruction",
            mastery=0.3,
            target_mastery=0.8,
            step_count=5,
            last_intent="question",
            last_affect="neutral",
            last_control_type="continue",
            last_pedagogical_action=PedagogicalTutorAction.EXPLAIN,
            awaiting_mcq_answer=False,
            last_mcq_answered=False,
            last_quiz_correct=None,
            ped_action_counts={PedagogicalTutorAction.EXPLAIN.value: 3},
            recent_ped_actions=[PedagogicalTutorAction.EXPLAIN.value],
        )

        obs = PedagogicalTutorObservation(
            plan_index=1,
            plan_length=3,
            at_end_of_plan=False,
            phase="instruction",
            mastery=0.3,
            target_mastery=0.8,
            mastery_gap=0.5,
            last_intent="question",
            last_affect="neutral",
            last_control_type="continue",
            awaiting_mcq_answer=False,
            last_mcq_answered=False,
            last_quiz_correct=None,
            num_explain_steps=3,
            num_practice_steps=0,
            num_quiz_steps=0,
        )
        return state, obs

    def test_build_pedagogical_step_event(self):
        state, obs = self._make_state_and_obs()

        reward = {"mastery_delta": 0.1}
        mcq_outcome = None
        mastery_before = 0.2
        mastery_after = 0.3

        event = build_pedagogical_step_event(
            state=state,
            observation=obs,
            action=PedagogicalTutorAction.EXPLAIN,
            reward=reward,
            turn_index=7,
            mcq_outcome=mcq_outcome,
            mastery_before=mastery_before,
            mastery_after=mastery_after,
        )

        self.assertEqual(event.episode_id, "pt-e1")
        self.assertEqual(event.session_id, "s1")
        self.assertEqual(event.concept_id, "c1")
        self.assertEqual(event.turn_index, 7)
        self.assertEqual(event.action, PedagogicalTutorAction.EXPLAIN.value)
        self.assertEqual(event.reward["mastery_delta"], 0.1)
        self.assertEqual(event.mastery_before, mastery_before)
        self.assertEqual(event.mastery_after, mastery_after)

        payload = pedagogical_step_event_to_dict(event)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["episode_id"], "pt-e1")
        self.assertIn("observation", payload)


if __name__ == "__main__":
    unittest.main()
