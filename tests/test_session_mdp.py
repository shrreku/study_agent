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

from agents.tutor.mdp.session import (
    SessionState,
    SessionObservation,
    SessionMDPAction,
    apply_session_transition,
    build_session_state_from_session,
    build_session_observation,
)
from agents.tutor.state import TutorSessionPolicy


class TestSessionMDPTransitions(unittest.TestCase):
    def _make_state(self, plan_index: int, total_concepts: int) -> SessionState:
        return SessionState(
            episode_id="se-test",
            session_id="s1",
            user_id="u1",
            strategy="learning_path",
            concept_plan=[f"c{i}" for i in range(total_concepts)],
            plan_index=plan_index,
            completed_concepts=plan_index,
            total_concepts=total_concepts,
            avg_mastery=None,
            min_mastery=None,
            max_mastery=None,
            remaining_low_mastery_count=0,
            current_concept_id=(f"c{plan_index}" if 0 <= plan_index < total_concepts else None),
            current_concept_mastery=None,
            current_concept_target_mastery=None,
            current_concept_episode_id=None,
            current_concept_terminated=False,
            current_concept_termination_reason=None,
            session_step_count=0,
            concept_episodes_completed=0,
            time_budget_steps=None,
            time_used_steps=0,
            last_session_action=None,
        )

    def _make_tutor_context(self, session_id: str = "s1", user_id: str = "u1", turn_index: int = 3):
        class DummyContext:
            pass

        ctx = DummyContext()
        ctx.session_id = session_id
        ctx.user_id = user_id
        ctx.turn_index = turn_index
        return ctx

    def test_follow_plan_concept_no_index_change(self):
        state = self._make_state(plan_index=0, total_concepts=3)
        outcome = apply_session_transition(
            prev_state=state,
            mdp_action=SessionMDPAction.FOLLOW_PLAN_CONCEPT,
            concept_termination_reason=None,
        )
        self.assertEqual(outcome.state.plan_index, 0)
        self.assertFalse(outcome.terminated)
        self.assertIsNone(outcome.termination_reason)

    def test_advance_in_plan_increments_index(self):
        state = self._make_state(plan_index=0, total_concepts=3)
        outcome = apply_session_transition(
            prev_state=state,
            mdp_action=SessionMDPAction.ADVANCE_IN_PLAN,
            concept_termination_reason="mastery_reached",
        )
        self.assertEqual(outcome.state.plan_index, 1)
        self.assertFalse(outcome.terminated)
        self.assertIsNone(outcome.termination_reason)
        self.assertTrue(outcome.state.current_concept_terminated)
        self.assertEqual(
            outcome.state.current_concept_termination_reason,
            "mastery_reached",
        )

    def test_advance_in_plan_terminates_at_end(self):
        state = self._make_state(plan_index=2, total_concepts=3)
        outcome = apply_session_transition(
            prev_state=state,
            mdp_action=SessionMDPAction.ADVANCE_IN_PLAN,
            concept_termination_reason="max_steps_reached",
        )
        self.assertEqual(outcome.state.plan_index, 3)
        self.assertTrue(outcome.terminated)
        self.assertEqual(outcome.termination_reason, "completed_plan")

    def test_terminate_session_marks_terminated(self):
        state = self._make_state(plan_index=1, total_concepts=3)
        outcome = apply_session_transition(
            prev_state=state,
            mdp_action=SessionMDPAction.TERMINATE_SESSION,
            concept_termination_reason="session_end",
        )
        self.assertTrue(outcome.terminated)
        self.assertEqual(outcome.termination_reason, "session_end")

    def test_build_session_state_and_observation_from_policy(self):
        policy = TutorSessionPolicy()
        policy.session_plan = {
            "strategy": "learning_path",
            "concept_plan": ["c1", "c2"],
        }
        policy.session_plan_index = 1
        policy.session_strategy = "learning_path"

        mastery_map = {
            "c1": {"mastery": 0.5},
            "c2": {"mastery": 0.2},
        }

        ctx = self._make_tutor_context(session_id="s123", user_id="u123", turn_index=3)

        state = build_session_state_from_session(
            policy_state=policy,
            tutor_context=ctx,
            mastery_map=mastery_map,
        )

        # Core identifiers and plan wiring
        self.assertEqual(state.session_id, "s123")
        self.assertEqual(state.user_id, "u123")
        self.assertEqual(state.concept_plan, ["c1", "c2"])
        self.assertEqual(state.plan_index, 1)
        self.assertEqual(state.total_concepts, 2)
        self.assertEqual(state.current_concept_id, "c2")

        # Mastery aggregates and remaining_low_mastery_count
        self.assertAlmostEqual(state.avg_mastery, (0.5 + 0.2) / 2.0)
        self.assertEqual(state.min_mastery, 0.2)
        self.assertEqual(state.remaining_low_mastery_count, 2)

        # Session progress proxies
        self.assertEqual(state.session_step_count, 3)
        self.assertEqual(state.time_used_steps, 3)

        # Observation derived from state
        obs = build_session_observation(state)
        self.assertEqual(obs.current_index, 1)
        self.assertEqual(obs.total_concepts, 2)
        self.assertAlmostEqual(obs.position_fraction, 0.5)
        self.assertEqual(obs.current_concept_id, "c2")
        self.assertEqual(obs.current_mastery, state.current_concept_mastery)


if __name__ == "__main__":
    unittest.main()
