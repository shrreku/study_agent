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

from agents.tutor.runtime.context import TurnContext
from agents.tutor.mdp.plans import SessionPlan, SessionPlanEntry, ConceptPlan, ConceptPlanStep
import agents.tutor.runtime_v2.orchestrator_pedagogical_mdp as orch
from agents.tutor.config import get_tutor_config


class TestPedagogicalOrchestrator(unittest.TestCase):
    def setUp(self) -> None:
        # Patch persistence and knowledge helpers to avoid DB access.
        self._orig_get_session_state = orch.get_session_state
        self._orig_get_recent_turns = orch.get_recent_turns
        self._orig_insert_turn = orch.insert_turn
        self._orig_update_session = orch.update_session
        self._orig_fetch_mastery_map = orch.fetch_mastery_map
        self._orig_fetch_prereq_chain = orch.fetch_prereq_chain

        orch.get_session_state = lambda cur, session_id: {
            "last_concept": None,
            "last_action": None,
            "target_concepts": ["c1"],
            "policy": {},
        }
        orch.get_recent_turns = lambda cur, session_id, limit=6: []
        orch.insert_turn = lambda *args, **kwargs: None
        orch.update_session = lambda *args, **kwargs: None
        orch.fetch_mastery_map = lambda cur, user_id: {"c1": {"mastery": 0.2}}
        orch.fetch_prereq_chain = lambda concepts: concepts

        # Patch tools factories to return simple deterministic implementations.
        self._orig_make_session_planner_tool = orch.make_session_planner_tool
        self._orig_make_concept_planner_tool = orch.make_concept_planner_tool
        self._orig_make_mastery_estimator_tool = orch.make_mastery_estimator_tool
        self._orig_make_quiz_evaluator_tool = orch.make_quiz_evaluator_tool
        self._orig_make_pedagogical_response_tool = orch.make_pedagogical_response_tool

        def fake_session_planner(**kwargs):
            entry = SessionPlanEntry(concept_id="c1", target_mastery=0.8)
            return SessionPlan(
                plan_id="sp1",
                strategy="learning_path",
                entries=[entry],
                created_at_step=0,
                source="test",
            )

        def fake_concept_planner(**kwargs):
            step = ConceptPlanStep(
                step_id="st1",
                step_type="EXPLAIN",
                subgoal=None,
                instruction="Explain c1",
                tool_call=None,
                params={},
                evaluation_type=None,
                expected_duration_steps=1,
            )
            return ConceptPlan(
                plan_id="cp1",
                concept_id="c1",
                steps=[step],
                created_at_step=0,
                source="test",
                initial_mastery=None,
                target_mastery=0.8,
            )

        orch.make_session_planner_tool = lambda config: fake_session_planner
        orch.make_concept_planner_tool = lambda config: fake_concept_planner
        orch.make_mastery_estimator_tool = lambda config: (
            lambda concept_id, mastery_before, recent_interactions: (mastery_before + 0.1, 0.1)
        )
        orch.make_quiz_evaluator_tool = lambda config: (
            lambda question, user_answer, correct_answer: ({"answer_correct": True}, 0.5)
        )

        class FakePedagogicalResponseTool:
            def __call__(
                self,
                *,
                user_id,
                session_id,
                concept_id,
                pedagogical_action,
                ped_state,
                concept_plan,
                context_obs,
            ):
                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": f"TEST_RESPONSE action={pedagogical_action.value}",
                        }
                    ],
                    "ui_mode": "free_text",
                    "mcq_payload": None,
                    "debug": {"pedagogical_action": pedagogical_action.value},
                }

        orch.make_pedagogical_response_tool = lambda config: FakePedagogicalResponseTool()

        # Patch classifier to avoid LLM calls.
        self._orig_TurnClassifier = orch.TurnClassifier

        class FakeTurnClassifier:
            def classify(self, ctx, session_state, controls):
                return orch.ClassificationContext(
                    intent="question",
                    affect="neutral",
                    concept="c1",
                    confidence=0.9,
                )

        orch.TurnClassifier = FakeTurnClassifier

    def tearDown(self) -> None:
        # Restore patched functions
        orch.get_session_state = self._orig_get_session_state
        orch.get_recent_turns = self._orig_get_recent_turns
        orch.insert_turn = self._orig_insert_turn
        orch.update_session = self._orig_update_session
        orch.fetch_mastery_map = self._orig_fetch_mastery_map
        orch.fetch_prereq_chain = self._orig_fetch_prereq_chain

        orch.make_session_planner_tool = self._orig_make_session_planner_tool
        orch.make_concept_planner_tool = self._orig_make_concept_planner_tool
        orch.make_mastery_estimator_tool = self._orig_make_mastery_estimator_tool
        orch.make_quiz_evaluator_tool = self._orig_make_quiz_evaluator_tool
        orch.make_pedagogical_response_tool = self._orig_make_pedagogical_response_tool

        orch.TurnClassifier = self._orig_TurnClassifier

    def test_run_pedagogical_mdp_turn_basic_flow(self):
        config = get_tutor_config()
        _ = config  # ensure config is initialised

        ctx = TurnContext(
            session_id="s1",
            user_id="u1",
            turn_index=0,
            message="Can you explain concept c1?",
            target_concepts=["c1"],
            resource_id=None,
            dry_run=False,
            emit_state_requested=False,
            payload={},
        )

        result = orch.run_pedagogical_mdp_turn(ctx, cur=None)

        self.assertIsInstance(result, dict)
        self.assertIn("messages", result)
        self.assertIn("debug", result)
        self.assertEqual(result.get("ui_mode"), "free_text")
        self.assertEqual(result.get("agent_action_mode"), "pedagogical_step_by_step")

        debug = result.get("debug") or {}
        self.assertIn("session_mdp_action", debug)
        self.assertIn("concept_mdp_action", debug)
        self.assertIn("pedagogical_tutor_action", debug)


if __name__ == "__main__":
    unittest.main()
