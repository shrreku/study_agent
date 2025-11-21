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
    PedagogicalTutorAction,
    PedagogicalTutorState,
    build_pedagogical_tutor_state,
)
from agents.tutor.mdp.tools_factory import make_pedagogical_response_tool
from agents.tutor.config import get_tutor_config


class TestPedagogicalResponseTool(unittest.TestCase):
    def _make_state(self) -> PedagogicalTutorState:
        return build_pedagogical_tutor_state(
            episode_id="pt-e1",
            session_id="s1",
            user_id="u1",
            concept_id="c1",
            plan_id="plan-1",
            plan_index=0,
            plan_length=3,
            phase="instruction",
            mastery=0.2,
            target_mastery=0.8,
            last_intent="ask_question",
            last_affect="neutral",
            last_control_type=None,
            awaiting_mcq_answer=False,
            last_mcq_answered=False,
            last_quiz_correct=None,
            previous_state=None,
            last_pedagogical_action=None,
        )

    def test_explain_action_produces_free_text_message(self):
        config = get_tutor_config()
        tool = make_pedagogical_response_tool(config)
        state = self._make_state()

        context_obs = {
            "student_message": "How does this formula work?",
            "base_prefix": "",
        }

        out = tool(
            user_id="u1",
            session_id="s1",
            concept_id="c1",
            pedagogical_action=PedagogicalTutorAction.EXPLAIN,
            ped_state=state,
            concept_plan=None,
            context_obs=context_obs,
        )

        self.assertIsInstance(out, dict)
        self.assertIn("messages", out)
        self.assertEqual(out.get("ui_mode"), "free_text")
        self.assertIsNone(out.get("mcq_payload"))
        self.assertGreater(len(out["messages"]), 0)

    def test_quiz_action_produces_mcq_payload(self):
        config = get_tutor_config()
        tool = make_pedagogical_response_tool(config)
        state = self._make_state()

        context_obs = {
            "student_message": "I think I understand.",
        }

        out = tool(
            user_id="u1",
            session_id="s1",
            concept_id="c1",
            pedagogical_action=PedagogicalTutorAction.QUIZ_MCQ,
            ped_state=state,
            concept_plan=None,
            context_obs=context_obs,
        )

        self.assertEqual(out.get("ui_mode"), "mcq")
        mcq = out.get("mcq_payload")
        self.assertIsInstance(mcq, dict)
        self.assertIn("question", mcq)
        self.assertIn("options", mcq)
        self.assertIn("correct_option_id", mcq)


if __name__ == "__main__":
    unittest.main()
