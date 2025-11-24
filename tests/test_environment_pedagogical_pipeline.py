import sys
import os
import unittest
from unittest import mock
import types

# Mock external dependencies expected by backend imports.

# psycopg2 and psycopg2.extras
mock_psycopg2 = types.ModuleType("psycopg2")
mock_psycopg2_extras = types.ModuleType("psycopg2.extras")

def _json_identity(obj, *args, **kwargs):
    """Lightweight stand-in for psycopg2.extras.Json.

    The real Json wrapper is used only for serialization; in tests we
    just need a callable that returns the wrapped object. We accept
    *args/**kwargs to be permissive.
    """

    return obj


mock_psycopg2_extras.Json = _json_identity  # type: ignore[attr-defined]
sys.modules.setdefault("psycopg2", mock_psycopg2)
sys.modules.setdefault("psycopg2.extras", mock_psycopg2_extras)

# Stub core and core.db with a dummy get_db_conn so imports in agents.* succeed
mock_core = types.ModuleType("core")
mock_core_db = types.ModuleType("core.db")
setattr(mock_core_db, "get_db_conn", lambda *args, **kwargs: None)
sys.modules.setdefault("core", mock_core)
sys.modules.setdefault("core.db", mock_core_db)

# Add backend to path so we can import agents.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from agents.tutor.environment import run_environment_turn  # type: ignore
from agents.tutor.runtime.context import TurnContext  # type: ignore


class DummyCursor:
    """Minimal cursor stub for EnvironmentStateManager.

    This stub returns a small, in-memory policy shape sufficient for the
    environment orchestrator to run without talking to a real DB. We do
    not attempt to persist state across calls; the test focuses on the
    control flow and LLM plumbing, not on SQL behaviour.
    """

    def __init__(self) -> None:
        self._last_query = None
        self._last_params = None
        self._last_result = None
        # In-memory store: session_id -> policy dict
        self._policy_by_session = {}

    # The EnvironmentStateManager uses `execute`, `fetchone`, `fetchall`.
    def execute(self, query, params=None):  # type: ignore[override]
        self._last_query = " ".join(str(query).split()).strip()
        self._last_params = params
        q = self._last_query or ""

        # Load-session query: return policy + target_concepts.
        if "SELECT policy, target_concepts" in q and "FROM tutor_session" in q:
            session_id = params[0]
            policy = self._policy_by_session.get(session_id, {})
            self._last_result = (policy, ["c1"])
        # Load-concept / policy-only query.
        elif "SELECT policy" in q and "FROM tutor_session" in q:
            session_id = params[0]
            policy = self._policy_by_session.get(session_id, {})
            self._last_result = (policy,)
        # Save policy from environment (session or concept state).
        elif "UPDATE tutor_session" in q and "SET policy" in q:
            policy_obj, session_id = params
            # Json wrapper is an identity in tests, so policy_obj is a dict.
            self._policy_by_session[session_id] = policy_obj
            self._last_result = None
        # Mastery queries just default to 0.0 / empty.
        elif "FROM user_concept_mastery" in q:
            self._last_result = (0.0,)
        else:
            self._last_result = None

    def fetchone(self):  # type: ignore[override]
        return self._last_result

    def fetchall(self):  # type: ignore[override]
        q = self._last_query or ""
        if "FROM user_concept_mastery" in q and "WHERE user_id" in q:
            return []
        return []


class EnvironmentPedagogicalPipelineTest(unittest.TestCase):
    def test_environment_turns_with_llm_and_plan_navigation(self):
        from agents.tutor.environment.tools import EnvironmentStateManager  # type: ignore
        from agents.tutor.tools import pedagogical_response  # type: ignore

        cur = DummyCursor()
        session_id = "00000000-0000-0000-0000-000000000001"
        user_id = "11111111-1111-1111-1111-111111111111"

        # Sanity: EnvironmentStateManager should be constructible with the stub cursor.
        _ = EnvironmentStateManager(cur)

        # Mock the pedagogical response LLM call to return a deterministic payload.
        with mock.patch.object(
            pedagogical_response, "call_llm_json", return_value={"response": "llm-text", "ui_mode": "free_text"}
        ) as llm_mock:
            # First turn: start session and generate plan.
            ctx = TurnContext(
                session_id=session_id,
                user_id=user_id,
                turn_index=0,
                message="hello",
                target_concepts=["c1"],
            )
            out1 = run_environment_turn(ctx, cur)

            self.assertIn("messages", out1)
            self.assertGreaterEqual(len(out1["messages"]), 1)
            self.assertIn("debug", out1)
            # Tutor response should have come from the (mocked) LLM path.
            llm_mock.assert_called()

            # Second turn: simulate a button click to advance the concept step.
            ctx2 = TurnContext(
                session_id=session_id,
                user_id=user_id,
                turn_index=1,
                message="continue",
                payload={"button_clicked": True},
            )
            out2 = run_environment_turn(ctx2, cur)
            self.assertIn("messages", out2)
            self.assertIn("debug", out2)
            debug2 = out2["debug"]
            self.assertIn("concept", debug2)
            self.assertIn("current_step_index", debug2["concept"])


if __name__ == "__main__":
    unittest.main()
