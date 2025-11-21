from __future__ import annotations

import os
import sys
from typing import Dict

import pytest

# ensure project root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.tutor.agent import tutor_agent  # noqa: E402
from agents.tutor.config import reset_tutor_config  # noqa: E402


@pytest.fixture()
def mock_payload() -> Dict[str, object]:
    return {
        "message": "Can you remind me about conduction?",
        "user_id": "11111111-2222-3333-4444-555555555555",
        "session_id": None,
        "resource_id": None,
        "target_concepts": ["Heat Transfer Fundamentals"],
        "session_policy": {"version": 1, "strategy": "baseline"},
    }


def _call_tutor(payload: Dict[str, object]) -> Dict[str, object]:
    resp = tutor_agent(payload)
    assert isinstance(resp, dict)
    return resp


def test_tutor_emit_observation_default(mock_payload):
    payload = {**mock_payload, "emit_state": True}
    resp = _call_tutor(payload)
    assert "observation" in resp
    observation = resp["observation"]
    assert observation["metadata"]["version"] == 1
    assert observation["session"]["turn_index"] == 0


def test_tutor_observation_has_retrieval(mock_payload):
    payload = {**mock_payload, "emit_state": True}
    resp = _call_tutor(payload)
    observation = resp["observation"]
    retrieval = observation["retrieval"]
    assert isinstance(retrieval, dict)
    chunk_ids = retrieval.get("chunk_ids")
    assert isinstance(chunk_ids, list)


def test_tutor_action_override_ask(mock_payload):
    payload = {
        **mock_payload,
        "emit_state": True,
        "action_override": {
            "type": "ask",
            "params": {
                "concept": "Heat Transfer",
                "difficulty": "medium",
                "question_type": "conceptual",
            },
        },
    }
    resp = _call_tutor(payload)
    assert resp["action_type"] == "ask"
    mcq = resp.get("mcq")
    assert isinstance(mcq, dict)
    options = mcq.get("options")
    assert isinstance(options, list) and len(options) > 0

    observation = resp["observation"]
    assert observation["action"]["override_applied"] is True
    assert observation["action"]["override_type"] == "ask"
    assert observation["action"]["applied_override_type"] == "ask"


def test_tutor_action_override_worked_example(mock_payload):
    payload = {
        **mock_payload,
        "emit_state": True,
        "action_override": {
            "type": "worked_example",
            "params": {"concept": "Energy Balance"},
        },
    }
    resp = _call_tutor(payload)
    assert resp["action_type"] == "worked_example"
    assert "worked example" in resp["response"].lower()
    observation = resp["observation"]
    assert observation["action"]["override_applied"] is True
    assert observation["action"]["override_type"] == "worked_example"
    assert observation["action"]["applied_override_type"] == "worked_example"


def test_step_mode_includes_step_plan_and_controls(monkeypatch, mock_payload):
    monkeypatch.setenv("USE_LLM_MOCK", "1")
    monkeypatch.setenv("TUTOR_MODE", "step_by_step")
    reset_tutor_config()

    payload = {
        **mock_payload,
        "emit_state": True,
        "agent_action_mode": "step_by_step",
    }

    resp = _call_tutor(payload)

    # Basic sanity: step-by-step mode should still return a normal tutor response
    assert resp.get("session_id")
    assert resp.get("turn_index") == 0
    assert resp.get("action_type") in {"explain", "ask", "hint", "review", "reflect"}

    # Canonical step projection should be present with a fixed control surface
    step = resp.get("step")
    assert isinstance(step, dict)
    assert step.get("id")
    assert step.get("type")
    assert step.get("phase")

    controls = step.get("controls")
    assert isinstance(controls, list) and len(controls) > 0
    control_types = {c.get("type") for c in controls if isinstance(c, dict)}
    # The fixed control surface should expose the standard step-by-step buttons
    assert {"continue", "skip_to_quiz", "skip_to_next_concept", "end_session"}.issubset(
        control_types
    )

    # Plan projection and legacy SRL fields are optional on a single turn; if
    # present they should expose a reasonable shape for backward compatibility.
    plan = resp.get("plan")
    if plan is not None:
        assert isinstance(plan, dict)
        assert plan.get("steps") is not None

        # Legacy SRL fields should still be threaded through when a plan is
        # available on the response payload.
        assert "srl_plan" in resp
        assert "srl_next_step" in resp

    # Reset config so other tests are not affected by TUTOR_MODE override
    reset_tutor_config()


def test_auto_mode_exposes_debug_step_not_public_step(monkeypatch, mock_payload):
    """Auto mode should still route through StepEngine and expose debug_step only.

    This verifies the V3-04 contract that non-step-by-step flows use
    StepEngine.decide_step(control_mode="auto") and that the orchestrator attaches
    a StudyStep projection for tooling via debug_step while keeping the public
    step field reserved for explicit step-by-step mode.
    """

    # Ensure we are in an auto/intelligent tutor mode, not step_by_step.
    monkeypatch.setenv("USE_LLM_MOCK", "1")
    monkeypatch.setenv("TUTOR_MODE", "intelligent")
    reset_tutor_config()

    payload = {
        **mock_payload,
        "emit_state": True,
        # Omit agent_action_mode or set it explicitly to auto; either way this
        # should take the auto/control_mode="auto" path.
        "agent_action_mode": "auto",
    }

    resp = _call_tutor(payload)

    # Sanity: auto mode still returns a normal tutor turn.
    assert resp.get("session_id")
    assert resp.get("turn_index") == 0
    assert resp.get("action_type") in {"explain", "ask", "hint", "review", "reflect"}

    # In auto mode we do not expose the public step field, but we should attach
    # a canonical StudyStep projection for tooling via debug_step.
    assert "step" not in resp

    debug_step = resp.get("debug_step")
    assert isinstance(debug_step, dict)
    assert debug_step.get("id")
    assert debug_step.get("type")
    assert debug_step.get("phase")

    # Reset config so other tests are not affected by TUTOR_MODE override
    reset_tutor_config()


def test_step_mode_two_turn_flow_has_consistent_step_contract(monkeypatch, mock_payload):
    """Step mode should expose step/controls consistently across turns.

    This provides a lightweight V3-06 E2E check that a single session using
    step-by-step mode can advance from turn 0 to turn 1 via a control-only
    continue turn while preserving the canonical step projection.
    """

    monkeypatch.setenv("USE_LLM_MOCK", "1")
    monkeypatch.setenv("TUTOR_MODE", "step_by_step")
    reset_tutor_config()

    # Turn 0: initial step-mode message
    payload0 = {
        **mock_payload,
        "emit_state": True,
        "agent_action_mode": "step_by_step",
    }

    resp0 = _call_tutor(payload0)

    session_id = resp0.get("session_id")
    assert session_id
    assert resp0.get("turn_index") == 0

    step0 = resp0.get("step")
    assert isinstance(step0, dict)
    assert step0.get("id")
    assert step0.get("type")
    assert step0.get("phase")
    controls0 = step0.get("controls")
    assert isinstance(controls0, list) and len(controls0) > 0

    # Turn 1: control-only continue turn in the same session
    payload1 = {
        **mock_payload,
        "session_id": session_id,
        "message": "",
        "emit_state": True,
        "agent_action_mode": "step_by_step",
        "confirmed_action": "continue",
    }

    resp1 = _call_tutor(payload1)

    assert resp1.get("session_id") == session_id
    assert resp1.get("turn_index") == 1

    step1 = resp1.get("step")
    assert isinstance(step1, dict)
    assert step1.get("id")
    assert step1.get("type")
    assert step1.get("phase")

    controls1 = step1.get("controls")
    assert isinstance(controls1, list) and len(controls1) > 0

    # Step IDs should differ across turns within the same session.
    assert step1.get("id") != step0.get("id")

    # Reset config so other tests are not affected by TUTOR_MODE override
    reset_tutor_config()


def test_auto_mode_two_turn_flow_has_consistent_debug_step(monkeypatch, mock_payload):
    """Auto mode should expose debug_step consistently across turns.

    This provides a lightweight V3-06 E2E check that an auto-mode session
    advances from turn 0 to turn 1 while preserving the canonical StudyStep
    projection via debug_step and not exposing the public step field.
    """

    monkeypatch.setenv("USE_LLM_MOCK", "1")
    monkeypatch.setenv("TUTOR_MODE", "intelligent")
    reset_tutor_config()

    # Turn 0: initial auto-mode message
    payload0 = {
        **mock_payload,
        "emit_state": True,
        "agent_action_mode": "auto",
    }

    resp0 = _call_tutor(payload0)

    session_id = resp0.get("session_id")
    assert session_id
    assert resp0.get("turn_index") == 0

    # Auto mode should not expose public step but should expose debug_step.
    assert "step" not in resp0
    debug_step0 = resp0.get("debug_step")
    assert isinstance(debug_step0, dict)
    assert debug_step0.get("id")
    assert debug_step0.get("type")
    assert debug_step0.get("phase")

    # Turn 1: follow-up auto-mode message in the same session
    payload1 = {
        **mock_payload,
        "session_id": session_id,
        "message": "Thanks, can you ask me a question now?",
        "emit_state": True,
        "agent_action_mode": "auto",
    }

    resp1 = _call_tutor(payload1)

    assert resp1.get("session_id") == session_id
    assert resp1.get("turn_index") == 1
    assert "step" not in resp1

    debug_step1 = resp1.get("debug_step")
    assert isinstance(debug_step1, dict)
    assert debug_step1.get("id")
    assert debug_step1.get("type")
    assert debug_step1.get("phase")

    # Debug step IDs should differ across turns within the same session.
    assert debug_step1.get("id") != debug_step0.get("id")

    # Reset config so other tests are not affected by TUTOR_MODE override
    reset_tutor_config()
