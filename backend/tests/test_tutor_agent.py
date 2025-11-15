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


@pytest.fixture()
def mock_payload() -> Dict[str, object]:
    return {
        "message": "Can you remind me about conduction?",
        "user_id": "00000000-0000-0000-0000-000000000001",
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
    assert observation["retrieval"]["chunk_ids"] == []


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
    assert "What step should come next" in resp["response"]
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
