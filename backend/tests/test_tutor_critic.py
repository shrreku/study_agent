from __future__ import annotations

from typing import Dict, List

import os
import sys

import pytest

# ensure backend package root is on path when running tests directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.tutor.critic import preference_with_critic, score_with_critic


@pytest.fixture()
def observation() -> Dict[str, object]:
    return {
        "classifier": {
            "intent": "question",
            "affect": "confused",
        },
        "tutor": {
            "focus_concept": "Conduction",
            "inference_concept": "Conduction",
        },
        "retrieval": {
            "chunk_ids": ["chunk-1", "chunk-2"],
            "chunks": [
                {
                    "id": "chunk-1",
                    "pedagogy_role": "definition",
                    "snippet": "Conduction transfers heat through solids via particle collisions.",
                },
                {
                    "id": "chunk-2",
                    "pedagogy_role": "example",
                    "snippet": "For example, a metal spoon heats in a pot of soup.",
                },
            ],
        },
        "action": {
            "type": "explain",
            "source_chunk_ids": ["chunk-1"],
        },
    }


@pytest.fixture(autouse=True)
def enable_mock_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USE_LLM_MOCK", "1")
    yield
    monkeypatch.delenv("USE_LLM_MOCK", raising=False)


def test_score_with_critic_mock_mode(observation):
    response = (
        "Conduction is the transfer of heat through solids because particles collide and share energy. "
        "For example, a spoon warms when left in hot soup."
    )
    metadata = {"source_chunk_ids": ["chunk-1"]}

    result = score_with_critic(observation, response, metadata)

    assert set(result.keys()) >= {"clarity", "accuracy", "support", "confidence", "hallucination_flag", "notes"}
    assert 0.0 <= result["clarity"] <= 1.0
    assert result["accuracy"] >= 0.6
    assert result["hallucination_flag"] is False
    assert isinstance(result["notes"], str)


def test_preference_with_critic_prefers_higher_reward(observation):
    candidates: List[Dict[str, object]] = [
        {
            "action": {"type": "explain"},
            "response": "Conduction moves heat through solids via particle collisions.",
            "reward": {"total": 0.62},
            "critic": {"confidence": 0.7},
        },
        {
            "action": {"type": "ask"},
            "response": "Can you describe conduction?",
            "reward": {"total": 0.78},
            "critic": {"confidence": 0.4},
        },
    ]

    decision = preference_with_critic(observation, candidates)

    assert decision["chosen"] == 1
    assert len(decision["scores"]) == len(candidates)
    assert decision["confidence"] >= 0.6

