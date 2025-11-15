from __future__ import annotations

import os
import sys

import pytest

# ensure backend package root is on path when running tests directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.tutor.validators import score_response  # type: ignore


@pytest.fixture()
def sample_observation() -> dict:
    return {
        "metadata": {"version": 1},
        "user": {
            "message": "Explain conduction with an example.",
            "user_id": "00000000-0000-0000-0000-000000000003",
            "target_concepts": ["Conduction"],
        },
        "classifier": {
            "intent": "question",
            "affect": "confused",
            "concept": "Conduction",
            "confidence": 0.72,
            "needs_escalation": False,
        },
        "tutor": {
            "focus_concept": "Conduction",
            "concept_level": "beginner",
            "inference_concept": "Conduction",
            "learning_path": ["Conduction", "Convection"],
            "target_concepts": ["Conduction"],
            "mastery_snapshot": {"mastery": 0.2, "attempts": 0},
        },
        "retrieval": {
            "chunk_ids": ["chunk-1", "chunk-2", "chunk-3"],
            "source_chunk_ids": ["chunk-1"],
            "pedagogy_roles": ["definition", "example"],
            "chunks": [
                {
                    "id": "chunk-1",
                    "snippet": "Conduction transfers heat through solids via particle collisions.",
                    "page_number": 12,
                },
                {
                    "id": "chunk-2",
                    "snippet": "For example, a metal spoon heats up when left in a hot pot.",
                    "page_number": 13,
                },
            ],
        },
        "action": {
            "type": "explain",
            "cold_start": False,
            "confidence": 0.62,
            "mastery_delta": None,
            "source_chunk_ids": ["chunk-1"],
            "params": {"concept": "Conduction"},
        },
        "session": {"session_id": "session-1", "turn_index": 0, "resource_id": None},
    }


def test_stepwise_component_emitted_when_enabled(sample_observation, monkeypatch):
    monkeypatch.setenv("TUTOR_STEPWISE_RUBRIC_ENABLED", "true")
    # keep export disabled to test primary path
    monkeypatch.setenv("TUTOR_RL_EXPORT_STEP_SCORES", "false")

    response = (
        "Conduction is the transfer of heat through solids because neighbouring particles collide. "
        "For example, a metal spoon warms when its end sits in hot soup. "
        "To check your understanding, can you describe another everyday example?"
    )

    result = score_response(sample_observation, response, {"source_chunk_ids": ["chunk-1"]})
    components = result.get("components", {})

    assert "stepwise_rubric" in components
    sw = components["stepwise_rubric"]
    assert isinstance(sw, dict)
    details = sw.get("details", {})
    assert isinstance(details, dict)
    step_scores = details.get("step_scores", [])
    assert isinstance(step_scores, list) and len(step_scores) >= 6


def test_stepwise_export_when_disabled(sample_observation, monkeypatch):
    # disabled, but export enabled should still include component
    monkeypatch.setenv("TUTOR_STEPWISE_RUBRIC_ENABLED", "false")
    monkeypatch.setenv("TUTOR_RL_EXPORT_STEP_SCORES", "true")

    response = "Conduction transfers heat through solids. Can you explain why this happens?"

    result = score_response(sample_observation, response, {"source_chunk_ids": ["chunk-1"]})
    components = result.get("components", {})
    assert "stepwise_rubric" in components
