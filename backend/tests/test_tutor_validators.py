from __future__ import annotations

from typing import Dict

import os
import sys

import pytest

# ensure backend package root is on path when running tests directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.tutor.validators import score_response


@pytest.fixture()
def sample_observation() -> Dict[str, object]:
    return {
        "metadata": {"version": 1},
        "user": {
            "message": "Can you help me review conduction?",
            "user_id": "00000000-0000-0000-0000-000000000002",
            "target_concepts": ["Heat Transfer Fundamentals"],
        },
        "classifier": {
            "intent": "question",
            "affect": "confused",
            "concept": "Conduction",
            "confidence": 0.7,
            "needs_escalation": False,
        },
        "tutor": {
            "focus_concept": "Conduction",
            "concept_level": "beginner",
            "inference_concept": "Conduction",
            "learning_path": ["Conduction", "Convection", "Radiation"],
            "target_concepts": ["Heat Transfer Fundamentals"],
            "mastery_snapshot": {"mastery": 0.35, "attempts": 1},
        },
        "retrieval": {
            "chunk_ids": ["chunk-1", "chunk-2"],
            "source_chunk_ids": ["chunk-1"],
            "pedagogy_roles": ["definition", "example"],
            "chunks": [
                {
                    "id": "chunk-1",
                    "snippet": "Conduction transfers heat through solids via particle collisions.",
                    "page_number": 12,
                    "score": 0.89,
                    "sim": 0.82,
                    "bm25": 11.2,
                },
                {
                    "id": "chunk-2",
                    "snippet": "For example, a metal spoon heats up when left in a hot pot.",
                    "page_number": 13,
                    "score": 0.77,
                    "sim": 0.74,
                    "bm25": 9.4,
                },
            ],
        },
        "policy": {
            "cold_start": False,
            "consecutive_explains": 0,
            "focus_concept": "Conduction",
        },
        "session": {
            "session_id": "session-1",
            "turn_index": 0,
            "resource_id": None,
        },
        "action": {
            "type": "explain",
            "cold_start": False,
            "confidence": 0.62,
            "mastery_delta": None,
            "source_chunk_ids": ["chunk-1"],
            "params": {"concept": "Conduction"},
            "override_type": None,
            "override_applied": False,
            "applied_override_type": None,
        },
    }


def test_score_response_happy_path(sample_observation):
    response = (
        "Conduction is the transfer of heat through solids because neighbouring particles collide. "
        "For example, a metal spoon warms when its end sits in hot soup. "
        "To check your understanding, can you describe another everyday example?"
    )
    metadata = {"source_chunk_ids": ["chunk-1"]}

    result = score_response(sample_observation, response, metadata)

    assert isinstance(result, dict)
    assert result["total"] >= 0.7 - 0.05
    rubric = result["components"]["rubric"]
    assert rubric["score"] == 1.0
    intent = result["components"]["intent"]
    assert intent["score"] >= 0.8
    assert "flags" in result and "gating_below_threshold" not in result["flags"]


def test_prereq_gate_flags_when_advanced_terms_detected(sample_observation):
    response = (
        "Conduction is important, but convection and radiation also move energy. "
        "Because of those modes, heat leaves surfaces quickly."
    )
    metadata = {"source_chunk_ids": ["chunk-1"]}

    result = score_response(sample_observation, response, metadata)
    gating = result["components"]["gating"]

    assert gating["score"] < 0.7
    assert "advanced_concept_drift" in gating["flags"]
    assert "gating_below_threshold" in result["flags"]


def test_grounding_penalizes_unknown_ids(sample_observation):
    response = "Conduction is the transfer of heat through solids."
    metadata = {"source_chunk_ids": ["chunk-3"]}

    result = score_response(sample_observation, response, metadata)
    grounding = result["components"]["grounding"]

    assert grounding["score"] < 0.6
    assert "unknown_grounding_ids" in grounding["flags"]
    assert "grounding_low" in grounding["flags"]

