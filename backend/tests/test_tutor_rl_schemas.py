from __future__ import annotations

import json
import os
import sys

import pytest

# ensure backend + scripts modules available
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.schemas.tutor_rl import PreferenceRecord, SFTRecord
from scripts.validate_tutor_datasets import validate_prefs, validate_sft


@pytest.fixture()
def sample_observation() -> dict:
    return {
        "metadata": {"version": 1},
        "user": {
            "message": "Please explain conduction.",
            "user_id": "user-123",
            "target_concepts": ["Conduction"],
        },
        "classifier": {
            "intent": "question",
            "affect": "confused",
            "concept": "Conduction",
            "confidence": 0.6,
        },
        "tutor": {
            "focus_concept": "Conduction",
            "concept_level": "beginner",
            "learning_path": ["Conduction", "Convection"],
        },
        "retrieval": {
            "chunk_ids": ["chunk-1"],
            "chunks": [
                {
                    "id": "chunk-1",
                    "pedagogy_role": "definition",
                    "snippet": "Conduction moves heat through direct contact.",
                }
            ],
        },
        "action": {
            "type": "explain",
            "source_chunk_ids": ["chunk-1"],
        },
        "session": {"session_id": "session-xyz", "turn_index": 0},
    }


def _reward_payload() -> dict:
    return {
        "components": {
            "rubric": {"score": 0.9, "details": {}, "flags": []},
            "intent": {"score": 0.8, "details": {}, "flags": []},
            "gating": {"score": 0.85, "details": {}, "flags": []},
            "grounding": {"score": 0.75, "details": {}, "flags": []},
            "style": {"score": 0.8, "details": {}, "flags": []},
        },
        "total": 0.82,
        "weights": {"rubric": 0.4, "intent": 0.2, "gating": 0.2, "grounding": 0.15, "style": 0.05},
        "normalized_weights": {"rubric": 0.4, "intent": 0.2, "gating": 0.2, "grounding": 0.15, "style": 0.05},
        "flags": [],
    }


def _critic_payload() -> dict:
    return {
        "clarity": 0.8,
        "accuracy": 0.85,
        "support": 0.75,
        "hallucination_flag": False,
        "confidence": 0.78,
        "notes": "focus=conduction",
    }


def test_sft_record_model_validation(sample_observation):
    record = {
        "observation": sample_observation,
        "action": sample_observation["action"],
        "response": "Conduction transfers heat through solids via particle collisions.",
        "reward": _reward_payload(),
        "critic": _critic_payload(),
        "meta": {"prompt_set": "baseline"},
    }

    validated = SFTRecord.model_validate(record)
    assert validated.reward.total == pytest.approx(0.82)
    assert validated.critic and not validated.critic.hallucination_flag


def test_preference_record_model_validation(sample_observation):
    record = {
        "observation": sample_observation,
        "candidates": [
            {
                "action": {"type": "explain"},
                "response": "Conduction transfers heat through solids",
                "reward": _reward_payload(),
                "critic": _critic_payload(),
            },
            {
                "action": {"type": "ask"},
                "response": "Can you give an example of conduction?",
                "reward": _reward_payload(),
                "critic": _critic_payload(),
            },
        ],
        "preference": {
            "chosen": 1,
            "scores": [0.7, 0.82],
            "confidence": 0.65,
            "reason": "candidate 1 more thorough",
        },
    }

    validated = PreferenceRecord.model_validate(record)
    assert validated.preference.chosen == 1
    assert len(validated.candidates) == 2


def test_validate_scripts_detect_weight_error(tmp_path, sample_observation):
    sft_path = tmp_path / "sft.jsonl"
    bad_record = {
        "observation": sample_observation,
        "action": sample_observation["action"],
        "response": "Conduction transfers heat.",
        "reward": {
            **_reward_payload(),
            "normalized_weights": {"rubric": 0.5, "intent": 0.4},
        },
    }
    sft_path.write_text(json.dumps(bad_record) + "\n", encoding="utf-8")

    errors = validate_sft(sft_path)
    assert errors
    assert "normalized weight" in errors[0]

    # preference with mismatched scores length
    prefs_path = tmp_path / "prefs.jsonl"
    pref_record = {
        "observation": sample_observation,
        "candidates": [
            {"action": {"type": "explain"}, "response": "opt a"},
            {"action": {"type": "ask"}, "response": "opt b"},
        ],
        "preference": {"chosen": 0, "scores": [0.8], "confidence": 0.6},
    }
    prefs_path.write_text(json.dumps(pref_record) + "\n", encoding="utf-8")

    pref_errors = validate_prefs(prefs_path)
    assert pref_errors
    assert "scores length" in pref_errors[0]

