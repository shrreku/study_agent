from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure script imports resolve backend modules
BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
for candidate in (BACKEND_DIR, ROOT_DIR, ROOT_DIR / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.tutor_rollout_bandit import main as rollout_main
from scripts.validate_tutor_datasets import validate_prefs, validate_sft


@pytest.fixture()
def observation_entries() -> list[dict[str, object]]:
    base_observation = {
        "metadata": {"version": 1},
        "user": {
            "message": "Can you explain conduction?",
            "user_id": "00000000-0000-0000-0000-000000000111",
            "target_concepts": ["Conduction"],
        },
        "classifier": {
            "intent": "question",
            "affect": "confused",
            "concept": "Conduction",
            "confidence": 0.6,
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
            "chunk_ids": ["chunk-1"],
            "source_chunk_ids": ["chunk-1"],
            "pedagogy_roles": ["definition"],
            "chunks": [
                {
                    "id": "chunk-1",
                    "pedagogy_role": "definition",
                    "snippet": "Conduction transfers heat through solid materials via particle collisions.",
                }
            ],
        },
        "policy": {"cold_start": False, "consecutive_explains": 0, "focus_concept": "Conduction"},
        "session": {"session_id": "mock-session-1", "turn_index": 0, "resource_id": None},
        "action": {
            "type": "explain",
            "cold_start": False,
            "confidence": 0.5,
            "mastery_delta": None,
            "source_chunk_ids": ["chunk-1"],
            "params": {"concept": "Conduction"},
        },
    }

    payload = {
        "message": "Can you explain conduction?",
        "user_id": "00000000-0000-0000-0000-000000000111",
        "session_id": "mock-session-1",
        "resource_id": None,
        "target_concepts": ["Conduction"],
        "session_policy": {"version": 1, "strategy": "baseline"},
    }

    other_observation = json.loads(json.dumps(base_observation))
    other_observation["user"]["message"] = "What is convection?"
    other_observation["classifier"]["concept"] = "Convection"
    other_observation["tutor"]["focus_concept"] = "Convection"
    other_observation["retrieval"]["chunk_ids"] = ["chunk-2"]
    other_observation["retrieval"]["chunks"][0]["id"] = "chunk-2"
    other_payload = json.loads(json.dumps(payload))
    other_payload["message"] = "What is convection?"
    other_payload["session_id"] = "mock-session-2"

    return [
        {"id": "obs-1", "payload": payload, "observation": base_observation},
        {"id": "obs-2", "payload": other_payload, "observation": other_observation},
    ]


def test_rollout_mock_mode(tmp_path: Path, observation_entries: list[dict[str, object]]):
    observations_path = tmp_path / "observations.jsonl"
    observations_path.write_text(
        "\n".join(json.dumps(entry) for entry in observation_entries) + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    args = [
        "--observations",
        str(observations_path),
        "--out-dir",
        str(out_dir),
        "--candidates",
        "2",
        "--actions",
        "explain,ask",
        "--mock",
        "--seed",
        "123",
        "--prompt-set",
        "baseline",
    ]

    # Ensure LLM calls remain mocked
    os.environ["USE_LLM_MOCK"] = "1"
    result = rollout_main(args)
    assert result == 0

    sft_path = out_dir / "sft.jsonl"
    prefs_path = out_dir / "prefs.jsonl"
    assert sft_path.exists()
    assert prefs_path.exists()

    sft_errors = validate_sft(sft_path)
    prefs_errors = validate_prefs(prefs_path)
    assert not sft_errors
    assert not prefs_errors

    # Ensure preference records align with candidate count
    prefs_lines = prefs_path.read_text(encoding="utf-8").strip().splitlines()
    assert prefs_lines
    for line in prefs_lines:
        record = json.loads(line)
        assert len(record["candidates"]) == 2
        assert len(record["preference"]["scores"]) == 2

