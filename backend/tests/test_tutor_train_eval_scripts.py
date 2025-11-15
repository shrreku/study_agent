from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
for candidate in (BACKEND_DIR, ROOT_DIR, ROOT_DIR / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from api.rl_tools import RolloutRequest, api_rollout
from scripts.train_tutor_pref import main as train_main
from scripts.eval_tutor_bandit import main as eval_main


@pytest.fixture()
def prefs_path() -> Path:
    return ROOT_DIR / "sample" / "mock_rollout" / "prefs.jsonl"


@pytest.fixture()
def sft_path() -> Path:
    return ROOT_DIR / "sample" / "mock_rollout" / "sft.jsonl"


def test_train_pref_mock(tmp_path: Path, prefs_path: Path):
    output_dir = tmp_path / "adapter"
    os.environ["USE_LLM_MOCK"] = "1"
    exit_code = train_main(
        [
            "--prefs",
            str(prefs_path),
            "--output-dir",
            str(output_dir),
            "--method",
            "dpo",
            "--mock",
            "--seed",
            "42",
        ]
    )
    assert exit_code == 0
    metrics_path = output_dir / "metrics.json"
    adapter_path = output_dir / "adapter.bin"
    assert metrics_path.exists()
    assert adapter_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["method"] == "dpo"
    assert metrics["mode"] == "mock"


def test_eval_bandit(tmp_path: Path, sft_path: Path):
    os.environ["USE_LLM_MOCK"] = "1"
    out_dir = tmp_path / "report"
    exit_code = eval_main(
        [
            "--baseline-sft",
            str(sft_path),
            "--optimized-sft",
            str(sft_path),
            "--out-dir",
            str(out_dir),
            "--baseline-label",
            "baseline",
            "--optimized-label",
            "optimized",
            "--mock",
        ]
    )
    assert exit_code == 0
    report_json = out_dir / "report.json"
    report_md = out_dir / "report.md"
    assert report_json.exists()
    assert report_md.exists()
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["baseline"]["count"] > 0
    assert report["optimized"]["count"] > 0


def test_rollout_api_endpoint():
    os.environ["USE_LLM_MOCK"] = "1"
    observations = json.loads((ROOT_DIR / "sample" / "mock_observations.jsonl").read_text(encoding="utf-8").splitlines()[0])
    other = json.loads((ROOT_DIR / "sample" / "mock_observations.jsonl").read_text(encoding="utf-8").splitlines()[1])

    request = RolloutRequest(
        observations=[observations, other],
        actions=["explain", "ask"],
        candidates=2,
        mock=True,
        prompt_set="baseline",
        seed=123,
    )

    response = api_rollout(request)
    assert len(response.sft) == 4
    assert len(response.prefs) == 2

