#!/usr/bin/env python3
"""Offline evaluation harness for Tutor RL datasets.

Compares baseline and preference-optimized responses by recomputing
validator + critic scores and producing aggregate metrics."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.agents.tutor.validators import RewardWeights, ValidatorConfig, score_response
from backend.agents.tutor.critic import score_with_critic


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx} in {path}: {exc}") from exc
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _aggregate_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    validator_config = ValidatorConfig.from_env()
    reward_weights = RewardWeights.from_env()

    count = 0
    component_sums: Dict[str, float] = defaultdict(float)
    critic_sums: Dict[str, float] = defaultdict(float)
    flag_counts: Dict[str, int] = defaultdict(int)

    for record in records:
        observation = record.get("observation") or {}
        response = record.get("response") or ""
        meta = record.get("meta") or {}
        response_metadata = {
            "source_chunk_ids": meta.get("source_chunk_ids") or observation.get("action", {}).get("source_chunk_ids", []),
            "confidence": meta.get("confidence") or observation.get("action", {}).get("confidence"),
        }
        reward_payload = score_response(
            observation,
            response,
            response_metadata,
            weights=reward_weights,
            config=validator_config,
        )
        critic_payload = score_with_critic(
            observation,
            response,
            response_metadata,
            prompt_set=meta.get("prompt_set"),
        )

        count += 1
        for name, payload in reward_payload["components"].items():
            component_sums[name] += float(payload.get("score", 0.0))
            for flag in payload.get("flags", []) or []:
                flag_counts[flag] += 1
        component_sums["total"] += float(reward_payload.get("total", 0.0))
        for key in ("clarity", "accuracy", "support", "confidence"):
            critic_sums[key] += float(critic_payload.get(key, 0.0))
        if critic_payload.get("hallucination_flag"):
            flag_counts["critic_hallucination"] += 1

    if count == 0:
        return {
            "count": 0,
            "component_averages": {},
            "critic_averages": {},
            "flag_rates": {},
        }

    component_avgs = {name: round(value / count, 4) for name, value in component_sums.items()}
    critic_avgs = {name: round(value / count, 4) for name, value in critic_sums.items()}
    flag_rates = {name: round(value / count, 4) for name, value in flag_counts.items()}

    return {
        "count": count,
        "component_averages": component_avgs,
        "critic_averages": critic_avgs,
        "flag_rates": flag_rates,
    }


def _delta(lhs: Dict[str, Any], rhs: Dict[str, Any]) -> Dict[str, Any]:
    deltas: Dict[str, Any] = {}
    lhs_components = lhs.get("component_averages", {})
    rhs_components = rhs.get("component_averages", {})
    for key in sorted(set(lhs_components) | set(rhs_components)):
        deltas[key] = round(rhs_components.get(key, 0.0) - lhs_components.get(key, 0.0), 4)
    return deltas


def _write_report(
    baseline: Dict[str, Any],
    optimized: Dict[str, Any],
    out_dir: Path,
    baseline_label: str,
    optimized_label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "baseline_label": baseline_label,
        "optimized_label": optimized_label,
        "baseline": baseline,
        "optimized": optimized,
        "delta": _delta(baseline, optimized),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        f"# Tutor RL Evaluation", "",
        f"Baseline: **{baseline_label}** ({baseline['count']} samples)",
        f"Optimized: **{optimized_label}** ({optimized['count']} samples)",
        "",
        "## Component Averages",
        "| Metric | Baseline | Optimized | Δ |",
        "| --- | --- | --- | --- |",
    ]
    delta = report["delta"]
    all_keys = sorted(set(baseline["component_averages"].keys()) | set(optimized["component_averages"].keys()))
    for key in all_keys:
        base_val = baseline["component_averages"].get(key, 0.0)
        opt_val = optimized["component_averages"].get(key, 0.0)
        diff = delta.get(key, 0.0)
        md_lines.append(f"| {key} | {base_val:.4f} | {opt_val:.4f} | {diff:+.4f} |")

    md_lines.extend(["", "## Critic Averages", "| Metric | Baseline | Optimized | Δ |", "| --- | --- | --- | --- |"])
    critic_keys = sorted(set(baseline["critic_averages"].keys()) | set(optimized["critic_averages"].keys()))
    for key in critic_keys:
        base_val = baseline["critic_averages"].get(key, 0.0)
        opt_val = optimized["critic_averages"].get(key, 0.0)
        diff = round(opt_val - base_val, 4)
        md_lines.append(f"| {key} | {base_val:.4f} | {opt_val:.4f} | {diff:+.4f} |")

    (out_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tutor RL offline evaluation")
    parser.add_argument("--baseline-sft", type=Path, required=True, help="Baseline SFT JSONL")
    parser.add_argument("--optimized-sft", type=Path, required=True, help="Optimized SFT JSONL")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write evaluation report")
    parser.add_argument("--baseline-label", type=str, default="baseline")
    parser.add_argument("--optimized-label", type=str, default="optimized")
    parser.add_argument("--mock", action="store_true", help="Force mock mode (keeps USE_LLM_MOCK=1)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    original_mock = os.environ.get("USE_LLM_MOCK")
    try:
        if args.mock:
            os.environ["USE_LLM_MOCK"] = "1"
        else:
            if original_mock is not None:
                os.environ.pop("USE_LLM_MOCK", None)

        baseline_records = _load_jsonl(args.baseline_sft)
        optimized_records = _load_jsonl(args.optimized_sft)

        baseline_summary = _aggregate_metrics(baseline_records)
        optimized_summary = _aggregate_metrics(optimized_records)

        _write_report(
            baseline=baseline_summary,
            optimized=optimized_summary,
            out_dir=args.out_dir,
            baseline_label=args.baseline_label,
            optimized_label=args.optimized_label,
        )

        logging.info(
            "Evaluation complete. Δ total reward = %.4f",
            _delta(baseline_summary, optimized_summary).get("total", 0.0),
        )
    finally:
        if original_mock is None:
            os.environ.pop("USE_LLM_MOCK", None)
        else:
            os.environ["USE_LLM_MOCK"] = original_mock
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

