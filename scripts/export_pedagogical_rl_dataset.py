#!/usr/bin/env python
from __future__ import annotations

"""Export pedagogical tutor RL trajectories from JSONL logs.

This script is intentionally simple and log-format-agnostic. It expects an
input JSONL file containing log records where each line is a JSON object.
Any record that contains a top-level "pedagogical_step" key (matching the
shape produced by rl_pedagogical_logging.PedagogicalTutorStepEvent) will be
emitted as a standalone trajectory record to the output JSONL file.

Usage:
    python scripts/export_pedagogical_rl_dataset.py \
        --input tutor_logs.jsonl \
        --output pedagogical_steps.jsonl
"""

import argparse
import json
from typing import Any, Dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pedagogical tutor RL dataset from logs.")
    parser.add_argument("--input", required=True, help="Path to input JSONL log file")
    parser.add_argument("--output", required=True, help="Path to output JSONL dataset file")
    return parser.parse_args()


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main() -> None:
    args = _parse_args()

    count_in = 0
    count_out = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for record in _iter_jsonl(args.input):
            count_in += 1
            step: Dict[str, Any] = record.get("pedagogical_step") or {}
            if not isinstance(step, dict):
                continue
            out_f.write(json.dumps(step, ensure_ascii=False) + "\n")
            count_out += 1

    print(f"Read {count_in} log records, exported {count_out} pedagogical step records to {args.output}")


if __name__ == "__main__":
    main()
