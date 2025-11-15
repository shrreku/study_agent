#!/usr/bin/env python3
"""Validate Tutor RL datasets (SFT + preference JSONL).

Usage examples:

    python scripts/validate_tutor_datasets.py --sft data/sft.jsonl --prefs data/prefs.jsonl
    python scripts/validate_tutor_datasets.py --sft data/sft.jsonl --redact out/redacted

The script performs:
- Pydantic validation against `backend.schemas.tutor_rl`
- Optional JSON Schema validation if `jsonschema` is installed
- Sanity checks for reward totals, weight normalization, candidate counts
- Optional redaction of sensitive identifiers when `--redact` is supplied
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Ensure backend modules are importable when running as a script
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:  # pragma: no cover - optional dependency
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore

from backend.schemas.tutor_rl import PreferenceRecord, SFTRecord


def _load_jsonl(path: Path) -> Iterable[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield idx, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: failed to parse JSON on line {idx}: {exc}") from exc


def _load_schema(schema_path: Path) -> Optional[dict]:
    if not schema_path.exists():
        return None
    try:
        with schema_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _sum_close(values: Iterable[float], target: float, tolerance: float = 0.05) -> bool:
    total = sum(values)
    return math.isclose(total, target, rel_tol=tolerance, abs_tol=tolerance)


def _redact_observation(observation: dict) -> None:
    user = observation.get("user")
    if isinstance(user, dict):
        user.pop("user_id", None)
    session = observation.get("session")
    if isinstance(session, dict):
        session.pop("session_id", None)


def _redact_record(record: dict) -> dict:
    observation = record.get("observation")
    if isinstance(observation, dict):
        _redact_observation(observation)
    return record


def validate_sft(path: Path, *, redact_dir: Optional[Path] = None) -> List[str]:
    schema_path = ROOT_DIR / "schemas" / "tutor_rl" / "sft.schema.json"
    schema = _load_schema(schema_path)
    validator = None
    if jsonschema and schema:
        validator = jsonschema.Draft202012Validator(schema)  # type: ignore

    errors: List[str] = []
    redacted_records: List[str] = []

    for idx, payload in _load_jsonl(path):
        try:
            record = SFTRecord.model_validate(payload)
        except Exception as exc:  # pragma: no cover - error path exercised in QA
            errors.append(f"{path}:{idx} - pydantic validation failed: {exc}")
            continue

        if validator:
            try:
                validator.validate(payload)
            except Exception as exc:  # pragma: no cover - optional
                errors.append(f"{path}:{idx} - schema validation failed: {exc}")

        reward = record.reward
        if not _sum_close(reward.normalized_weights.values(), 1.0, tolerance=0.05):
            errors.append(
                f"{path}:{idx} - normalized weight sum {sum(reward.normalized_weights.values()):.3f} != 1.0"
            )
        if not (0.0 <= reward.total <= 1.0):
            errors.append(f"{path}:{idx} - reward total {reward.total} outside [0,1]")

        if redact_dir:
            payload_copy = json.loads(json.dumps(payload))
            redacted = _redact_record(payload_copy)
            redacted_records.append(json.dumps(redacted, ensure_ascii=False))

    if redact_dir and redacted_records:
        redact_dir.mkdir(parents=True, exist_ok=True)
        out_path = redact_dir / path.name
        out_path.write_text("\n".join(redacted_records) + "\n", encoding="utf-8")

    return errors


def validate_prefs(path: Path, *, redact_dir: Optional[Path] = None) -> List[str]:
    schema_path = ROOT_DIR / "schemas" / "tutor_rl" / "prefs.schema.json"
    schema = _load_schema(schema_path)
    validator = None
    if jsonschema and schema:
        validator = jsonschema.Draft202012Validator(schema)  # type: ignore

    errors: List[str] = []
    redacted_records: List[str] = []

    for idx, payload in _load_jsonl(path):
        try:
            record = PreferenceRecord.model_validate(payload)
        except Exception as exc:
            errors.append(f"{path}:{idx} - pydantic validation failed: {exc}")
            continue

        if validator:
            try:
                validator.validate(payload)
            except Exception as exc:
                errors.append(f"{path}:{idx} - schema validation failed: {exc}")

        candidate_count = len(record.candidates)
        if record.preference.chosen >= candidate_count:
            errors.append(
                f"{path}:{idx} - chosen index {record.preference.chosen} >= candidate count {candidate_count}"
            )
        if len(record.preference.scores) != candidate_count:
            errors.append(
                f"{path}:{idx} - preference scores length {len(record.preference.scores)} != candidate count {candidate_count}"
            )

        if redact_dir:
            payload_copy = json.loads(json.dumps(payload))
            redacted = _redact_record(payload_copy)
            redacted_records.append(json.dumps(redacted, ensure_ascii=False))

    if redact_dir and redacted_records:
        redact_dir.mkdir(parents=True, exist_ok=True)
        out_path = redact_dir / path.name
        out_path.write_text("\n".join(redacted_records) + "\n", encoding="utf-8")

    return errors


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Tutor RL JSONL datasets")
    parser.add_argument("--sft", type=Path, help="Path to SFT JSONL dataset")
    parser.add_argument("--prefs", type=Path, help="Path to preference JSONL dataset")
    parser.add_argument(
        "--redact",
        type=Path,
        default=None,
        help="Optional directory to write redacted copies of provided datasets",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    errors: List[str] = []

    redact_dir = args.redact
    if redact_dir is not None:
        redact_dir = redact_dir.resolve()

    if args.sft:
        if not args.sft.exists():
            errors.append(f"missing SFT file: {args.sft}")
        else:
            errors.extend(validate_sft(args.sft.resolve(), redact_dir=redact_dir))

    if args.prefs:
        if not args.prefs.exists():
            errors.append(f"missing preference file: {args.prefs}")
        else:
            errors.extend(validate_prefs(args.prefs.resolve(), redact_dir=redact_dir))

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        print(f"Validation finished with {len(errors)} error(s).", file=sys.stderr)
        return 1

    print("Validation succeeded: datasets conform to Tutor RL schema.")
    if redact_dir:
        print(f"Redacted copies written to {redact_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

