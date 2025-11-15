#!/usr/bin/env python3
"""Preference-optimization training harness for Tutor RL datasets.

Supports DPO/ORPO/KTO using TRL when available, with a fast `--mock`
mode that produces LoRA-style artifacts for CI smoke tests."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


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
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _preference_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    chosen_scores: List[float] = []
    margins: List[float] = []
    for rec in records:
        pref = rec.get("preference") or {}
        chosen = pref.get("chosen", 0)
        scores = pref.get("scores") or []
        if not scores:
            continue
        chosen_idx = max(0, min(int(chosen), len(scores) - 1))
        chosen_score = float(scores[chosen_idx])
        chosen_scores.append(chosen_score)
        if len(scores) >= 2:
            sorted_scores = sorted(float(s) for s in scores)
            margins.append(sorted_scores[-1] - sorted_scores[-2])
    avg_chosen = sum(chosen_scores) / len(chosen_scores) if chosen_scores else 0.0
    avg_margin = sum(margins) / len(margins) if margins else 0.0
    return {
        "num_pairs": len(records),
        "avg_chosen_score": round(avg_chosen, 4),
        "avg_margin": round(avg_margin, 4),
    }


def _mock_train(
    method: str,
    records: List[Dict[str, Any]],
    output_dir: Path,
    base_model: str,
    seed: Optional[int],
) -> Dict[str, Any]:
    rng = random.Random(seed)
    stats = _preference_statistics(records)
    mock_loss_start = 1.0 - 0.1 * rng.random()
    mock_loss_end = mock_loss_start * (0.5 + 0.2 * rng.random())

    metrics = {
        "method": method,
        "base_model": base_model,
        "dataset_size": stats["num_pairs"],
        "avg_chosen_score": stats["avg_chosen_score"],
        "avg_margin": stats["avg_margin"],
        "loss_start": round(mock_loss_start, 4),
        "loss_end": round(mock_loss_end, 4),
        "seed": seed,
        "mode": "mock",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    adapter_config = {
        "lora_target": "tutor_head",
        "rank": 8,
        "alpha": 16,
        "method": method,
        "base_model": base_model,
        "generated_by": "mock_train",
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2), encoding="utf-8"
    )
    (output_dir / "adapter.bin").write_bytes(b"mock-adapter-weights")

    logging.info("Mock training complete; artifacts written to %s", output_dir)
    return metrics


def _trl_available() -> bool:
    try:  # pragma: no cover - import check
        import transformers  # type: ignore[import]  # noqa: F401
        import trl  # type: ignore[import]  # noqa: F401
        import peft  # type: ignore[import]  # noqa: F401
        return True
    except Exception:
        return False


def _real_train(
    method: str,
    records: List[Dict[str, Any]],
    output_dir: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: Optional[int],
    lora_r: int,
    lora_alpha: int,
    mock_fallback: bool,
) -> Dict[str, Any]:
    if not _trl_available() or mock_fallback:
        logging.warning("TRL stack not available or overridden; falling back to mock training")
        return _mock_train(method, records, output_dir, base_model, seed)

    logging.info("Starting %s training with TRL (dataset=%d)", method.upper(), len(records))
    try:  # pragma: no cover - heavy training path not exercised in CI
        import torch  # type: ignore[import]
        from datasets import Dataset  # type: ignore[import]
        from peft import LoraConfig  # type: ignore[import]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
        from trl import DPOTrainer, ORPOTrainer, KTOTrainer  # type: ignore[import]
    except Exception as exc:  # pragma: no cover
        logging.error("Failed to import TRL stack: %s", exc)
        return _mock_train(method, records, output_dir, base_model, seed)

    torch.manual_seed(seed or 42)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    formatted_records = []
    for rec in records:
        pref = rec.get("preference") or {}
        candidates = rec.get("candidates") or []
        chosen_idx = pref.get("chosen", 0)
        chosen_idx = max(0, min(int(chosen_idx), len(candidates) - 1))
        if len(candidates) < 2:
            continue
        pos = candidates[chosen_idx]
        neg = candidates[1 - chosen_idx]
        obs = rec.get("observation")
        prompt = json.dumps(obs, ensure_ascii=False)
        formatted_records.append(
            {
                "prompt": prompt,
                "chosen": pos.get("response", ""),
                "rejected": neg.get("response", ""),
            }
        )

    if not formatted_records:
        logging.warning("No usable pairs after preprocessing; falling back to mock")
        return _mock_train(method, records, output_dir, base_model, seed)

    dataset = Dataset.from_list(formatted_records)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(base_model)
    trainer_cls = {
        "dpo": DPOTrainer,
        "orpo": ORPOTrainer,
        "kto": KTOTrainer,
    }[method]

    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=lora_config,
        args=dict(
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            logging_steps=1,
            output_dir=str(output_dir),
            report_to=[],
        ),
    )
    trainer.train()
    trainer.save_model(str(output_dir))

    metrics = {
        "method": method,
        "base_model": base_model,
        "dataset_size": len(dataset),
        "loss_start": None,
        "loss_end": None,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "mode": "trl",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logging.info("Training complete; artifacts in %s", output_dir)
    return metrics


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tutor preference optimization trainer")
    parser.add_argument("--prefs", type=Path, required=True, help="Path to preference dataset JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store LoRA artifacts")
    parser.add_argument("--method", choices=["dpo", "orpo", "kto"], default="dpo")
    parser.add_argument("--base-model", type=str, default="gpt2", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Skip TRL and emit mock artifacts")
    parser.add_argument("--force-mock", action="store_true", help="Alias for --mock (backwards compat)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    records = _load_jsonl(args.prefs)
    os.environ.setdefault("USE_LLM_MOCK", "0")  # ensure deterministic validator usage downstream

    if args.mock or args.force_mock:
        _mock_train(args.method, records, args.output_dir, args.base_model, args.seed)
    else:
        _real_train(
            method=args.method,
            records=records,
            output_dir=args.output_dir,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            mock_fallback=False,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

