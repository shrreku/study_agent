#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

from dotenv import find_dotenv, load_dotenv
import yaml

from ..utils.db import get_db_conn
from ..builders.concept_inventory import build_concept_inventory
from ..builders.observation_builder import build_observations_for_domain


def _load_domain_config(domain: str, config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        root = Path(__file__).resolve().parents[3]
        default_path = root / "scripts" / "observations" / "config" / f"domain_{domain}.yaml"
        path = default_path
    else:
        path = config_path

    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a mapping at top level: {path}")

    data.setdefault("domain", domain)
    return data


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build observation specs for RL rollouts")
    parser.add_argument("--domain", required=True, help="Domain name (e.g. heat_transfer)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to domain config YAML (default: scripts/observations/config/domain_<domain>.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output observations.jsonl path (default: datasets/<domain>/<date>/observations.jsonl)",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for deterministic sampling")
    parser.add_argument(
        "--observations-per-concept",
        type=int,
        default=None,
        help="Override observations_per_concept from config",
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        default=None,
        help="Optional global cap on number of observations",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv(find_dotenv(), override=False)

    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    domain = args.domain
    config = _load_domain_config(domain, args.config)

    if args.observations_per_concept is not None:
        config["observations_per_concept"] = int(args.observations_per_concept)

    rng = random.Random(args.seed)

    conn = get_db_conn()
    try:
        concept_inventory = build_concept_inventory(conn, config)
        if not concept_inventory:
            logging.warning("No concepts found for domain %s; nothing to do", domain)
            return 0

        logging.info("Loaded %d concepts for domain %s", len(concept_inventory), domain)

        entries, stats = build_observations_for_domain(
            conn,
            config,
            concept_inventory,
            rng,
            max_observations=args.max_observations,
        )
    finally:
        conn.close()

    if not entries:
        logging.warning("No observations were generated; check config and DB contents")
        return 0

    if args.output is None:
        from datetime import date

        today = date.today().isoformat()
        root = Path(__file__).resolve().parents[3]
        out_dir = root / "datasets" / domain / today
        output_path = out_dir / "observations.jsonl"
    else:
        output_path = args.output

    _write_jsonl(output_path, entries)

    logging.info("Wrote %s (%d observations)", output_path, len(entries))

    # Lightweight summary
    total = stats.get("total_observations", len(entries))
    logging.info("Summary: total=%d concepts=%d", total, stats.get("total_concepts", 0))

    # Report coverage by scenario, mastery bucket, journey stage, and answer correctness
    for key, value in sorted(stats.items()):
        if key.startswith("scenario::") or key.startswith("mastery::") or key.startswith("stage::") or key.startswith("answer::"):
            logging.info("%s=%d", key, value)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
