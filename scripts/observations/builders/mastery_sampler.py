from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import random


@dataclass
class MasterySample:
    bucket: str
    snapshot: Dict[str, Any]
    journey_stage: str


def _choose_bucket(distribution: Dict[str, float], rng: random.Random) -> str:
    items = list(distribution.items())
    if not items:
        return "low"
    total = sum(max(0.0, float(p)) for _, p in items) or 1.0
    r = rng.random() * total
    acc = 0.0
    for bucket, prob in items:
        acc += max(0.0, float(prob))
        if r <= acc:
            return bucket
    return items[-1][0]


def _bucket_to_stage(bucket: str) -> str:
    if bucket == "low":
        return "discovery"
    if bucket == "medium":
        return "practice"
    if bucket == "high":
        return "mastery"
    return "unknown"


def sample_mastery(domain_config: Dict[str, Any], rng: random.Random) -> MasterySample:
    """Sample a synthetic mastery snapshot for OBS-01.

    Uses simple low/medium/high buckets configured in the domain YAML.
    """
    dist_cfg = domain_config.get("mastery_distribution") or {}
    distribution = {
        "low": float(dist_cfg.get("low", 0.5)),
        "medium": float(dist_cfg.get("medium", 0.3)),
        "high": float(dist_cfg.get("high", 0.2)),
    }

    values_cfg = domain_config.get("mastery_values") or {}
    mastery_values = {
        "low": float(values_cfg.get("low", 0.2)),
        "medium": float(values_cfg.get("medium", 0.5)),
        "high": float(values_cfg.get("high", 0.8)),
    }

    bucket = _choose_bucket(distribution, rng)
    stage = _bucket_to_stage(bucket)
    mastery_value = mastery_values.get(bucket, 0.2)

    # Simple synthetic attempt/correct counts keyed by bucket
    if bucket == "low":
        attempts, correct = 1, 0
    elif bucket == "medium":
        attempts, correct = 3, 2
    else:  # high
        attempts, correct = 5, 4

    snapshot = {
        "mastery": mastery_value,
        "attempts": attempts,
        "correct": correct,
    }

    return MasterySample(bucket=bucket, snapshot=snapshot, journey_stage=stage)


def sample_mastery_for_user_concept(
    conn: Any,
    user_id: str,
    concept: str,
    domain_config: Dict[str, Any],
    rng: random.Random,
) -> MasterySample:
    """Placeholder for future real mastery sampling via user_concept_mastery.

    OBS-03 only requires a hook for real mastery; for now this simply
    delegates to the synthetic sampler.
    """
    _ = (conn, user_id, concept)
    return sample_mastery(domain_config, rng)
