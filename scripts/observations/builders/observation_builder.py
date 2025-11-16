from __future__ import annotations

from typing import Any, Dict, List, Tuple
import random
import re

from .mastery_sampler import sample_mastery, sample_mastery_for_user_concept
from .scenario_templates import build_scenario, generate_concept_check_answer
from .retrieval_sampler import sample_chunks_for_concept
from ..validators.obs_validation import validate_observation_entry


def _slugify(text: str, max_len: int = 40) -> str:
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "concept"


def _concept_level_for_bucket(domain_config: Dict[str, Any], bucket: str) -> str:
    mapping = domain_config.get("concept_level_for_bucket") or {}
    return str(mapping.get(bucket) or {
        "low": "beginner",
        "medium": "developing",
        "high": "proficient",
    }.get(bucket, "beginner"))


def _build_single_observation(
    conn,
    *,
    domain_config: Dict[str, Any],
    concept_item: Dict[str, Any],
    rng: random.Random,
    user_index: int,
    obs_index: int,
) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
    """Build one observation entry for a concept.

    Returns (entry_or_none, stats_delta).
    """

    concept = concept_item.get("concept") or "concept"
    difficulty_band = concept_item.get("difficulty_band") or "beginner"
    learning_path = list(concept_item.get("learning_path") or [concept])

    domain_name = str(domain_config.get("domain") or "domain")
    user_id = f"{domain_name}-user-{user_index:03d}"

    use_real_mastery = bool(domain_config.get("use_real_mastery"))
    if use_real_mastery:
        mastery_sample = sample_mastery_for_user_concept(
            conn,
            user_id=user_id,
            concept=concept,
            domain_config=domain_config,
            rng=rng,
        )
    else:
        mastery_sample = sample_mastery(domain_config, rng)
    concept_level = _concept_level_for_bucket(domain_config, mastery_sample.bucket)

    scenario = build_scenario(domain_config, concept, mastery_sample.bucket, rng)

    retrieval = sample_chunks_for_concept(
        conn,
        concept,
        difficulty_band,
        scenario.scenario_type,
        domain_config,
        rng,
    )

    chunk_ids = retrieval.get("chunk_ids") or []
    chunks = retrieval.get("chunks") or []
    if len(chunk_ids) < 2 or len(chunks) < 2:
        return None, {"skipped_no_chunks": 1}

    # Ensure the focus concept appears in at least one snippet
    concept_lc = concept.lower()
    found_in_snippet = False
    for c in chunks:
        snippet = (c.get("snippet") or "").lower()
        if concept_lc and concept_lc in snippet:
            found_in_snippet = True
            break
    if not found_in_snippet:
        return None, {"skipped_no_concept_in_snippet": 1}

    concept_slug = _slugify(concept)
    session_id = f"{domain_name}-sess-{concept_slug}-{obs_index:04d}"

    payload = {
        "message": scenario.message,
        "user_id": user_id,
        "session_id": session_id,
        "resource_id": None,
        "target_concepts": [concept],
    }

    pedagogy_roles = retrieval.get("pedagogy_roles") or []

    observation = {
        "metadata": {"version": 1},
        "user": {
            "message": scenario.message,
            "user_id": user_id,
            "target_concepts": [concept],
        },
        "classifier": {
            "intent": scenario.intent,
            "affect": scenario.affect,
            "concept": concept,
            "confidence": 0.6,
            "needs_escalation": False,
        },
        "tutor": {
            "focus_concept": concept,
            "concept_level": concept_level,
            "inference_concept": concept,
            "learning_path": learning_path,
            "target_concepts": [concept],
            "mastery_snapshot": mastery_sample.snapshot,
        },
        "retrieval": {
            "query": None,
            "chunk_ids": chunk_ids,
            "source_chunk_ids": list(chunk_ids),
            "pedagogy_roles": pedagogy_roles,
            "chunks": chunks,
        },
        "policy": {
            "cold_start": False,
            "consecutive_explains": 0,
            "focus_concept": concept,
        },
        "session": {
            "session_id": session_id,
            "turn_index": 0,
            "resource_id": None,
        },
        # Leave action mostly empty; rollout script will populate as needed.
        "action": {},
    }

    meta: Dict[str, Any] = {
        "domain": domain_name,
        "concept": concept,
        "scenario_type": scenario.scenario_type,
        "mastery_bucket": mastery_sample.bucket,
        "journey_stage": mastery_sample.journey_stage,
        "difficulty_band": difficulty_band,
        "builder_version": "obs-02-llm",
    }

    # OBS-02: optionally attach LLM-generated concept-check answer metadata
    if scenario.scenario_type == "concept_check":
        answer = generate_concept_check_answer(domain_config, concept, mastery_sample.bucket)
        if answer is not None:
            meta["student_answer"] = answer.answer
            meta["answer_correctness"] = answer.answer_correctness
            if answer.misconception_type is not None:
                meta["misconception_type"] = answer.misconception_type

    entry = {
        "id": f"obs-{obs_index:06d}",
        "payload": payload,
        "observation": observation,
        "meta": meta,
    }

    stats_delta: Dict[str, int] = {
        "total": 1,
        f"scenario::{scenario.scenario_type}": 1,
        f"mastery::{mastery_sample.bucket}": 1,
        f"stage::{mastery_sample.journey_stage}": 1,
    }

    if meta.get("answer_correctness"):
        key = f"answer::{meta['answer_correctness']}"
        stats_delta[key] = stats_delta.get(key, 0) + 1
    return entry, stats_delta


def build_observations_for_domain(
    conn,
    domain_config: Dict[str, Any],
    concept_inventory: List[Dict[str, Any]],
    rng: random.Random,
    *,
    max_observations: int | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Generate observation entries for a domain.

    Returns (entries, stats).
    """

    observations_per_concept = int(domain_config.get("observations_per_concept", 1))
    entries: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {}

    # Track concept coverage for summary metrics
    concepts_with_obs: set[str] = set()

    obs_index = 0
    user_index = 1

    for concept_item in concept_inventory:
        for _ in range(observations_per_concept):
            if max_observations is not None and len(entries) >= max_observations:
                break

            obs_index += 1
            entry, delta = _build_single_observation(
                conn,
                domain_config=domain_config,
                concept_item=concept_item,
                rng=rng,
                user_index=user_index,
                obs_index=obs_index,
            )

            # Increment user index periodically to simulate multiple users
            if obs_index % 50 == 0:
                user_index += 1

            # Merge stats (skips and aggregates)
            for key, value in (delta or {}).items():
                stats[key] = stats.get(key, 0) + int(value)

            if entry is not None:
                # Run lightweight validation; skip invalid entries but track reasons
                is_valid, reason = validate_observation_entry(entry)
                if not is_valid:
                    key = f"invalid::{reason or 'unknown'}"
                    stats[key] = stats.get(key, 0) + 1
                    continue

                entries.append(entry)
                try:
                    concept = str(entry.get("meta", {}).get("concept") or "").strip()
                    if concept:
                        concepts_with_obs.add(concept)
                except Exception:
                    pass

        if max_observations is not None and len(entries) >= max_observations:
            break

    stats["total_observations"] = len(entries)
    stats["total_concepts"] = len(concept_inventory)
    stats["concepts_with_observations"] = len(concepts_with_obs)

    return entries, stats
