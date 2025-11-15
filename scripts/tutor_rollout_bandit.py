#!/usr/bin/env python3
"""Tutor bandit rollout CLI script.

Given a list of observations (JSONL or JSON array), force multiple tutor actions
via the `action_override` hook, score responses with validators + critic, and
emit SFT and preference JSONL datasets."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


# Add backend to path for CLI usage
ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agents.tutor.agent import tutor_agent  # type: ignore  # noqa: E402
from agents.tutor.validators import (  # type: ignore  # noqa: E402
    RewardWeights,
    ValidatorConfig,
    score_response,
)
from agents.tutor.critic import (  # type: ignore  # noqa: E402
    preference_with_critic,
    score_with_critic,
)


DEFAULT_ACTIONS: Sequence[str] = (
    "explain",
    "ask",
    "hint",
    "reflect",
    "worked_example",
    "review",
)


@dataclass
class RolloutConfig:
    actions: Sequence[str]
    candidates: int
    prompt_set: Optional[str]
    mock_mode: bool
    seed: Optional[int]
    model_per_candidate: Optional[List[Dict[str, str]]] = None  # [{"action": "...", "model": "..."}, ...]
    critic_model: Optional[str] = None  # Model to use for critic scoring


def _read_observations(path: Path) -> List[Dict[str, Any]]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    if content.lstrip().startswith("["):
        data = json.loads(content)
        if isinstance(data, list):
            return [entry for entry in data if isinstance(entry, dict)]
        raise ValueError("Observation JSON must be a list of objects")
    observations: List[Dict[str, Any]] = []
    for idx, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON on line {idx}: {exc}") from exc
        if isinstance(obj, dict):
            observations.append(obj)
    return observations


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ensure_observation(
    entry: Dict[str, Any],
    base_observation: Dict[str, Any],
    *,
    action_type: str,
    candidate_index: int,
    response_metadata: Dict[str, Any],
    override_applied: bool = False,
    override_type: Optional[str] = None,
) -> Dict[str, Any]:
    obs = copy.deepcopy(base_observation) if base_observation else {}
    metadata = obs.setdefault("metadata", {})
    metadata.setdefault("version", 1)

    payload = _safe_dict(entry.get("payload"))
    user = obs.setdefault("user", {})
    user.setdefault("message", payload.get("message", ""))
    user.setdefault("user_id", payload.get("user_id", "mock-user"))
    user.setdefault("target_concepts", _as_list(payload.get("target_concepts")))

    classifier = obs.setdefault("classifier", {})
    classifier.setdefault("intent", "question")
    classifier.setdefault("affect", "confused")
    classifier.setdefault("concept", user.get("target_concepts", ["concept"])[0] if user.get("target_concepts") else "")
    classifier.setdefault("confidence", 0.5)
    classifier.setdefault("needs_escalation", False)

    tutor = obs.setdefault("tutor", {})
    focus_concept = tutor.setdefault("focus_concept", classifier.get("concept", "")) or classifier.get("concept", "")
    tutor.setdefault("concept_level", "beginner")
    tutor.setdefault("inference_concept", focus_concept)
    tutor.setdefault("learning_path", user.get("target_concepts", [focus_concept]))
    tutor.setdefault("target_concepts", user.get("target_concepts", [focus_concept]))
    tutor.setdefault("mastery_snapshot", {"mastery": 0.2, "attempts": 0})

    retrieval = obs.setdefault("retrieval", {})
    chunk_ids = _as_list(retrieval.get("chunk_ids"))
    if not chunk_ids:
        chunk_ids = ["chunk-mock-1"]
    retrieval["chunk_ids"] = chunk_ids
    retrieval.setdefault("source_chunk_ids", response_metadata.get("source_chunk_ids", chunk_ids))
    retrieval.setdefault("pedagogy_roles", ["definition"])
    chunks = retrieval.setdefault("chunks", [])
    if not chunks:
        snippet = payload.get("message", "Review the focus concept.")
        chunks.append({
            "id": chunk_ids[0],
            "pedagogy_role": "definition",
            "snippet": snippet,
            "page_number": 1,
        })

    policy = obs.setdefault("policy", {})
    policy.setdefault("cold_start", False)
    policy.setdefault("consecutive_explains", 0)
    policy.setdefault("focus_concept", focus_concept)

    session = obs.setdefault("session", {})
    session.setdefault("session_id", payload.get("session_id", f"mock-session-{candidate_index}"))
    session.setdefault("turn_index", candidate_index)
    session.setdefault("resource_id", payload.get("resource_id"))

    action = obs.setdefault("action", {})
    # Respect existing action fields from agent; only set when missing
    action.setdefault("type", action_type)
    action.setdefault("cold_start", False)
    action.setdefault("confidence", response_metadata.get("confidence", 0.6))
    action.setdefault("mastery_delta", None)
    action.setdefault("source_chunk_ids", response_metadata.get("source_chunk_ids", chunk_ids))
    action.setdefault("params", {"concept": focus_concept})
    # Only annotate override flags if an override was explicitly applied (e.g., forced action or mock)
    if override_applied:
        action.setdefault("override_type", override_type or action_type)
        action.setdefault("override_applied", True)
        action.setdefault("applied_override_type", override_type or action_type)

    return obs


def _mock_tutor_turn(
    entry: Dict[str, Any],
    *,
    action_type: str,
    candidate_index: int,
    rng: random.Random,
) -> Dict[str, Any]:
    payload = _safe_dict(entry.get("payload"))
    base_observation = _safe_dict(entry.get("observation"))
    message = payload.get("message") or base_observation.get("user", {}).get("message") or "Review the concept."
    focus = base_observation.get("tutor", {}).get("focus_concept") or payload.get("focus_concept") or "concept"
    snippet = base_observation.get("retrieval", {}).get("chunks", [{}])[0].get("snippet") or message

    response_text = f"[{action_type.upper()}] {focus}: {snippet[:120]}"
    confidence = round(0.55 + 0.1 * rng.random(), 4)
    source_chunk_ids = base_observation.get("retrieval", {}).get("chunk_ids") or [f"chunk-{candidate_index+1}"]

    observation = _ensure_observation(
        entry,
        base_observation,
        action_type=action_type,
        candidate_index=candidate_index,
        response_metadata={"source_chunk_ids": source_chunk_ids, "confidence": confidence},
        override_applied=True,
        override_type=action_type,
    )

    return {
        "response": response_text,
        "confidence": confidence,
        "observation": observation,
        "source_chunk_ids": source_chunk_ids,
    }


def _call_tutor_agent(
    payload: Dict[str, Any],
    *,
    action_type: str,
    model_hint: Optional[str] = None,
) -> Dict[str, Any]:
    keyed_payload = copy.deepcopy(payload)
    keyed_payload["emit_state"] = True
    
    # Only add action_override if action_type is not "auto"
    if action_type != "auto":
        keyed_payload["action_override"] = {"type": action_type}
    
    # Pass model hint to the agent if provided
    if model_hint:
        keyed_payload["model_hint"] = model_hint
    
    # Filter out invalid session_id (e.g., "mock-session-0" is not a valid UUID)
    # Let the agent create a new session instead
    session_id = keyed_payload.get("session_id")
    if session_id and (not isinstance(session_id, str) or session_id.startswith("mock-")):
        keyed_payload.pop("session_id", None)
    
    result = tutor_agent(keyed_payload)

    # Extract observation and optionally enrich it with the tutor's progress trace
    observation = result.get("observation")
    if not observation:
        raise RuntimeError("Tutor agent did not return observation; ensure emit_state=true")

    progress = result.get("progress")
    if progress is not None:
        try:
            obs_copy = copy.deepcopy(observation)
            obs_copy["progress"] = progress
            observation = obs_copy
        except Exception:
            # If anything goes wrong, fall back to the original observation
            pass

    return {
        "response": result.get("response", ""),
        "confidence": result.get("confidence", 0.0),
        "observation": observation,
        "source_chunk_ids": result.get("source_chunk_ids") or observation.get("action", {}).get("source_chunk_ids", []),
        "action_type": observation.get("action", {}).get("type", action_type),
    }


def _prepare_candidates(
    entry: Dict[str, Any],
    *,
    config: RolloutConfig,
    rng: random.Random,
    validator_config: ValidatorConfig,
    reward_weights: RewardWeights,
) -> Dict[str, Any]:
    payload = _safe_dict(entry.get("payload"))
    base_observation = _safe_dict(entry.get("observation"))
    candidates: List[Dict[str, Any]] = []

    for idx in range(config.candidates):
        action_type = config.actions[idx % len(config.actions)]
        
        # Get model from model_per_candidate if provided
        model_hint = None
        if config.model_per_candidate and idx < len(config.model_per_candidate):
            model_hint = config.model_per_candidate[idx].get("model")
            # Also get action from model_per_candidate if provided
            candidate_action = config.model_per_candidate[idx].get("action")
            if candidate_action:
                action_type = candidate_action
        
        if config.mock_mode:
            tutor_result = _mock_tutor_turn(entry, action_type=action_type, candidate_index=idx, rng=rng)
        else:
            if not payload:
                raise ValueError("Non-mock mode requires 'payload' field per observation")
            tutor_result = _call_tutor_agent(payload, action_type=action_type, model_hint=model_hint)
        
        # Get the actual action type from the result (especially important for "auto")
        actual_action_type = tutor_result.get("action_type", action_type)

        # Determine whether an override was applied for this candidate
        is_override = bool(config.mock_mode or (action_type != "auto"))

        observation = _ensure_observation(
            entry,
            tutor_result.get("observation") or base_observation,
            action_type=actual_action_type,
            candidate_index=idx,
            response_metadata={
                "source_chunk_ids": tutor_result.get("source_chunk_ids", []),
                "confidence": tutor_result.get("confidence", 0.0),
            },
            override_applied=is_override,
            override_type=(None if not is_override else action_type),
        )

        reward_payload = score_response(
            observation,
            tutor_result.get("response", ""),
            {"source_chunk_ids": tutor_result.get("source_chunk_ids", [])},
            weights=reward_weights,
            config=validator_config,
        )

        critic_payload = score_with_critic(
            observation,
            tutor_result.get("response", ""),
            {"source_chunk_ids": tutor_result.get("source_chunk_ids", []), "confidence": tutor_result.get("confidence", 0.0)},
            prompt_set=config.prompt_set,
            model_hint=config.critic_model,
        )

        candidate = {
            "action": observation.get("action", {"type": action_type}),
            "response": tutor_result.get("response", ""),
            "reward": reward_payload,
            "critic": critic_payload,
            "observation": observation,
            "meta": {
                "candidate_index": idx,
                "source_chunk_ids": tutor_result.get("source_chunk_ids", []),
                "prompt_set": config.prompt_set,
                "confidence": tutor_result.get("confidence", None),
            },
        }
        candidates.append(candidate)

    preference_payload = preference_with_critic(
        candidates[0]["observation"],
        [{
            "action": c.get("action"),
            "response": c.get("response"),
            "reward": c.get("reward"),
            "critic": c.get("critic"),
        } for c in candidates],
        prompt_set=config.prompt_set,
        model_hint=config.critic_model,
    )

    return {
        "candidates": candidates,
        "preference": preference_payload,
    }


def run_rollout(
    observations: List[Dict[str, Any]],
    *,
    config: RolloutConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(config.seed)
    validator_config = ValidatorConfig.from_env()
    reward_weights = RewardWeights.from_env()

    sft_records: List[Dict[str, Any]] = []
    preference_records: List[Dict[str, Any]] = []

    for entry in observations:
        prepared = _prepare_candidates(
            entry,
            config=config,
            rng=rng,
            validator_config=validator_config,
            reward_weights=reward_weights,
        )
        candidates = prepared["candidates"]
        preference = prepared["preference"]

        for candidate in candidates:
            record = {
                "observation": candidate.get("observation"),
                "action": candidate.get("action"),
                "response": candidate.get("response"),
                "reward": candidate.get("reward"),
                "critic": candidate.get("critic"),
                "meta": candidate.get("meta", {}),
            }
            sft_records.append(record)

        preference_records.append(
            {
                "observation": candidates[0].get("observation"),
                "candidates": [
                    {
                        "action": candidate.get("action"),
                        "response": candidate.get("response"),
                        "reward": candidate.get("reward"),
                        "critic": candidate.get("critic"),
                        "meta": candidate.get("meta", {}),
                    }
                    for candidate in candidates
                ],
                "preference": preference,
            }
        )

    return {"sft": sft_records, "prefs": preference_records}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tutor bandit rollout generator")
    parser.add_argument("--observations", type=Path, required=True, help="Path to observations JSONL/JSON file")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for datasets")
    parser.add_argument("--candidates", type=int, default=3, help="Number of candidate actions per observation")
    parser.add_argument(
        "--actions",
        type=str,
        default=",".join(DEFAULT_ACTIONS),
        help="Comma-separated list of actions to cycle through",
    )
    parser.add_argument("--prompt-set", type=str, default=None, help="Optional prompt set tag (sets PROMPT_SET)")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock tutor responses (no DB required)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic mock mode")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    actions = [a.strip() for a in (args.actions or "").split(",") if a.strip()]
    if not actions:
        raise ValueError("At least one action must be provided")

    if args.prompt_set:
        os.environ["PROMPT_SET"] = args.prompt_set

    if args.mock:
        os.environ.setdefault("USE_LLM_MOCK", "1")

    observations = _read_observations(args.observations)
    if not observations:
        logging.warning("No observations found at %s", args.observations)
        return 0

    rollout_config = RolloutConfig(
        actions=actions,
        candidates=max(1, args.candidates),
        prompt_set=args.prompt_set,
        mock_mode=args.mock,
        seed=args.seed,
    )

    results = run_rollout(observations, config=rollout_config)

    out_dir = args.out_dir.resolve()
    sft_path = out_dir / "sft.jsonl"
    prefs_path = out_dir / "prefs.jsonl"
    _write_jsonl(sft_path, results["sft"])
    _write_jsonl(prefs_path, results["prefs"])

    logging.info("Wrote %s (%d rows)", sft_path, len(results["sft"]))
    logging.info("Wrote %s (%d rows)", prefs_path, len(results["prefs"]))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual CLI entry
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())

