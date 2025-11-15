from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from llm.common import call_json_chat
from prompts import get as prompt_get, render as prompt_render


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _retrieved_context(observation: Dict[str, Any], limit: int = 3) -> str:
    retrieval = observation.get("retrieval") or {}
    chunks: Sequence[Dict[str, Any]] = retrieval.get("chunks") or []
    lines: List[str] = []
    for idx, chunk in enumerate(chunks[:limit]):
        snippet = (chunk.get("snippet") or "").strip()
        pedagogy = chunk.get("pedagogy_role") or chunk.get("role") or ""
        chunk_id = chunk.get("id") or chunk.get("chunk_id") or f"chunk-{idx+1}"
        if snippet:
            lines.append(f"[{chunk_id} | {pedagogy}] {snippet[:220]}")
    return "\n".join(lines) or "(no grounded snippets available)"


def _cited_ids(observation: Dict[str, Any], response_metadata: Dict[str, Any]) -> List[str]:
    action_block = observation.get("action") or {}
    cited: List[str] = []
    for source_list in (
        response_metadata.get("source_chunk_ids"),
        action_block.get("source_chunk_ids"),
    ):
        if isinstance(source_list, list):
            cited.extend(str(item) for item in source_list if item is not None)
    return list(dict.fromkeys(cited))


def _heuristic_score(
    observation: Dict[str, Any],
    response_text: str,
    response_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    words = response_text.split()
    word_count = len(words)
    clarity = _clamp(word_count / 120.0) if word_count else 0.2

    tutor_block = observation.get("tutor") or {}
    focus_concept = (tutor_block.get("focus_concept") or "").lower()
    accuracy = 0.6
    if focus_concept and focus_concept in response_text.lower():
        accuracy = 0.85
    if "hallucinate" in response_text.lower():
        accuracy = 0.4

    retrieval_block = observation.get("retrieval") or {}
    retrieved_ids = retrieval_block.get("chunk_ids") or []
    cited_ids = _cited_ids(observation, response_metadata)
    overlap = len({cid for cid in cited_ids if cid in retrieved_ids})
    support = 0.5
    if retrieved_ids and cited_ids:
        support = 0.6 + 0.4 * (overlap / max(len(retrieved_ids), 1))
    elif retrieved_ids and not cited_ids:
        support = 0.5
    elif not retrieved_ids:
        support = 0.55

    hallucination_flag = support < 0.5 or accuracy < 0.5
    confidence = _clamp((clarity + accuracy + support) / 3)

    notes_parts: List[str] = []
    if focus_concept:
        notes_parts.append(f"focus={focus_concept}")
    if overlap:
        notes_parts.append(f"cited={overlap}")
    if hallucination_flag:
        notes_parts.append("check grounding")
    return {
        "clarity": round(_clamp(clarity), 4),
        "accuracy": round(_clamp(accuracy), 4),
        "support": round(_clamp(support), 4),
        "hallucination_flag": bool(hallucination_flag),
        "notes": ", ".join(notes_parts)[:200],
        "confidence": round(_clamp(confidence), 4),
    }


def score_with_critic(
    observation: Dict[str, Any],
    response_text: str,
    response_metadata: Optional[Dict[str, Any]] = None,
    *,
    model_hint: Optional[str] = None,
    prompt_set: Optional[str] = None,
    max_tokens: int = 2000,
) -> Dict[str, Any]:
    response_metadata = response_metadata or {}
    template = prompt_get("tutor_rl.critic_score")
    if not template:
        raise RuntimeError("missing prompt tutor_rl.critic_score")

    tutor_block = observation.get("tutor") or {}
    classifier_block = observation.get("classifier") or {}
    action_block = observation.get("action") or {}

    prompt_vars = {
        "focus_concept": tutor_block.get("focus_concept") or tutor_block.get("inference_concept") or "",
        "action_type": action_block.get("type") or "",
        "intent": classifier_block.get("intent") or "unknown",
        "retrieved_context": _retrieved_context(observation),
        "response": response_text.strip() or "(empty response)",
    }
    prompt = prompt_render(template, prompt_vars)

    default_payload = _heuristic_score(observation, response_text, response_metadata)
    result = call_json_chat(
        prompt,
        default=default_payload,
        max_tokens=max_tokens,
        model_hint=model_hint,
    )

    # Ensure keys exist even if model omitted some
    merged = dict(default_payload)
    for key, value in (result or {}).items():
        merged[key] = value

    merged["clarity"] = round(_clamp(float(merged.get("clarity", 0.0))), 4)
    merged["accuracy"] = round(_clamp(float(merged.get("accuracy", 0.0))), 4)
    merged["support"] = round(_clamp(float(merged.get("support", 0.0))), 4)
    merged["confidence"] = round(_clamp(float(merged.get("confidence", 0.0))), 4)
    merged["hallucination_flag"] = bool(merged.get("hallucination_flag", False))
    merged["notes"] = str(merged.get("notes", ""))[:280]
    if prompt_set:
        merged["prompt_set"] = prompt_set
    return merged


def _default_preference_payload(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scores: List[float] = []
    best_idx = 0
    best_score = -1.0
    for idx, candidate in enumerate(candidates):
        reward_total = float(candidate.get("reward", {}).get("total", 0.0)
                             if isinstance(candidate.get("reward"), dict)
                             else 0.0)
        critic_conf = float(candidate.get("critic", {}).get("confidence", 0.0)
                            if isinstance(candidate.get("critic"), dict)
                            else 0.0)
        baseline = 0.4 + 0.5 * reward_total + 0.1 * critic_conf
        baseline = _clamp(baseline)
        scores.append(round(baseline, 4))
        if baseline > best_score:
            best_idx = idx
            best_score = baseline
    if not scores:
        scores = [0.5]
    return {
        "chosen": best_idx,
        "scores": scores,
        "confidence": round(_clamp(best_score if best_score >= 0 else 0.6), 4),
        "reason": "selected highest reward surrogate",
    }


def preference_with_critic(
    observation: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    *,
    model_hint: Optional[str] = None,
    prompt_set: Optional[str] = None,
    max_tokens: int = 2000,
) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("candidates required")
    template = prompt_get("tutor_rl.critic_preference")
    if not template:
        raise RuntimeError("missing prompt tutor_rl.critic_preference")

    tutor_block = observation.get("tutor") or {}
    classifier_block = observation.get("classifier") or {}

    summaries: List[str] = []
    for idx, candidate in enumerate(candidates):
        action_type = (candidate.get("action") or {}).get("type") or "" if isinstance(candidate.get("action"), dict) else ""
        reward_total = None
        if isinstance(candidate.get("reward"), dict):
            reward_total = candidate["reward"].get("total")
        critic_conf = None
        if isinstance(candidate.get("critic"), dict):
            critic_conf = candidate["critic"].get("confidence")
        snippet = str(candidate.get("response", "")).strip()[:180]
        summaries.append(
            f"[{idx}] action={action_type or 'n/a'} reward={reward_total} critic_conf={critic_conf} -> {snippet}"
        )

    prompt_vars = {
        "focus_concept": tutor_block.get("focus_concept") or tutor_block.get("inference_concept") or "",
        "action_types": ", ".join(filter(None, [
            (candidate.get("action") or {}).get("type")
            if isinstance(candidate.get("action"), dict)
            else ""
            for candidate in candidates
        ])) or "",
        "intent": classifier_block.get("intent") or "unknown",
        "candidate_summaries": "\n".join(summaries),
    }
    prompt = prompt_render(template, prompt_vars)

    default_payload = _default_preference_payload(candidates)
    result = call_json_chat(
        prompt,
        default=default_payload,
        max_tokens=max_tokens,
        model_hint=model_hint,
    )

    merged = dict(default_payload)
    if isinstance(result, dict):
        merged.update(result)

    chosen = merged.get("chosen", 0)
    try:
        chosen_idx = int(chosen)
    except Exception:
        chosen_idx = 0
    merged["chosen"] = max(0, min(chosen_idx, len(candidates) - 1))

    scores = merged.get("scores")
    if not isinstance(scores, list) or len(scores) != len(candidates):
        scores = default_payload["scores"]
    merged["scores"] = [round(_clamp(float(val)), 4) for val in scores]
    merged["confidence"] = round(_clamp(float(merged.get("confidence", 0.6))), 4)
    merged["reason"] = str(merged.get("reason", "")).strip()[:200]
    if prompt_set:
        merged["prompt_set"] = prompt_set
    return merged


__all__ = [
    "score_with_critic",
    "preference_with_critic",
]

