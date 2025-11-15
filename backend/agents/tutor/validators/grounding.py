from __future__ import annotations

from typing import Any, Dict, List

from config.tutor_rl import ValidatorConfig
from .types import ValidatorComponentResult, ValidatorContext


def _id_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item)]
    return []


def grounding_check(context: ValidatorContext, _: ValidatorConfig) -> ValidatorComponentResult:
    observation = context.observation
    response_metadata = context.response_metadata

    retrieval_block: Dict[str, Any] = observation.get("retrieval") or {}
    retrieved_ids = _id_list(retrieval_block.get("chunk_ids"))
    cited_ids = _id_list(response_metadata.get("source_chunk_ids")) or _id_list(
        observation.get("action", {}).get("source_chunk_ids")
    )

    missing: List[str] = []
    unknown: List[str] = []
    for cid in cited_ids:
        if cid not in retrieved_ids:
            unknown.append(cid)

    if retrieved_ids:
        for rid in retrieved_ids:
            if rid not in cited_ids:
                missing.append(rid)

    if cited_ids and not unknown:
        score = 1.0 if not missing else 0.85
    elif cited_ids and unknown:
        score = 0.4
    else:
        score = 0.6 if retrieved_ids else 0.5

    flags: List[str] = []
    if unknown:
        flags.append("unknown_grounding_ids")
    if score < 0.6:
        flags.append("grounding_low")

    details = {
        "retrieved_ids": retrieved_ids,
        "cited_ids": cited_ids,
        "missing_ids": missing,
        "unknown_ids": unknown,
    }

    return ValidatorComponentResult(name="grounding", score=round(score, 4), details=details, flags=flags)


__all__ = ["grounding_check"]

