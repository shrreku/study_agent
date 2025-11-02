from __future__ import annotations

from typing import Any, Dict, List, Optional


def normalize_concepts(raw: Optional[Any]) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    try:
        return [str(raw).strip()]
    except Exception:
        return []


def format_concept_list(concepts: List[str]) -> str:
    if not concepts:
        return "None"
    return ", ".join(concepts[:6])


def format_context_snippets(chunks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        snippet = (chunk.get("snippet") or "").strip()
        if not snippet:
            continue
        label = chunk.get("id") or "?"
        parts.append(f"[Chunk {idx} | {label}] {snippet}")
    return "\n\n".join(parts)


def clamp_confidence(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return 0.0
    if num < 0.0:
        return 0.0
    if num > 1.0:
        return 1.0
    return num
