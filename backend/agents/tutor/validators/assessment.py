from __future__ import annotations

from typing import Any, Dict, List
import os


def _format_chunks(chunks: List[Dict[str, Any]] | None, limit: int = 3, max_chars: int = 800) -> str:
    chunks = chunks or []
    parts: List[str] = []
    for ch in chunks[:limit]:
        text = ch.get("text_snippet") or ch.get("full_text") or ""
        if not isinstance(text, str):
            text = str(text)
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        parts.append(text)
    return "\n---\n".join(parts)


def assess_student_response(
    student_message: str,
    expected_concept: str,
    reference_chunks: List[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    use_mock = os.getenv("USE_LLM_MOCK", "0").strip() in {"1", "true", "TRUE"}
    if use_mock:
        msg = (student_message or "").lower()
        is_correct = bool(msg and ("correct" in msg or "right" in msg or "agreed" in msg))
        quality = 0.8 if is_correct else 0.4
        return {"correct": is_correct, "quality": quality, "reasoning": "mock"}

    prompt = (
        "Student is learning about: "
        + (expected_concept or "")
        + "\n\nReference material:\n"
        + _format_chunks(reference_chunks)
        + "\n\nStudent said: \""
        + (student_message or "")
        + "\"\n\nAssess and return JSON with keys: correct (true/false/unclear), quality (0-1), reasoning (short)."
    )

    try:
        from llm.common import call_json_chat
    except Exception:
        return {"correct": None, "quality": 0.5, "reasoning": "llm_unavailable"}

    try:
        data = call_json_chat(
            prompt,
            default={"correct": None, "quality": 0.5, "reasoning": "default"},
            system_prompt="Return ONLY minified JSON with keys: correct (true/false/unclear), quality (0-1), reasoning (short).",
            allow_text_fallback=False,
        )
        correct = data.get("correct")
        if isinstance(correct, str):
            c = correct.lower().strip()
            correct = True if c in {"true", "correct", "yes"} else False if c in {"false", "no"} else None
        quality = data.get("quality")
        try:
            quality = float(quality)
        except Exception:
            quality = 0.5
        reasoning = data.get("reasoning") or ""
        return {"correct": correct, "quality": quality, "reasoning": reasoning}
    except Exception:
        return {"correct": None, "quality": 0.5, "reasoning": "llm_error"}
