from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..summarization import HistorySummarizer


def _turn_to_line(turn: Dict[str, Any]) -> Optional[str]:
    role = turn.get("role") or turn.get("speaker") or "unknown"
    text = turn.get("message") or turn.get("content") or ""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    return f"{role}: {text}"


def summarize_history(
    session_id: str,
    recent_turns: List[Dict[str, Any]],
    current_summary: str = "",
    model_hint: str = "mini",
) -> str:
    lines: List[str] = []
    for t in recent_turns:
        try:
            line = _turn_to_line(t)
        except Exception:
            line = None
        if line:
            lines.append(line)
    summarizer = HistorySummarizer(model_hint=model_hint)
    try:
        return summarizer.summarize(lines, current_summary=current_summary)
    except Exception:
        return current_summary


def build_short_context(
    recent_turns: List[Dict[str, Any]],
    max_chars: int = 1000,
) -> str:
    lines: List[str] = []
    for t in recent_turns:
        try:
            line = _turn_to_line(t)
        except Exception:
            line = None
        if line:
            lines.append(line)
    if not lines:
        return ""
    text = "\n".join(lines)
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]
