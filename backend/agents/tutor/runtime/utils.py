"""Shared utilities for tutor runtime."""

import re
from typing import Any, Optional


def safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float, returning None if not possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_single_concept_label(value: Optional[str]) -> Optional[str]:
    """Normalize a possibly noisy concept label to a single canonical name.
    
    Defensive against classifier outputs like "convection, diffusion, Fourier's law".
    We choose the first comma- or slash-separated item.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = [p.strip() for p in re.split(r"[,/]+", text) if p.strip()]
    if not parts:
        return None
    return parts[0]


def looks_like_study_plan_request(message: str) -> bool:
    """Check if message looks like a study plan request."""
    text = (message or "").lower().strip()
    if not text:
        return False
    phrases = [
        "what can i study",
        "what should i study",
        "what should i learn",
        "how should i study",
        "how do i start",
        "where should i start",
        "what to study",
        "help me study",
        "plan my study",
        "plan a short study session",
        "short study session",
        "plan a study session",
    ]
    return any(p in text for p in phrases)


def should_use_multi_step(intent: str, phase: str) -> bool:
    """Gate SRL multi-step execution to assessment-style turns."""
    i = (intent or "").strip().lower()
    if i in {"answer", "reflection"}:
        return True
    if phase == "assessment":
        return True
    return False


def looks_like_confirmation(message: str) -> bool:
    text = (message or "").lower().strip()
    if not text:
        return False
    phrases = [
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "alright",
        "understood",
        "got it",
        "makes sense",
        "sounds good",
        "sure",
        "of course",
        "let's continue",
        "lets continue",
        "continue",
        "go on",
    ]
    return any(p in text for p in phrases)
