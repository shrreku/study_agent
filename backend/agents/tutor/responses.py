from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _extract_concept(args: List[Any], kwargs: Dict[str, Any]) -> Optional[str]:
    if "concept" in kwargs:
        return kwargs.get("concept")  # type: ignore[return-value]
    if args:
        # In most call sites the first positional arg is the concept id/label
        value = args[0]
        return str(value) if value is not None else None
    return None


def _build_text(prefix: str, concept: Optional[str]) -> str:
    label = concept or "this topic"
    return f"{prefix} {label}."


def generate_explain_response(*args: Any, **kwargs: Any) -> Tuple[str, float, List[str], Optional[str]]:
    """Lightweight fallback for explanation responses.

    Returns a tuple of (text, confidence, source_chunk_ids, inferred_concept).
    """

    concept = _extract_concept(list(args), kwargs)
    text = _build_text("Let me explain", concept)
    return text, 0.5, [], concept


def build_review_response(*args: Any, **kwargs: Any) -> Tuple[str, float, List[str]]:
    """Fallback for review / summary style responses."""

    concept = _extract_concept(list(args), kwargs)
    text = _build_text("Let's review", concept)
    return text, 0.5, []


def build_worked_example_response(*args: Any, **kwargs: Any) -> Tuple[str, float, List[str]]:
    """Fallback for worked example responses."""

    concept = _extract_concept(list(args), kwargs)
    text = _build_text("Here's a worked example for", concept)
    return text, 0.5, []


def build_followup_question(*args: Any, **kwargs: Any) -> Tuple[str, float, List[str]]:
    """Fallback for guided practice / follow-up question responses."""

    concept = _extract_concept(list(args), kwargs)
    text = _build_text("Try this question about", concept)
    return text, 0.5, []


def build_mcq_assessment_question(*args: Any, **kwargs: Any) -> Tuple[str, float, List[str], Dict[str, Any]]:
    """Fallback MCQ generator used by assessment flows.

    Returns (text, confidence, source_chunk_ids, mcq_payload).
    """

    concept = _extract_concept(list(args), kwargs)
    text = _build_text("Quick check on", concept)
    mcq_payload: Dict[str, Any] = {
        "question": text,
        "options": [
            {"id": "A", "text": "I understand this well."},
            {"id": "B", "text": "I am somewhat unsure."},
            {"id": "C", "text": "I do not understand this yet."},
        ],
        "correct_option_id": "A",
    }
    return text, 0.5, [], mcq_payload


def build_reflect_response(*args: Any, **kwargs: Any) -> Tuple[str, float, List[str]]:
    """Fallback for reflection prompts."""

    concept = _extract_concept(list(args), kwargs)
    text = _build_text("Take a moment to reflect on", concept)
    return text, 0.5, []


def build_cold_start_question(*args: Any, **kwargs: Any) -> Tuple[str, float, List[str]]:
    """Fallback used when no retrieval context is available."""

    concept = _extract_concept(list(args), kwargs)
    text = _build_text("To get started, what do you know about", concept)
    return text, 0.5, []


def build_hint_response(*args: Any, **kwargs: Any) -> Tuple[str, float, List[str]]:
    """Fallback for hint-style responses."""

    concept = _extract_concept(list(args), kwargs)
    text = _build_text("Here's a hint about", concept)
    return text, 0.5, []


def build_override_question(*args: Any, **kwargs: Any) -> Tuple[str, float, List[str]]:
    """Generic override question helper.

    Not heavily used in the environment MVP, but provided for
    compatibility with legacy handlers.
    """

    concept = _extract_concept(list(args), kwargs)
    text = _build_text("What would you like to focus on for", concept)
    return text, 0.5, []
