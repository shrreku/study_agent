from __future__ import annotations

import logging
from typing import Any, Dict, List

from prompts import get as prompt_get, render as prompt_render
from llm import call_llm_json

from .auto_classifiers import ActionSuggestion, PhaseSuggestion, RetrievalSuggestion, TurnSignals
from .context_model import TutorContext


logger = logging.getLogger(__name__)


def _default_retrieval_payload() -> Dict[str, Any]:
    return {
        "strategy": "fresh_retrieval",
        "retrieval_query": None,
        "max_chunks": 8,
        "prefer_sections": [],
    }


def _parse_retrieval(raw: Dict[str, Any]) -> RetrievalSuggestion:
    payload = _default_retrieval_payload()
    if isinstance(raw, dict):
        payload.update({k: raw.get(k) for k in payload.keys() if k in raw})

    strategy = str(payload.get("strategy") or "fresh_retrieval").strip().lower()
    if strategy not in {"reuse_last_chunks", "fresh_retrieval", "refine_query", "no_retrieval"}:
        strategy = "fresh_retrieval"

    query = payload.get("retrieval_query")
    if query is not None:
        try:
            query = str(query).strip() or None
        except Exception:
            query = None

    try:
        max_chunks_val = int(payload.get("max_chunks", 8) or 8)
    except Exception:
        max_chunks_val = 8
    if max_chunks_val < 1:
        max_chunks_val = 1

    sections_raw = payload.get("prefer_sections")
    prefer_sections: List[str] | None = None
    if isinstance(sections_raw, list):
        cleaned: List[str] = []
        for s in sections_raw:
            try:
                v = str(s or "").strip()
            except Exception:
                v = ""
            if v:
                cleaned.append(v)
        prefer_sections = cleaned or None

    return RetrievalSuggestion(
        strategy=strategy,
        retrieval_query=query,
        max_chunks=max_chunks_val,
        prefer_sections=prefer_sections,
    )


def classify_retrieval(
    context: TutorContext,
    turn_signals: TurnSignals,
    phase: PhaseSuggestion,
    action: ActionSuggestion,
) -> RetrievalSuggestion:
    """Classify retrieval/reuse strategy for auto/intelligent mode.

    Uses LLM JSON with robust defaults.
    """
    try:
        template = prompt_get("tutor.retrieval_strategy")
    except Exception:
        logger.exception("tutor_retrieval_prompt_missing")
        return RetrievalSuggestion()

    recent_chunks = []
    try:
        for c in (context.retrieval_chunks or [])[:3]:
            snippet = (c.get("text") or "")[:200]
            if snippet:
                recent_chunks.append(snippet)
    except Exception:
        recent_chunks = []

    observation = {
        "student_message": context.message,
        "intent": context.intent,
        "focus_concept": context.focus_concept or "",
        "current_state": context.current_state.value,
        "suggested_phase": phase.suggested_state,
        "next_action": action.next_action,
        "student_confirmation": turn_signals.student_confirmation,
        "has_previous_chunks": bool(context.retrieval_chunks),
        "recent_chunk_summaries": recent_chunks,
    }

    prompt = prompt_render(template, observation)
    default_payload = _default_retrieval_payload()

    try:
        raw = call_llm_json(prompt, default=default_payload)
    except Exception:
        logger.exception("tutor_retrieval_classifier_error")
        return RetrievalSuggestion()

    try:
        return _parse_retrieval(raw or {})
    except Exception:
        logger.exception("tutor_retrieval_parse_error")
        return RetrievalSuggestion()
