from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from prompts import get as prompt_get, render as prompt_render
from llm import call_json_chat

from .constants import logger
from .utils import format_context_snippets, clamp_confidence, format_concept_list


def _fallback_response_text(concept: Optional[str], chunks: List[Dict[str, str]]) -> str:
    if chunks:
        top = (chunks[0].get("snippet") or "").strip()
        if top:
            return (
                "Here's what your materials say about this topic:\n\n"
                f"{top}\n\n"
                "Let me know if you'd like a different angle."
            )
    return (
        "I couldn't find a grounded snippet yet. Let's review the relevant materials together. "
        "Do you recall which section covers this concept?"
    )


def build_cold_start_question(
    concept: Optional[str],
    chunks: List[Dict[str, str]],
) -> Tuple[str, float, List[str]]:
    default_question = f"What is a key idea about {concept}?" if concept else "What is a key idea here?"
    cold_prompt = prompt_render(
        prompt_get("tutor.ask"),
        {
            "concept": concept or "the concept",
            "level": "beginner",
            "context": "",
        },
    )
    ask_default = {
        "question": default_question,
        "answer": "",
        "confidence": 0.4,
        "options": [],
    }
    try:
        ask_result = call_json_chat(cold_prompt, default=ask_default)
    except Exception:
        logger.exception("tutor_cold_start_prompt_failed")
        ask_result = ask_default
    response_text = str(ask_result.get("question") or default_question).strip()
    confidence = clamp_confidence(ask_result.get("confidence") or 0.4)
    source_ids = [cid for cid in [c.get("id") for c in chunks] if cid]
    return response_text, confidence, source_ids


def build_hint_response(
    concept: Optional[str],
    level: str,
    chunks: List[Dict[str, str]],
) -> Tuple[str, float, List[str]]:
    hint_prompt = prompt_render(
        prompt_get("tutor.hint"),
        {
            "concept": concept or "the concept",
            "level": level,
            "context": format_context_snippets(chunks),
        },
    )
    hint_default = {
        "response": _fallback_response_text(concept, chunks),
        "confidence": 0.5,
    }
    try:
        hint_result = call_json_chat(hint_prompt, default=hint_default)
    except Exception:
        logger.exception("tutor_hint_prompt_failed")
        hint_result = hint_default
    response_text = str(hint_result.get("response") or hint_default["response"]).strip()
    confidence = clamp_confidence(hint_result.get("confidence") or 0.5)
    source_ids = [c.get("id") for c in chunks if c.get("id")]
    return response_text, confidence, source_ids


def build_reflect_response(
    concept: Optional[str],
    level: str,
    chunks: List[Dict[str, str]],
) -> Tuple[str, float, List[str]]:
    reflect_prompt = prompt_render(
        prompt_get("tutor.reflect"),
        {
            "concept": concept or "the concept",
            "level": level,
            "context": format_context_snippets(chunks),
        },
    )
    reflect_default = {
        "response": "Could you summarize what you learned just now?",
        "confidence": 0.6,
    }
    try:
        reflect_result = call_json_chat(reflect_prompt, default=reflect_default)
    except Exception:
        logger.exception("tutor_reflect_prompt_failed")
        reflect_result = reflect_default
    response_text = str(reflect_result.get("response") or reflect_default["response"]).strip()
    confidence = clamp_confidence(reflect_result.get("confidence") or 0.6)
    source_ids = [c.get("id") for c in chunks if c.get("id")]
    return response_text, confidence, source_ids


def build_followup_question(
    concept: Optional[str],
    level: str,
    chunks: List[Dict[str, str]],
) -> Tuple[str, float, List[str]]:
    """Build a follow-up assessment question to check understanding after explaining."""
    default_question = f"Can you explain {concept} in your own words?" if concept else "Can you summarize what you learned?"
    followup_prompt = prompt_render(
        prompt_get("tutor.ask"),
        {
            "concept": concept or "the concept",
            "level": level,
            "context": format_context_snippets(chunks) or "",
        },
    )
    ask_default = {
        "question": default_question,
        "answer": "",
        "confidence": 0.7,
        "options": [],
    }
    try:
        ask_result = call_json_chat(followup_prompt, default=ask_default)
    except Exception:
        logger.exception("tutor_followup_question_failed")
        ask_result = ask_default
    response_text = str(ask_result.get("question") or default_question).strip()
    confidence = clamp_confidence(ask_result.get("confidence") or 0.7)
    source_ids = [cid for cid in [c.get("id") for c in chunks] if cid]
    return response_text, confidence, source_ids


def generate_explain_response(
    concept: Optional[str],
    level: str,
    chunks: List[Dict[str, str]],
    fallback_response: Optional[str] = None,
) -> Tuple[str, float, List[str], Optional[str]]:
    if not chunks:
        response = (
            "I couldn't find a grounded snippet yet. Let's review your materials first. "
            "Could you point me to the chapter or section?"
        )
        return response, 0.2, [], concept

    context_block = format_context_snippets(chunks)
    default_payload = {
        "response": fallback_response or _fallback_response_text(concept, chunks),
        "confidence": 0.5,
    }
    prompt = prompt_render(
        prompt_get("tutor.explain"),
        {
            "concept": concept or "the concept",
            "level": level,
            "context": context_block,
        },
    )
    try:
        result = call_json_chat(prompt, default=default_payload)
    except Exception:
        logger.exception("tutor_explain_prompt_failed")
        result = default_payload

    response_text = str(result.get("response") or default_payload["response"]).strip()
    if not response_text:
        response_text = default_payload["response"]
    confidence = clamp_confidence(result.get("confidence")) or default_payload["confidence"]
    source_ids = [c.get("id") for c in chunks if c.get("id")]
    return response_text, confidence, source_ids, concept
