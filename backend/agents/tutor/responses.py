from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import textwrap
import os

from prompts import get as prompt_get, render as prompt_render
from llm import call_json_chat

from .constants import logger
from .utils import format_context_snippets, clamp_confidence, format_concept_list
from .tools.example_generator import ExampleGenerator, ExampleRequest
from .planning import TutorPlan


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


def build_prerequisite_review_prompt(
    target_concept: Optional[str],
    missing_prereqs: List[str],
    chunks: List[Dict[str, str]],
) -> Tuple[str, float, List[str]]:
    """Generate a supportive, prerequisite-aware review response.

    Uses prompt 'tutor.prereq_review' and returns (response_text, confidence, source_ids).
    """
    if not missing_prereqs:
        return ("", 0.0, [])
    first_prereq = missing_prereqs[0]
    prereq_names = ", ".join(missing_prereqs[:2])
    prompt = prompt_render(
        prompt_get("tutor.prereq_review"),
        {
            "target_concept": target_concept or "the target concept",
            "prereq_names": prereq_names,
            "first_prereq": first_prereq,
            "context": format_context_snippets(chunks),
        },
    )
    default_payload = {
        "response": (
            f"Great question about {target_concept or 'the topic'}! "
            f"Before we dive in, let's quickly review {first_prereq} using your materials."
        ),
        "confidence": 0.6,
    }
    try:
        result = call_json_chat(prompt, default=default_payload)
    except Exception:
        logger.exception("tutor_prereq_review_prompt_failed")
        result = default_payload
    response_text = str(result.get("response") or default_payload["response"]).strip()
    confidence = clamp_confidence(result.get("confidence") or 0.6)
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


def build_override_question(
    concept: Optional[str],
    level: str,
    chunks: List[Dict[str, str]],
    *,
    difficulty: Optional[str] = None,
    question_type: Optional[str] = None,
) -> Tuple[str, float, List[str]]:
    """Deterministic formative question honoring override hints."""
    concept_label = concept or "this concept"
    type_label = (question_type or "question").replace("_", " ").strip()
    difficulty_label = (difficulty or "").replace("_", " ").strip()
    context_block = format_context_snippets(chunks)
    prompt_header = f"{type_label.title()} for {concept_label}".strip()
    if difficulty_label:
        prompt_header += f" ({difficulty_label})"
    if context_block:
        response_text = f"{prompt_header}:\n{context_block.splitlines()[0].strip()}\nWhat step should come next?"
        confidence = 0.7
    else:
        response_text = f"{prompt_header}: Let's recall one key fact about {concept_label}."
        confidence = 0.6
    source_ids = [c.get("id") for c in chunks if c.get("id")]
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


def generate_explain_response_with_plan(
    plan: TutorPlan,
    concept: Optional[str],
    level: str,
    chunks: List[Dict[str, str]],
    fallback_response: Optional[str] = None,
) -> Tuple[str, float, List[str], Optional[str]]:
    """Generate an explanation guided by an internal plan.

    Uses prompt 'tutor.explain_with_plan' and includes plan thinking and rationale.
    """
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
        prompt_get("tutor.explain_with_plan"),
        {
            "concept": concept or "the concept",
            "level": level,
            "context": context_block,
            "plan_thinking": plan.thinking,
            "plan_rationale": plan.action_rationale,
            "pedagogy_focus": ", ".join(plan.pedagogy_focus or []),
        },
    )
    try:
        result = call_json_chat(prompt, default=default_payload)
    except Exception:
        logger.exception("tutor_explain_with_plan_prompt_failed")
        result = default_payload

    response_text = str(result.get("response") or default_payload["response"]).strip()
    if not response_text:
        response_text = default_payload["response"]
    confidence = clamp_confidence(result.get("confidence")) or default_payload["confidence"]
    source_ids = [c.get("id") for c in chunks if c.get("id")]
    return response_text, confidence, source_ids, concept


def build_worked_example_response(
    concept: Optional[str],
    level: str,
    chunks: List[Dict[str, str]],
) -> Tuple[str, float, List[str]]:
    """Craft a deterministic worked example summary grounded in retrieved chunks."""
    use_generator = (os.getenv("TUTOR_EXAMPLE_GENERATION_ENABLED", "false").strip().lower() == "true")
    context_block = format_context_snippets(chunks)

    if use_generator and (not context_block or not chunks):
        try:
            gen = ExampleGenerator()
            req = ExampleRequest(
                concept=concept or "the concept",
                difficulty=level,
                context_type=os.getenv("TUTOR_EXAMPLE_DEFAULT_CONTEXT", "everyday"),
                student_background=os.getenv("TUTOR_STUDENT_BACKGROUND", "general"),
                prerequisites_mastered=None,
                avoid_patterns=None,
            )
            result = gen.generate_example(req, grounding_chunks=chunks)
            min_rel = getattr(gen, "min_relevance", 0.6)
            min_conf = getattr(gen, "min_confidence", 0.5)
            if result.relevance_score >= min_rel and result.confidence >= min_conf:
                text = f"Example: {result.example_text}\n\nWhy this helps: {result.explanation}"
                source_ids = [c.get("id") for c in chunks if c.get("id")]
                return text, float(result.confidence), source_ids
        except Exception:
            logger.exception("tutor_contextual_example_failed")

    concept_label = concept or "this concept"
    lines: List[str] = []
    if context_block:
        lines.append(f"Let's step through a worked example on {concept_label}.")
        snippet = context_block.splitlines()[0].strip()
        if snippet:
            lines.append(f"Given: {snippet}")
    else:
        lines.append(f"Let's outline a worked example for {concept_label} even without detailed context.")
    lines.extend(
        [
            "1. Identify the known quantities or facts from the prompt.",
            "2. Apply the relevant principle for the concept.",
            "3. Compute or reason through the result step by step.",
            "4. Double-check the conclusion and relate it back to the concept.",
        ]
    )
    response_text = "\n".join(lines)
    confidence = 0.65 if context_block else 0.55
    source_ids = [c.get("id") for c in chunks if c.get("id")]
    return response_text, confidence, source_ids


def build_review_response(
    concept: Optional[str],
    level: str,
    chunks: List[Dict[str, str]],
) -> Tuple[str, float, List[str]]:
    """Provide a concise review summary grounded in context snippets."""
    concept_label = concept or "our topic"
    bullet_items: List[str] = []
    for chunk in chunks[:3]:
        snippet = (chunk.get("snippet") or "").strip()
        if snippet:
            bullet_items.append(snippet[:160])
    if not bullet_items:
        bullet_items = [f"Revisit the core definition of {concept_label}.", "Note the key relationships and examples discussed."]
    intro = f"Quick review for {concept_label} ({level} level):"
    bullets = "\n".join(f"- {textwrap.shorten(item, width=140, placeholder='…')}" for item in bullet_items)
    response_text = f"{intro}\n{bullets}"
    confidence = 0.6 if chunks else 0.5
    source_ids = [c.get("id") for c in chunks if c.get("id")]
    return response_text, confidence, source_ids
