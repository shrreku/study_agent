from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import textwrap
import os
import re
import uuid

from prompts import get as prompt_get, render as prompt_render
from llm import call_json_chat

from .constants import logger
from .utils import format_context_snippets, clamp_confidence, format_concept_list
from .tools.example_generator import ExampleGenerator, ExampleRequest
from .planning import TutorPlan


def clean_snippet_for_display(snippet: str, max_length: int = 400) -> str:
    """Clean OCR artifacts and format snippet for student display.
    
    Removes common OCR artifacts like (cid:4), normalizes whitespace,
    and truncates if too long.
    """
    if not snippet:
        return ""
    
    # Remove common OCR artifacts
    cleaned = snippet.replace("(cid:4)", "§")  # Section symbol
    cleaned = re.sub(r'\(cid:\d+\)', '', cleaned)  # Remove all (cid:N) patterns
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
    cleaned = cleaned.strip()
    
    # Truncate if too long
    if len(cleaned) > max_length:
        # Try to break at sentence or word boundary
        truncated = cleaned[:max_length]
        last_period = truncated.rfind('.')
        last_space = truncated.rfind(' ')
        
        if last_period > max_length * 0.7:  # Period found in last 30%
            cleaned = cleaned[:last_period + 1] + "..."
        elif last_space > max_length * 0.7:  # Space found in last 30%
            cleaned = cleaned[:last_space] + "..."
        else:
            cleaned = truncated + "..."
    
    return cleaned


def _fallback_response_text(concept: Optional[str], chunks: List[Dict[str, str]]) -> str:
    if chunks:
        top = (chunks[0].get("snippet") or "").strip()
        if top:
            cleaned_snippet = clean_snippet_for_display(top)
            return (
                "Here's what your materials say about this topic:\n\n"
                f"{cleaned_snippet}\n\n"
                "Let me know if you'd like a different angle."
            )
    return (
        "I couldn't find a grounded snippet yet. Let's review the relevant materials together. "
        "Do you recall which section covers this concept?"
    )


def _format_mastery_snapshot(mastery_map: Dict[str, Dict[str, object]], limit: int = 5) -> str:
    if not mastery_map:
        return "No mastery data yet."
    lines: List[str] = []
    for concept, data in list(mastery_map.items())[:limit]:
        try:
            mastery = float((data or {}).get("mastery", 0.0) or 0.0)  # type: ignore[arg-type]
        except Exception:
            mastery = 0.0
        lines.append(f"- {concept}: {mastery:.2f}")
    return "\n".join(lines)


def build_orientation_response(
    message: str,
    focus_concept: Optional[str],
    learning_targets: List[str],
    learning_path: List[str],
    mastery_map: Dict[str, Dict[str, object]],
    chunks: List[Dict[str, str]],
) -> Tuple[str, float, List[str], Optional[str]]:
    orientation_prompt = prompt_render(
        prompt_get("tutor.orient"),
        {
            "student_message": message,
            "focus_concept": focus_concept or "",
            "target_concepts": format_concept_list(learning_targets),
            "learning_path": ", ".join(learning_path or []),
            "mastery_snapshot": _format_mastery_snapshot(mastery_map),
            "overview_snippets": format_context_snippets(chunks),
        },
    )

    default_recommended = focus_concept or (learning_path[0] if learning_path else (learning_targets[0] if learning_targets else ""))
    default_payload = {
        "response": (
            "Let's plan our study session. "
            "Based on your course materials, we can work through a short sequence of key topics. "
            "I'll suggest a starting point and a couple of next steps, then you can choose where to begin."
        ),
        "recommended_concept": default_recommended,
        "confidence": 0.7,
    }

    try:
        result = call_json_chat(orientation_prompt, default=default_payload)
    except Exception:
        logger.exception("tutor_orientation_prompt_failed")
        result = default_payload

    response_text = str(result.get("response") or default_payload["response"]).strip()
    confidence = clamp_confidence(result.get("confidence") or default_payload["confidence"])
    recommended_concept = str(result.get("recommended_concept") or default_recommended).strip() or None
    source_ids = [c.get("id") for c in chunks if c.get("id")]
    return response_text, confidence, source_ids, recommended_concept


def build_cold_start_question(
    concept: Optional[str],
    chunks: List[Dict[str, str]],
    message: Optional[str] = None,
) -> Tuple[str, float, List[str]]:
    default_question = f"What is a key idea about {concept}?" if concept else "What is a key idea here?"
    cold_prompt = prompt_render(
        prompt_get("tutor.ask"),
        {
            "concept": concept or "the concept",
            "level": "beginner",
            "context": "",
            "student_message": message or "",
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
    message: Optional[str] = None,
) -> Tuple[str, float, List[str]]:
    hint_prompt = prompt_render(
        prompt_get("tutor.hint"),
        {
            "concept": concept or "the concept",
            "level": level,
            "context": format_context_snippets(chunks),
            "student_message": message or "",
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
    message: Optional[str] = None,
    history: Optional[str] = None,
) -> Tuple[str, float, List[str]]:
    reflect_prompt = prompt_render(
        prompt_get("tutor.reflect"),
        {
            "concept": concept or "the concept",
            "level": level,
            "context": format_context_snippets(chunks),
            "student_message": message or "",
            "recent_history": history or "",
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
    message: Optional[str] = None,
) -> Tuple[str, float, List[str]]:
    """Build a simple, open follow-up question to check understanding.

    For assessment turns we want the student to produce their own explanation,
    not answer a multiple-choice question. To keep behaviour predictable (and
    avoid MCQ-style prompts), this helper favours an open, free-form question
    with a grounded snippet and uses the LLM only to phrase the check
    question, falling back to a deterministic variant if needed.
    """

    # Prefer a grounded question that invites the student to explain in their own words.
    if concept:
        default_question = f"Can you explain {concept} in your own words?"
    else:
        default_question = "Can you summarize what you understand so far?"

    # Optionally anchor the question with the first retrieved snippet.
    snippet = ""
    if chunks:
        try:
            snippet = (chunks[0].get("snippet") or "").strip()
        except Exception:
            snippet = ""

    # Ask the LLM to phrase an open-ended check question while avoiding MCQs.
    prompt_parts = [
        "You are a helpful tutor.",
        f"Concept: {concept or 'this concept'}.",
        f"Level: {level}.",
    ]
    if snippet:
        prompt_parts.append("Here is a snippet from the student's notes:")
        prompt_parts.append(snippet)
    if message:
        prompt_parts.append("Student message:")
        prompt_parts.append(message)
    prompt_parts.append(
        "Return JSON with fields 'question' and 'confidence'. Ask ONE open-ended "
        "question that gets the student to explain in their own words. Do NOT "
        "use multiple choice or options."
    )
    prompt = "\n\n".join(prompt_parts)

    ask_default = {
        "question": default_question,
        "confidence": 0.7,
    }
    try:
        ask_result = call_json_chat(prompt, default=ask_default)
    except Exception:
        logger.exception("tutor_followup_question_prompt_failed")
        ask_result = ask_default

    question_text = str(ask_result.get("question") or default_question).strip()
    if snippet:
        cleaned_snippet = clean_snippet_for_display(snippet)
        response_text = (
            f"Based on this part of your notes:\n\n{cleaned_snippet}\n\n"
            f"{question_text}"
        )
    else:
        response_text = question_text or default_question

    confidence = clamp_confidence(ask_result.get("confidence") or 0.7)
    source_ids = [cid for cid in [c.get("id") for c in chunks] if cid]
    return response_text, confidence, source_ids


def build_mcq_assessment_question(
    concept: Optional[str],
    level: str,
    chunks: List[Dict[str, str]],
    message: Optional[str] = None,
) -> Tuple[str, float, List[str], Dict[str, object]]:
    """Build a simple MCQ-style assessment question with an Explain/Teach option.

    This helper is intentionally deterministic and lightweight; it does not
    depend on LLM calls. The question text mirrors the MCQ structure so that
    existing clients that only render `response` still see a sensible
    question, while richer clients can use the structured `mcq` block.
    """

    concept_label = concept or "this concept"
    question_id = str(uuid.uuid4())

    level_norm = (level or "").strip().lower()
    if level_norm in {"beginner", "intro"}:
        difficulty = "easy"
    elif level_norm in {"intermediate", "proficient"}:
        difficulty = "medium"
    else:
        difficulty = "hard"

    question = f"Which of the following statements best describes {concept_label}?"

    option_a = f"It is a key idea related to {concept_label} discussed in your materials."
    option_b = f"It is unrelated to {concept_label} or your current course content."
    option_c = f"It is only about memorizing formulas, not understanding {concept_label}."
    option_explain = "I'm not sure, please explain/teach this."

    options: List[Dict[str, object]] = [
        {
            "id": "a",
            "label": option_a,
            "short_label": "A",
            "is_explain_option": False,
            "difficulty": difficulty,
            "tag": "correct",
        },
        {
            "id": "b",
            "label": option_b,
            "short_label": "B",
            "is_explain_option": False,
            "difficulty": difficulty,
            "tag": "distractor_course",
        },
        {
            "id": "c",
            "label": option_c,
            "short_label": "C",
            "is_explain_option": False,
            "difficulty": difficulty,
            "tag": "distractor_memorization",
        },
        {
            "id": "explain",
            "label": option_explain,
            "short_label": "Explain",
            "is_explain_option": True,
            "difficulty": difficulty,
            "tag": "explain_request",
        },
    ]

    lines: List[str] = [question, ""]
    for opt in options[:3]:
        lines.append(f"{opt['short_label']}. {opt['label']}")
    lines.append("")
    lines.append(f"D. {option_explain}")
    response_text = "\n".join(lines)

    confidence = 0.7
    source_ids = [cid for cid in [c.get("id") for c in chunks] if cid]

    mcq: Dict[str, object] = {
        "question_id": question_id,
        "question": question,
        "options": options,
        "correct_option_id": "a",
        "explain_option_id": "explain",
        "concept": concept,
        "level": level,
        "difficulty": difficulty,
        "context_chunk_ids": source_ids,
    }

    return response_text, confidence, source_ids, mcq


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
    message: Optional[str] = None,
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
            "student_message": message or "",
        },
    )
    try:
        result = call_json_chat(
            prompt,
            default=default_payload,
            allow_text_fallback=True,
            text_field="response",
        )
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
    message: Optional[str] = None,
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
            "student_message": message or "",
        },
    )
    try:
        result = call_json_chat(
            prompt,
            default=default_payload,
            allow_text_fallback=True,
            text_field="response",
        )
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
