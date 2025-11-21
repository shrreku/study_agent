from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..state import TutorSessionPolicy
from ..knowledge import record_cold_start
from ..responses import (
    build_cold_start_question,
    build_hint_response,
    build_reflect_response,
    build_followup_question,
    build_override_question,
    build_review_response,
    build_worked_example_response,
    generate_explain_response,
    build_mcq_assessment_question,
)
from .types import ActionResult


def handle_cold_start(
    focus_concept: Optional[str],
    concept_level: str,
    chunks: List[Dict[str, Any]],
    cur: Any,
    session_id: str,
    policy_state: TutorSessionPolicy,
    *,
    dry_run: bool = False,
) -> ActionResult:
    """Handle cold start scenario for new concepts.

    Prefer a brief grounded explanation when context is available.
    Fall back to a light diagnostic question only when no context is retrieved.
    """
    if not dry_run:
        record_cold_start(cur, session_id, focus_concept or "")
        policy_state.mark_cold_start(focus_concept)

    action_params = {k: v for k, v in {
        "concept": focus_concept,
        "level": concept_level,
        "mode": "cold_start",
    }.items() if v}

    if chunks:
        response_text, confidence, source_chunk_ids, inferred = generate_explain_response(
            focus_concept,
            concept_level,
            chunks,
        )
        return ActionResult(
            action_type="explain",
            response_text=response_text,
            confidence=confidence,
            source_chunk_ids=source_chunk_ids,
            inference_concept=inferred or focus_concept,
            action_params=action_params,
            cold_start_triggered=True,
        )
    else:
        response_text, confidence, source_chunk_ids = build_cold_start_question(
            focus_concept,
            chunks,
        )
        return ActionResult(
            action_type="ask",
            response_text=response_text,
            confidence=confidence,
            source_chunk_ids=source_chunk_ids,
            inference_concept=focus_concept,
            action_params=action_params,
            cold_start_triggered=True,
        )


def handle_override(
    override_type: str,
    focus_concept: Optional[str],
    concept_level: str,
    chunks: List[Dict[str, Any]],
    override_params: Dict[str, Any],
) -> ActionResult:
    """Handle explicit action override requests."""
    concept_for_override = override_params.get("concept") or focus_concept
    level_for_override = override_params.get("level") or concept_level
    difficulty = override_params.get("difficulty")
    question_type = override_params.get("question_type")

    action_params = {
        key: value
        for key, value in {
            "concept": concept_for_override,
            "level": level_for_override,
            "difficulty": difficulty,
            "question_type": question_type,
        }.items()
        if value
    }

    if override_type == "ask":
        # For override-driven "ask" requests, reuse the assessment pipeline so
        # the tutor produces an MCQ-style question with options. This ensures
        # the frontend can render a consistent MCQ UI and drive the SRL plan
        # via structured option selection, instead of a free-text-only prompt.
        return handle_assessment(
            concept_for_override,
            level_for_override,
            chunks,
            cause="override_ask",
        )
    elif override_type == "hint":
        response_text, confidence, source_chunk_ids = build_hint_response(
            concept_for_override,
            level_for_override,
            chunks,
        )
        inference_concept = concept_for_override
    elif override_type == "reflect":
        response_text, confidence, source_chunk_ids = build_reflect_response(
            concept_for_override,
            level_for_override,
            chunks,
        )
        inference_concept = concept_for_override
    elif override_type == "worked_example":
        response_text, confidence, source_chunk_ids = build_worked_example_response(
            concept_for_override,
            level_for_override,
            chunks,
        )
        inference_concept = concept_for_override
    elif override_type == "review":
        response_text, confidence, source_chunk_ids = build_review_response(
            concept_for_override,
            level_for_override,
            chunks,
        )
        inference_concept = concept_for_override
    else:
        response_text, confidence, source_chunk_ids, inferred_concept_candidate = generate_explain_response(
            concept_for_override,
            level_for_override,
            chunks,
        )
        inference_concept = inferred_concept_candidate or concept_for_override
        override_type = "explain"

    return ActionResult(
        action_type=override_type,
        response_text=response_text,
        confidence=confidence,
        source_chunk_ids=source_chunk_ids,
        inference_concept=inference_concept,
        action_params=action_params,
    )


def handle_assessment(
    focus_concept: Optional[str],
    concept_level: str,
    chunks: List[Dict[str, Any]],
    cause: str,
) -> ActionResult:
    """Handle assessment questions after consecutive explains or student reflection."""
    response_text, confidence, source_chunk_ids, mcq = build_mcq_assessment_question(
        focus_concept,
        concept_level,
        chunks,
    )

    action_params = {
        key: value
        for key, value in {
            "concept": focus_concept,
            "level": concept_level,
            "mode": "assessment",
        }.items()
        if value
    }

    return ActionResult(
        action_type="ask",
        response_text=response_text,
        confidence=confidence,
        source_chunk_ids=source_chunk_ids,
        inference_concept=focus_concept,
        action_params=action_params,
        mcq=mcq,
    )


def handle_reflection(
    focus_concept: Optional[str],
    concept_level: str,
    chunks: List[Dict[str, Any]],
    message: str,
    history: Optional[str] = None,
) -> ActionResult:
    """Handle student answer with reflection prompt."""
    response_text, confidence, source_chunk_ids = build_reflect_response(
        focus_concept,
        concept_level,
        chunks,
        message=message,
        history=history,
    )

    action_params = {
        key: value
        for key, value in {
            "concept": focus_concept,
            "level": concept_level,
            "mode": "reflection",
        }.items()
        if value
    }

    return ActionResult(
        action_type="reflect",
        response_text=response_text,
        confidence=confidence,
        source_chunk_ids=source_chunk_ids,
        inference_concept=focus_concept,
        action_params=action_params,
    )


def handle_explain(
    focus_concept: Optional[str],
    concept_level: str,
    chunks: List[Dict[str, Any]],
    mode: str = "default",
) -> ActionResult:
    """Handle explanation response (default or for confusion)."""
    response_text, confidence, source_chunk_ids, inferred_concept_candidate = generate_explain_response(
        focus_concept,
        concept_level,
        chunks,
    )
    inference_concept = inferred_concept_candidate or focus_concept

    action_params = {
        key: value
        for key, value in {
            "concept": inference_concept,
            "level": concept_level,
            "mode": mode,
        }.items()
        if value
    }

    return ActionResult(
        action_type="explain",
        response_text=response_text,
        confidence=confidence,
        source_chunk_ids=source_chunk_ids,
        inference_concept=inference_concept,
        action_params=action_params,
    )
