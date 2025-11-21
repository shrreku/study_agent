from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..mdp.plans import ConceptPlanStep
from .. import responses


def _extract_level(context_obs: Dict[str, Any]) -> str:
    level = context_obs.get("student_level") or context_obs.get("concept_level") or "intermediate"
    if isinstance(level, str):
        return level or "intermediate"
    return "intermediate"


def _extract_chunks(context_obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = context_obs.get("retrieval_chunks")
    if not isinstance(chunks, list):
        chunks = context_obs.get("chunks")
    if not isinstance(chunks, list):
        return []
    return [c for c in chunks if isinstance(c, dict)]


def _extract_message(context_obs: Dict[str, Any]) -> str:
    msg = context_obs.get("student_message") or context_obs.get("message") or ""
    return msg if isinstance(msg, str) else ""


def _extract_history(context_obs: Dict[str, Any]) -> str:
    h = context_obs.get("recent_history") or ""
    return h if isinstance(h, str) else ""


def execute_concept_step(
    *,
    session_id: str,
    user_id: str,
    concept_id: str,
    step: ConceptPlanStep,
    context_obs: Dict[str, Any],
) -> Dict[str, Any]:
    step_type_raw = step.step_type or ""
    step_type = step_type_raw.strip().upper()
    params = step.params or {}

    level = _extract_level(context_obs)
    chunks = _extract_chunks(context_obs)
    message = _extract_message(context_obs)
    history = _extract_history(context_obs)

    concept_label: Optional[str] = concept_id or None

    ui_mode = "free_text"
    mcq_payload: Optional[Dict[str, Any]] = None

    if step_type == "EXPLAIN":
        text, _, _, _ = responses.generate_explain_response(
            concept=concept_label,
            level=level,
            chunks=chunks,
            message=message,
        )
    elif step_type == "SUMMARY":
        text, _, _ = responses.build_review_response(
            concept=concept_label,
            level=level,
            chunks=chunks,
        )
    elif step_type == "EXAMPLE":
        text, _, _ = responses.build_worked_example_response(
            concept=concept_label,
            level=level,
            chunks=chunks,
        )
    elif step_type == "GUIDED_PRACTICE":
        text, _, _ = responses.build_followup_question(
            concept=concept_label,
            level=level,
            chunks=chunks,
            message=message,
        )
    elif step_type == "QUIZ_MCQ":
        quiz_level = str(params.get("difficulty") or level)
        text, _, _, mcq = responses.build_mcq_assessment_question(
            concept=concept_label,
            level=quiz_level,
            chunks=chunks,
            message=message,
        )
        ui_mode = "mcq"
        mcq_payload = mcq
    elif step_type == "REFLECT":
        text, _, _ = responses.build_reflect_response(
            concept=concept_label,
            level=level,
            chunks=chunks,
            message=message,
            history=history,
        )
    elif step_type == "CHECKPOINT":
        prompt = (
            f"Quick checkpoint on {concept_label or 'this topic'}: "
            "On a scale of 1–5, how confident do you feel, and what still feels unclear?"
        )
        text = prompt
    else:
        text, _, _, _ = responses.generate_explain_response(
            concept=concept_label,
            level=level,
            chunks=chunks,
            message=message,
        )

    messages = [
        {
            "role": "tutor",
            "content": text,
        }
    ]

    debug: Dict[str, Any] = {
        "step_id": step.step_id,
        "step_type": step.step_type,
        "session_id": session_id,
        "user_id": user_id,
        "concept_id": concept_id,
    }

    return {
        "messages": messages,
        "ui_mode": ui_mode,
        "mcq_payload": mcq_payload,
        "debug": debug,
    }
