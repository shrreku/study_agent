from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .planning import TutorPlan
from .responses import (
    build_followup_question,
    build_hint_response,
    build_reflect_response,
    build_review_response,
    build_worked_example_response,
    generate_explain_response,
    generate_explain_response_with_plan,
)


def execute_action(
    *,
    action: str,
    plan: Optional[TutorPlan],
    concept: Optional[str],
    level: str,
    chunks: List[Dict[str, Any]],
    message: str,
) -> Tuple[str, float, List[str], Optional[str]]:
    a = (action or "").lower().strip() or "explain"

    if a == "ask":
        text, conf, src_ids = build_followup_question(concept, level, chunks)
        return text, float(conf), list(src_ids or []), concept

    if a == "hint":
        text, conf, src_ids = build_hint_response(concept, level, chunks)
        return text, float(conf), list(src_ids or []), concept

    if a == "reflect":
        text, conf, src_ids = build_reflect_response(concept, level, chunks)
        return text, float(conf), list(src_ids or []), concept

    if a == "review":
        text, conf, src_ids = build_review_response(concept, level, chunks)
        return text, float(conf), list(src_ids or []), concept

    if a == "worked_example":
        text, conf, src_ids = build_worked_example_response(concept, level, chunks)
        return text, float(conf), list(src_ids or []), concept

    if plan is not None:
        text, conf, src_ids, inferred = generate_explain_response_with_plan(
            plan=plan, concept=concept, level=level, chunks=chunks
        )
        return text, float(conf), list(src_ids or []), (inferred or concept)

    text, conf, src_ids, inferred = generate_explain_response(
        concept, level, chunks
    )
    return text, float(conf), list(src_ids or []), (inferred or concept)
