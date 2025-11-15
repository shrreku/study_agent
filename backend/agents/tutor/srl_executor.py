from __future__ import annotations

from typing import Any, Dict, List, Optional

from .planning import TutorPlan
from .retrieval import retrieve_chunks
from .tools_runtime import execute_action


def execute_plan_steps(
    plan: "TutorPlan",
    focus_concept: Optional[str],
    concept_level: str,
    message: str,
    resource_id: Optional[str],
) -> Dict[str, Any]:
    """Execute up to 4 steps from a TutorPlan with per-step retrieval.

    Returns a dict with keys:
      - text: combined response text
      - confidence: averaged confidence across steps
      - source_chunk_ids: union of source chunk ids
      - inference_concept: inferred or focus concept
      - last_action: action of the last executed step
      - chunks: union of all retrieved chunks
      - step_progress: list of step progress entries for tracing
    """
    steps = list(getattr(plan, "steps", []) or [])
    if not steps:
        steps = [
            {
                "action": getattr(plan, "intended_action", "explain") or "explain",
                "pedagogy_focus": list(getattr(plan, "pedagogy_focus", []) or []),
            },
            {"action": "ask", "pedagogy_focus": ["concept_check"]},
        ]

    combined_text_parts: List[str] = []
    combined_confidences: List[float] = []
    combined_source_ids: List[str] = []
    all_chunks: Dict[str, Dict[str, Any]] = {}
    step_progress: List[Dict[str, Any]] = []
    last_action = "explain"
    inference_concept = focus_concept

    for idx, step in enumerate(steps[:4]):
        action = str(step.get("action") or "explain").lower()
        roles = list(step.get("pedagogy_focus") or getattr(plan, "pedagogy_focus", []) or [])
        # step-specific query override -> plan query -> focus concept -> message
        q = step.get("target_concept") or getattr(plan, "retrieval_query", None) or focus_concept or message
        step_chunks = retrieve_chunks(q, resource_id, roles)

        for c in step_chunks or []:
            cid = c.get("id")
            if cid and cid not in all_chunks:
                all_chunks[cid] = c

        text, conf, src_ids, inferred = execute_action(
            action=action,
            plan=plan,
            concept=focus_concept,
            level=concept_level,
            chunks=step_chunks or [],
            message=message,
        )
        if inferred and not inference_concept:
            inference_concept = inferred

        if text:
            combined_text_parts.append(str(text).strip())
        try:
            combined_confidences.append(float(conf))
        except Exception:
            pass
        for sid in src_ids or []:
            if sid and sid not in combined_source_ids:
                combined_source_ids.append(sid)

        try:
            step_progress.append(
                {
                    "stage": "step",
                    "index": idx,
                    "action": action,
                    "roles": roles,
                    "query": q,
                    "retrieved": len(step_chunks or []),
                    "chunk_ids": [c.get("id") for c in (step_chunks or []) if c.get("id")],
                }
            )
        except Exception:
            pass
        last_action = action

    text = "\n\n".join(filter(None, combined_text_parts))
    conf_out = sum(combined_confidences) / len(combined_confidences) if combined_confidences else 0.6
    return {
        "text": text,
        "confidence": conf_out,
        "source_chunk_ids": combined_source_ids,
        "inference_concept": inference_concept,
        "last_action": last_action,
        "chunks": list(all_chunks.values()),
        "step_progress": step_progress,
    }
