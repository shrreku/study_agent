from __future__ import annotations

from typing import Any, Dict, List, Optional
import os


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_pedagogy_role(chunk: Dict[str, Any]) -> Optional[str]:
    role = chunk.get("pedagogy_role")
    if role:
        return role
    tags = chunk.get("tags")
    if isinstance(tags, dict):
        candidate = tags.get("pedagogy_role") or tags.get("content_type")
        if candidate:
            return candidate
    return None


def build_observation(
    *,
    message: str,
    user_id: str,
    learning_targets: List[str],
    classification: Dict[str, Any],
    focus_concept: Optional[str],
    concept_level: Optional[str],
    inference_concept: Optional[str],
    learning_path: List[str],
    mastery_map: Dict[str, Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    role_sequence: List[str],
    source_chunk_ids: List[str],
    policy_state: Any,
    session_id: str,
    turn_index: int,
    resource_id: Optional[str],
    action_type: str,
    cold_start_triggered: bool,
    confidence: float,
    mastery_delta: Optional[float],
    action_params: Dict[str, Any],
    requested_override_type: Optional[str],
    applied_override_type: Optional[str],
    retrieval_query: Optional[str],
) -> Dict[str, Any]:
    chunk_ids: List[str] = []
    chunk_summaries: List[Dict[str, Any]] = []
    include_snippets = os.getenv("TUTOR_RETRIEVAL_INCLUDE_SNIPPETS_IN_OBS", "false").strip().lower() in {"1", "true", "yes"}
    for chunk in chunks or []:
        cid = chunk.get("id")
        cid_str = str(cid) if cid is not None else None
        if cid_str:
            chunk_ids.append(cid_str)
        item = {
            "id": cid_str,
            "pedagogy_role": _extract_pedagogy_role(chunk),
            "page_number": chunk.get("page_number"),
            "score": _safe_float(chunk.get("score")),
            "sim": _safe_float(chunk.get("sim")),
            "bm25": _safe_float(chunk.get("bm25")),
        }
        if include_snippets:
            try:
                sn = (chunk.get("snippet") or "")
                item["snippet"] = str(sn)[:320]
            except Exception:
                pass
        chunk_summaries.append(item)

    mastery_snapshot = mastery_map.get(focus_concept) if focus_concept else None
    policy_dict = policy_state.to_dict()

    source_ids = [str(cid) for cid in source_chunk_ids if cid is not None]

    classifier_confidence = _safe_float(classification.get("confidence"))

    observation = {
        "metadata": {"version": 1},
        "user": {
            "message": message,
            "user_id": user_id,
            "target_concepts": list(learning_targets or []),
        },
        "classifier": {
            "intent": classification.get("intent", "unknown"),
            "affect": classification.get("affect", "neutral"),
            "concept": classification.get("concept") or "",
            "confidence": classifier_confidence,
            "needs_escalation": bool(classification.get("needs_escalation", False)),
        },
        "tutor": {
            "focus_concept": focus_concept,
            "concept_level": concept_level,
            "inference_concept": inference_concept,
            "learning_path": list(learning_path or []),
            "target_concepts": list(learning_targets or []),
            "mastery_snapshot": mastery_snapshot,
        },
        "retrieval": {
            "query": retrieval_query,
            "chunk_ids": chunk_ids,
            "source_chunk_ids": source_ids,
            "pedagogy_roles": list(role_sequence or []),
            "chunks": chunk_summaries,
        },
        "policy": policy_dict,
        "session": {
            "session_id": session_id,
            "turn_index": turn_index,
            "resource_id": resource_id,
        },
        "action": {
            "type": action_type,
            "cold_start": bool(cold_start_triggered),
            "confidence": _safe_float(confidence),
            "mastery_delta": _safe_float(mastery_delta),
            "source_chunk_ids": source_ids,
            "params": action_params or {},
            "override_type": requested_override_type,
            "override_applied": bool(applied_override_type),
            "applied_override_type": applied_override_type,
        },
    }
    return observation
