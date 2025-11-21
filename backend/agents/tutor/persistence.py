from __future__ import annotations

from typing import Any, Dict, List, Optional
from psycopg2.extras import Json


def ensure_session(
    cursor,
    user_id: str,
    session_id: Optional[str],
    target_concepts: List[str],
    resource_id: Optional[str],
    policy: Optional[Dict[str, Any]],
) -> str:
    if session_id:
        cursor.execute(
            """
            SELECT id::text
            FROM tutor_session
            WHERE id = %s::uuid
            LIMIT 1
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if row and row[0]:
            return row[0]
    cursor.execute(
        """
        INSERT INTO tutor_session (
            user_id,
            resource_id,
            target_concepts,
            status,
            policy,
            last_concept,
            last_action
        )
        VALUES (
            %s::uuid,
            NULLIF(%s, '')::uuid,
            %s::text[],
            %s,
            %s,
            %s,
            %s
        )
        RETURNING id::text
        """,
        (
            user_id,
            resource_id,
            target_concepts or [],
            "active",
            Json(policy) if policy is not None else None,
            None,
            None,
        ),
    )
    row = cursor.fetchone()
    return row[0] if row else session_id or ""


def get_session_state(cursor, session_id: str) -> Dict[str, Any]:
    cursor.execute(
        """
        SELECT last_concept, last_action, target_concepts, policy
        FROM tutor_session
        WHERE id = %s::uuid
        """,
        (session_id,),
    )
    row = cursor.fetchone()
    if not row:
        return {
            "last_concept": None,
            "last_action": None,
            "target_concepts": [],
            "policy": {},
        }
    return {
        "last_concept": row[0],
        "last_action": row[1],
        "target_concepts": row[2] or [],
        "policy": row[3] or {},
    }


def next_turn_index(cursor, session_id: str) -> int:
    cursor.execute(
        """
        SELECT COALESCE(MAX(turn_index), -1)
        FROM tutor_turn
        WHERE session_id = %s::uuid
        """,
        (session_id,),
    )
    row = cursor.fetchone()
    last_index = int(row[0]) if row and row[0] is not None else -1
    return last_index + 1


def get_recent_turns(cursor, session_id: str, limit: int = 6) -> List[Dict[str, Any]]:
    """Fetch a small number of recent turns for history-aware policy decisions.

    Returns newest-first rows with minimal fields to avoid pulling large payloads.
    """
    if limit <= 0:
        return []
    cursor.execute(
        """
        SELECT turn_index, user_text, intent, affect, concept, action_type, response_text
        FROM tutor_turn
        WHERE session_id = %s::uuid
        ORDER BY turn_index DESC
        LIMIT %s
        """,
        (session_id, limit),
    )
    rows = cursor.fetchall() or []
    history: List[Dict[str, Any]] = []
    for row in rows:
        try:
            history.append(
                {
                    "turn_index": int(row[0]) if row[0] is not None else None,
                    "user_text": row[1],
                    "intent": row[2],
                    "affect": row[3],
                    "concept": row[4],
                    "action_type": row[5],
                    "response_text": row[6],
                }
            )
        except Exception:
            continue
    return history


def get_last_turn_retrieval(cursor, session_id: str) -> Optional[Dict[str, Any]]:
    cursor.execute(
        """
        SELECT turn_index, intent, affect, concept, source_chunk_ids, retrieval_metadata
        FROM tutor_turn
        WHERE session_id = %s::uuid
        ORDER BY turn_index DESC
        LIMIT 1
        """,
        (session_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None

    try:
        turn_index = int(row[0]) if row[0] is not None else None
    except Exception:
        turn_index = None

    intent = row[1]
    affect = row[2]
    concept = row[3]

    raw_ids = row[4] or []
    # Normalize source_chunk_ids which may be returned as a list or as a
    # Postgres array string like "{uuid1,uuid2}".
    normalized_ids: List[str] = []
    try:
        if isinstance(raw_ids, str):
            txt = raw_ids.strip()
            if txt.startswith("{") and txt.endswith("}"):
                txt = txt[1:-1]
            if txt:
                parts = [p.strip() for p in txt.split(",") if p.strip()]
                normalized_ids = parts
        elif isinstance(raw_ids, (list, tuple)):
            normalized_ids = [str(cid) for cid in raw_ids if cid is not None]
    except Exception:
        normalized_ids = []

    source_chunk_ids = normalized_ids

    retrieval_metadata = row[5] or {}
    if not isinstance(retrieval_metadata, dict):
        retrieval_metadata = {}

    metadata_count = None
    try:
        metadata_count = int(retrieval_metadata.get("count")) if "count" in retrieval_metadata else None
    except Exception:
        metadata_count = None

    chunk_count = metadata_count if metadata_count is not None else len(source_chunk_ids)

    if not source_chunk_ids and not retrieval_metadata:
        return None

    return {
        "turn_index": turn_index,
        "intent": intent,
        "affect": affect,
        "concept": concept,
        "source_chunk_ids": source_chunk_ids,
        "retrieval_metadata": retrieval_metadata,
        "chunk_count": chunk_count,
    }


def insert_turn(
    cursor,
    session_id: str,
    turn_index: int,
    user_text: str,
    intent: str,
    affect: str,
    concept: Optional[str],
    action_type: str,
    response_text: str,
    source_chunk_ids: List[str],
    confidence: float,
    mastery_delta: Optional[float],
    *,
    model_id: Optional[str] = None,
    model_name: Optional[str] = None,
    tool_calls: Optional[Dict[str, Any]] = None,
    retrieval_metadata: Optional[Dict[str, Any]] = None,
    policy_trace: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    cursor.execute(
        """
        INSERT INTO tutor_turn (
            session_id,
            turn_index,
            user_text,
            intent,
            affect,
            concept,
            action_type,
            response_text,
            source_chunk_ids,
            confidence,
            mastery_delta,
            model_id,
            model_name,
            tool_calls,
            retrieval_metadata,
            policy_trace
        )
        VALUES (
            %s::uuid,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s::uuid[],
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        RETURNING id::text
        """,
        (
            session_id,
            turn_index,
            user_text,
            intent,
            affect,
            concept,
            action_type,
            response_text,
            source_chunk_ids or [],
            confidence,
            mastery_delta,
            model_id,
            model_name,
            Json(tool_calls) if tool_calls is not None else None,
            Json(retrieval_metadata) if retrieval_metadata is not None else None,
            Json(policy_trace) if policy_trace is not None else None,
        ),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def update_session(cursor, session_id: str, concept: Optional[str], action_type: str, policy: Dict[str, Any]) -> None:
    cursor.execute(
        """
        UPDATE tutor_session
        SET updated_at = now(),
            last_concept = COALESCE(%s, last_concept),
            last_action = %s,
            policy = %s
        WHERE id = %s::uuid
        """,
        (concept, action_type, Json(policy), session_id),
    )
