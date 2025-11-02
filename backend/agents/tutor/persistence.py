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
            mastery_delta
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
