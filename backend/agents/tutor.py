"""Compatibility shim for historical imports.

This module previously contained the full tutor agent implementation. The
logic now lives in ``backend.agents.tutor.agent`` alongside supporting
submodules under ``backend.agents.tutor``. Importing ``tutor_agent`` from
here continues to work for older call sites.
"""

from .tutor.agent import tutor_agent  # re-export for backward compatibility

__all__ = ["tutor_agent"]


ALLOWED_INTENTS = {"question", "answer", "reflection", "off_topic", "greeting", "unknown"}
ALLOWED_AFFECTS = {"confused", "unsure", "engaged", "frustrated", "neutral"}

LEVEL_BUCKETS: List[Tuple[str, float, float]] = [
    ("beginner", 0.0, 0.3),
    ("developing", 0.3, 0.6),
    ("proficient", 0.6, 0.8),
    ("mastering", 0.8, 1.01),
]


def _normalize_concepts(raw: Optional[Any]) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    try:
        return [str(raw).strip()]
    except Exception:
        return []


def _format_concept_list(concepts: List[str]) -> str:
    if not concepts:
        return "None"
    return ", ".join(concepts[:6])


def _format_context_snippets(chunks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        snippet = (chunk.get("snippet") or "").strip()
        if not snippet:
            continue
        label = chunk.get("id") or "?"
        parts.append(f"[Chunk {idx} | {label}] {snippet}")
    return "\n\n".join(parts)


def _clamp_confidence(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return 0.0
    if num < 0.0:
        return 0.0
    if num > 1.0:
        return 1.0
    return num


def _classify_message(
    message: str,
    target_concepts: List[str],
    last_concept: Optional[str],
) -> Dict[str, Any]:
    template = prompt_get("tutor.classify")
    prompt = prompt_render(
        template,
        {
            "student_message": message,
            "target_concepts": _format_concept_list(target_concepts),
            "last_concept": last_concept or "",
        },
    )
    default_concept = last_concept or (target_concepts[0] if target_concepts else "")
    default_payload = {
        "intent": "unknown",
        "affect": "neutral",
        "concept": default_concept,
        "confidence": 0.3,
        "needs_escalation": False,
    }
    try:
        result = call_json_chat(prompt, default=default_payload)
    except Exception:
        logger.exception("tutor_classify_call_failed")
        result = default_payload

    intent = str(result.get("intent") or "unknown").lower()
    if intent not in ALLOWED_INTENTS:
        intent = "unknown"
    affect = str(result.get("affect") or "neutral").lower()
    if affect not in ALLOWED_AFFECTS:
        affect = "neutral"
    concept = str(result.get("concept") or "").strip()
    if not concept:
        concept = default_concept or ""
    confidence = _clamp_confidence(result.get("confidence"))
    needs_escalation = bool(result.get("needs_escalation", False))

    return {
        "intent": intent,
        "affect": affect,
        "concept": concept,
        "confidence": confidence,
        "needs_escalation": needs_escalation,
    }


def _ensure_session(
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


def _get_session_state(cursor, session_id: str) -> Dict[str, Any]:
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


def _next_turn_index(cursor, session_id: str) -> int:
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


def _retrieve_chunks(
    query: str,
    resource_id: Optional[str],
    k: int = 5,
    pedagogy_roles: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if not query.strip():
        return []
    try:
        results = hybrid_search(query, k=max(5, k), resource_id=resource_id)
        results = filter_relevant(results)
        results = _score_with_pedagogy(results, pedagogy_roles)
        results = diversify_by_page(results, per_page=1)
        return results[:k]
    except Exception:
        logger.exception("tutor_retrieval_failed")
        return []


def _level_for_mastery(mastery: Optional[Any]) -> str:
    try:
        score = float(mastery)
    except Exception:
        score = None
    if score is None:
        return "beginner"
    for name, lower, upper in LEVEL_BUCKETS:
        if lower <= score < upper:
            return name
    return "beginner"


def _needs_cold_start(
    concept: Optional[str],
    mastery_map: Dict[str, Dict[str, Any]],
    policy_state: Optional[Dict[str, Any]] = None,
) -> bool:
    if not concept:
        return False
    if policy_state:
        completed = policy_state.get("cold_start_completed") or []
        if isinstance(completed, (list, tuple, set)) and concept in completed:
            return False
    info = mastery_map.get(concept)
    if not info:
        return True
    attempts = info.get("attempts") or 0
    mastery = info.get("mastery") or 0.0
    return attempts < 1 or mastery < 0.15


def _record_cold_start(cursor, session_id: str, concept: str) -> None:
    cursor.execute(
        """
        INSERT INTO tutor_event (session_id, event_type, payload)
        VALUES (%s::uuid, %s, %s)
        """,
        (
            session_id,
            "cold_start_triggered",
            Json({"concept": concept}),
        ),
    )


def _fetch_mastery_map(cursor, user_id: str) -> Dict[str, Dict[str, Any]]:
    cursor.execute(
        """
        SELECT concept, mastery, attempts, correct
        FROM user_concept_mastery
        WHERE user_id = %s::uuid
        """,
        (user_id,),
    )
    rows = cursor.fetchall()
    mastery_map: Dict[str, Dict[str, Any]] = {}
    for r in rows or []:
        concept = r[0]
        mastery_map[concept] = {
            "mastery": float(r[1]) if r[1] is not None else None,
            "attempts": int(r[2]) if r[2] is not None else 0,
            "correct": int(r[3]) if r[3] is not None else 0,
        }
    return mastery_map


def _fetch_prereq_chain(concepts: List[str], max_depth: int = 4) -> List[str]:
    if not concepts:
        return []
    try:
        from neo4j import GraphDatabase
    except Exception:
        return list(dict.fromkeys(concepts))

    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
    order: List[str] = []
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
    except Exception:
        logger.exception("tutor_neo4j_driver_failed")
        return list(dict.fromkeys(concepts))

    def _fetch(tx, name: str) -> List[str]:
        try:
            q = (
                "MATCH (c:Concept {name:$name})\n"
                "CALL apoc.path.subgraphNodes(c, {relationshipFilter:'<PREREQUISITE_OF', maxLevel:$depth}) YIELD node\n"
                "RETURN DISTINCT node.name AS name"
            )
            res = tx.run(q, name=name, depth=max_depth)
        except Exception:
            q2 = (
                "MATCH (p:Concept)-[:PREREQUISITE_OF*0..$depth]->(c:Concept {name:$name})\n"
                "RETURN DISTINCT p.name AS name"
            )
            res = tx.run(q2, name=name, depth=max_depth)
        return [row["name"] for row in res]

    try:
        with driver.session() as session:
            seen: List[str] = []
            for concept in concepts:
                try:
                    names = session.execute_read(_fetch, concept)
                except Exception:
                    names = []
                if concept not in names:
                    names.append(concept)
                for name in names:
                    if name not in seen:
                        seen.append(name)
            order = seen
    except Exception:
        logger.exception("tutor_prereq_fetch_failed")
        order = list(dict.fromkeys(concepts))
    finally:
        try:
            driver.close()
        except Exception:
            pass
    return order


def _select_focus_concept(
    classification: Dict[str, Any],
    learning_path: List[str],
    mastery_map: Dict[str, Dict[str, Any]],
    fallback_concepts: List[str],
) -> Optional[str]:
    primary = (classification.get("concept") or "").strip()
    if primary:
        info = mastery_map.get(primary)
        if not info or (info.get("mastery") or 0.0) < 0.85:
            return primary
    for concept in learning_path:
        info = mastery_map.get(concept)
        mastery_val = info.get("mastery") if info else None
        if mastery_val is None or mastery_val < 0.8:
            return concept
    for concept in fallback_concepts:
        if concept:
            return concept
    return primary or (learning_path[0] if learning_path else None)


def _fallback_response_text(concept: Optional[str], chunks: List[Dict[str, Any]]) -> str:
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


def _generate_explain_response(
    concept: Optional[str],
    level: str,
    chunks: List[Dict[str, Any]],
    fallback_response: Optional[str] = None,
) -> Tuple[str, str, float, List[str], Optional[str]]:
    if not chunks:
        response = (
            "I couldn't find a grounded snippet yet. Let's review your materials first. "
            "Could you point me to the chapter or section?"
        )
        return response, "needs-review", 0.2, [], concept

    context_block = _format_context_snippets(chunks)
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
    confidence = _clamp_confidence(result.get("confidence")) or default_payload["confidence"]
    source_ids = [c.get("id") for c in chunks if c.get("id")]
    return response_text, "explain", confidence, source_ids, concept


def tutor_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    message = (payload.get("message") or "").strip()
    if not message:
        raise ValueError("invalid payload, missing message")

    user_id = (payload.get("user_id") or os.getenv("TEST_USER_ID", "")).strip()
    if not user_id:
        raise ValueError("user_id required (set TEST_USER_ID env var or pass user_id)")

    session_id = (payload.get("session_id") or "").strip() or None
    resource_id = (payload.get("resource_id") or "").strip() or None
    target_concepts = _normalize_concepts(payload.get("target_concepts"))
    session_policy = payload.get("session_policy") or {"version": 1, "strategy": "baseline"}

    conn = get_db_conn()
    response_payload: Dict[str, Any]
    try:
        with conn.cursor() as cur:
            session_id = _ensure_session(cur, user_id, session_id, target_concepts, resource_id, session_policy)
            session_state = _get_session_state(cur, session_id)
            turn_index = _next_turn_index(cur, session_id)

            classification = _classify_message(
                message,
                target_concepts or session_state.get("target_concepts", []),
                session_state.get("last_concept"),
            )

            learning_targets = target_concepts or session_state.get("target_concepts", [])
            mastery_map = _fetch_mastery_map(cur, user_id)
            learning_path = _fetch_prereq_chain(
                [classification.get("concept")] + list(learning_targets)
                if classification.get("concept")
                else list(learning_targets),
            )

            focus_concept = _select_focus_concept(
                classification,
                learning_path,
                mastery_map,
                learning_targets,
            )
            concept_level = _level_for_mastery(
                (mastery_map.get(focus_concept) or {}).get("mastery")
            )

            logger.info(
                "tutor_policy_stage",
                extra=
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "turn_index": turn_index,
                    "classification": classification,
                    "focus_concept": focus_concept,
                    "concept_level": concept_level,
                    "learning_path": learning_path,
                    "targets": learning_targets,
                },
            )

            role_sequence: List[str] = []
            if concept_level in {"beginner", "developing"}:
                role_sequence = ["definition", "explanation", "example"]
            elif concept_level == "proficient":
                role_sequence = ["example", "application", "derivation"]
            else:
                role_sequence = ["derivation", "proof", "application"]

            chunks = _retrieve_chunks(message, resource_id, k=4, pedagogy_roles=role_sequence)
            if not chunks and focus_concept:
                chunks = _retrieve_chunks(focus_concept, resource_id, k=4, pedagogy_roles=role_sequence)

            logger.info(
                "tutor_retrieval_summary",
                extra=
                {
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "query": message,
                    "resource_id": resource_id,
                    "focus_concept": focus_concept,
                    "pedagogy_roles": role_sequence,
                    "chunk_ids": [c.get("id") for c in chunks],
                },
            )

            affect = classification.get("affect", "neutral")
            intent = classification.get("intent", "unknown")

            cold_start_triggered = False
            action_type = "explain"
            inference_concept_candidate: Optional[str] = focus_concept
            inferred_concept: Optional[str] = focus_concept

            if _needs_cold_start(focus_concept, mastery_map, session_state.get("policy")):
                cold_start_triggered = True
                action_type = "ask"
                default_question = f"What is a key idea about {focus_concept}?"
                cold_prompt = prompt_render(
                    prompt_get("tutor.ask"),
                    {
                        "concept": focus_concept or "the concept",
                        "level": "beginner",
                        "context": _format_context_snippets(chunks) or "",
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
                confidence = _clamp_confidence(ask_result.get("confidence") or 0.4)
                source_chunk_ids = [cid for cid in [c.get("id") for c in chunks] if cid]
                _record_cold_start(cur, session_id, focus_concept or "")
                logger.info(
                    "tutor_action_decision",
                    extra=
                    {
                        "session_id": session_id,
                        "turn_index": turn_index,
                        "action_type": action_type,
                        "focus_concept": focus_concept,
                        "cause": "cold_start",
                        "confidence": confidence,
                        "chunk_ids": source_chunk_ids,
                    },
                )
                inferred_concept = focus_concept
                completed_list = session_policy_state.get("cold_start_completed")
                if not isinstance(completed_list, list):
                    completed_list = []
                if focus_concept and focus_concept not in completed_list:
                    completed_list = [*completed_list, focus_concept]
                session_policy_state["cold_start_completed"] = completed_list
            else:
                if affect in {"confused", "unsure"} and chunks:
                    action_type = "hint"
                    hint_prompt = prompt_render(
                        prompt_get("tutor.hint"),
                        {
                            "concept": focus_concept or "the concept",
                            "level": concept_level,
                            "context": _format_context_snippets(chunks),
                        },
                    )
                    hint_default = {
                        "response": _fallback_response_text(focus_concept, chunks),
                        "confidence": 0.5,
                    }
                    try:
                        hint_result = call_json_chat(hint_prompt, default=hint_default)
                    except Exception:
                        logger.exception("tutor_hint_prompt_failed")
                        hint_result = hint_default
                    response_text = str(hint_result.get("response") or hint_default["response"]).strip()
                    confidence = _clamp_confidence(hint_result.get("confidence") or 0.5)
                    source_chunk_ids = [c.get("id") for c in chunks if c.get("id")]
                    logger.info(
                        "tutor_action_decision",
                        extra=
                        {
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "action_type": action_type,
                            "focus_concept": focus_concept,
                            "cause": "affect_confused",
                            "confidence": confidence,
                            "chunk_ids": source_chunk_ids,
                        },
                    )
                elif intent == "answer" and chunks:
                    action_type = "reflect"
                    reflect_prompt = prompt_render(
                        prompt_get("tutor.reflect"),
                        {
                            "concept": focus_concept or "the concept",
                            "level": concept_level,
                            "context": _format_context_snippets(chunks),
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
                    confidence = _clamp_confidence(reflect_result.get("confidence") or 0.6)
                    source_chunk_ids = [c.get("id") for c in chunks if c.get("id")]
                    logger.info(
                        "tutor_action_decision",
                        extra=
                        {
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "action_type": action_type,
                            "focus_concept": focus_concept,
                            "cause": "student_answer",
                            "confidence": confidence,
                            "chunk_ids": source_chunk_ids,
                        },
                    )
                else:
                    response_text, action_type, confidence, source_chunk_ids, inference_concept_candidate = _generate_explain_response(
                        focus_concept,
                        concept_level,
                        chunks,
                    )
                    logger.info(
                        "tutor_action_decision",
                        extra=
                        {
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "action_type": action_type,
                            "focus_concept": focus_concept,
                            "cause": "explain_default",
                            "confidence": confidence,
                            "chunk_ids": source_chunk_ids,
                        },
                    )

                inferred_concept = inference_concept_candidate or focus_concept

            intent = classification.get("intent", "unknown")
            affect = classification.get("affect", "neutral")
            mastery_delta = payload.get("mastery_delta")

            session_policy_state = session_state.get("policy") or {}
            session_policy_state.update(
                {
                    "learning_path": learning_path,
                    "focus_concept": focus_concept,
                    "focus_level": concept_level,
                    "cold_start": cold_start_triggered,
                }
            )

            cur.execute(
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
                    message,
                    intent,
                    affect,
                    inferred_concept,
                    action_type,
                    response_text,
                    source_chunk_ids or [],
                    confidence,
                    mastery_delta,
                ),
            )
            turn_row = cur.fetchone()

            cur.execute(
                """
                UPDATE tutor_session
                SET updated_at = now(),
                    last_concept = COALESCE(%s, last_concept),
                    last_action = %s,
                    policy = %s
                WHERE id = %s::uuid
                """,
                (inferred_concept, action_type, Json(session_policy_state), session_id),
            )

            response_payload = {
                "session_id": session_id,
                "turn_id": turn_row[0] if turn_row else None,
                "turn_index": turn_index,
                "response": response_text,
                "action_type": action_type,
                "source_chunk_ids": source_chunk_ids,
                "confidence": confidence,
                "intent": intent,
                "affect": affect,
                "concept": inferred_concept,
                "level": concept_level,
                "learning_path": learning_path,
                "cold_start": cold_start_triggered,
                "classification_confidence": classification.get("confidence"),
            }
            logger.info(
                "tutor_turn_committed",
                extra=
                {
                    "session_id": session_id,
                    "turn_id": response_payload.get("turn_id"),
                    "turn_index": turn_index,
                    "user_id": user_id,
                    "action_type": action_type,
                    "intent": intent,
                    "affect": affect,
                    "concept": inferred_concept,
                    "confidence": confidence,
                    "cold_start": cold_start_triggered,
                },
            )
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("tutor_agent_failed")
        raise
    finally:
        conn.close()

    return response_payload
