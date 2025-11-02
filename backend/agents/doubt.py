from typing import List, Dict, Any
import os
import logging
from uuid import UUID

from llm import call_llm_json, call_llm_for_tagging
from core.db import get_db_conn
from metrics import MetricsCollector
from .retrieval import hybrid_search, fetch_chunks_by_ids, search_chunks_simple, consolidate_adjacent_microchunks, dedup_by_id, filter_relevant
from prompts import get as prompt_get, render as prompt_render


def _fetch_chunks_by_ids(ids: List[str]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            user = os.getenv("POSTGRES_USER", "postgres")
            password = os.getenv("POSTGRES_PASSWORD", "postgres")
            host = os.getenv("POSTGRES_HOST", "postgres")
            port = os.getenv("POSTGRES_PORT", "5432")
            db = os.getenv("POSTGRES_DB", "app")
            dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id::text, resource_id::text, page_number, LEFT(full_text, 300) AS snippet FROM chunk WHERE id = ANY(%s::uuid[])",
                    (ids,),
                )
                return cur.fetchall()
        finally:
            conn.close()
    except Exception:
        logging.exception("fetch_chunks_by_ids_failed")
        return []


def _search_chunks_simple(q: str, k: int = 5) -> List[Dict[str, Any]]:
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            user = os.getenv("POSTGRES_USER", "postgres")
            password = os.getenv("POSTGRES_PASSWORD", "postgres")
            host = os.getenv("POSTGRES_HOST", "postgres")
            port = os.getenv("POSTGRES_PORT", "5432")
            db = os.getenv("POSTGRES_DB", "app")
            dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id::text, resource_id::text, page_number, LEFT(full_text, 300) AS snippet FROM chunk WHERE full_text ILIKE %s LIMIT %s",
                    (f"%{q}%", k),
                )
                return cur.fetchall()
        finally:
            conn.close()
    except Exception:
        logging.exception("search_chunks_failed")
        return []


def _resolve_user_id(payload: Dict[str, Any]) -> str | None:
    user_id = (payload or {}).get("user_id") or os.getenv("TEST_USER_ID")
    if not user_id:
        return None
    try:
        return str(UUID(str(user_id)))
    except Exception:
        logging.warning("doubt_user_id_invalid", extra={"user_id": user_id})
        return None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def doubt_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Accept multiple aliases for question to remain compatible with UI
    question = payload.get("question") or payload.get("question_text") or payload.get("q") or ""
    context_chunk_ids = payload.get("context_chunk_ids") or []
    resource_id = payload.get("resource_id")

    chunks: List[Dict[str, Any]] = []
    if context_chunk_ids:
        chunks = fetch_chunks_by_ids(context_chunk_ids)
    if not chunks and question:
        try:
            chunks = hybrid_search(question, 12, resource_id=resource_id)
        except Exception:
            chunks = search_chunks_simple(question, 12, resource_id=resource_id)

    if not context_chunk_ids:
        chunks = filter_relevant(chunks)
    if not chunks:
        return {"answer": "I don't know based on the provided materials.", "citations": [], "expanded_steps": ""}

    # Micro-chunk adaptation: deduplicate and consolidate adjacent items for context
    chunks = dedup_by_id(chunks)

    target_chars = _env_int("DOUBT_TARGET_CHARS", 1200)
    consolidated = consolidate_adjacent_microchunks(chunks, window=2, target_chars=target_chars)

    context_lines = []
    for idx, group in enumerate(consolidated[:6], start=1):
        # show first chunk id as representative, include list in citations
        rep_id = group.get("chunk_ids")[0] if group.get("chunk_ids") else None
        context_lines.append(f"[{idx}] {group.get('snippet','')[:320]} (chunk_id:{rep_id}, page:{group.get('page_number')})")
    context_text = "\n".join(context_lines)

    # Build prompt via registry
    tmpl = prompt_get("doubt.answer")
    prompt = prompt_render(tmpl, {"question": question, "context": context_text})

    try:
        default_citations = []
        if consolidated:
            for g in consolidated[:3]:
                cid = (g.get("chunk_ids") or [None])[0]
                default_citations.append({"chunk_id": cid, "page": g.get("page_number"), "snippet": g.get("snippet")})
        default = {"answer": "", "expanded_steps": ""}
        out = call_llm_json(prompt, default)
        result = {
            "answer": out.get("answer", ""),
            "citations": default_citations,
            "expanded_steps": out.get("expanded_steps"),
        }

        # --- S5-B: Doubt logging (best-effort, non-blocking) ---
        concepts: List[str] = []
        try:
            # Prefer tagging on the user's question; fallback to first consolidated snippet
            if question:
                tag = call_llm_for_tagging(question)
                concepts = tag.get("concepts") or []
            if not concepts and consolidated:
                tag2 = call_llm_for_tagging(consolidated[0].get("snippet") or "")
                concepts = tag2.get("concepts") or []
        except Exception:
            logging.exception("doubt_concepts_extract_failed")
            concepts = []

        weak_signal = False
        try:
            cls = call_llm_json(
                (
                    "Classify if the user is revealing a weakness that needs remediation. "
                    "Answer Yes if they ask for explanation, steps, or express confusion; No if it's purely curiosity.\n"
                    "Return JSON: {\"weak_signal\": boolean}.\n\n"
                    f"Question: {question}"
                ),
                {"weak_signal": False},
            )
            wv = cls.get("weak_signal")
            weak_signal = bool(wv) if wv is not None else False
        except Exception:
            logging.exception("doubt_weak_classifier_failed")
            weak_signal = False

        # Emit metric if weakness detected
        try:
            if weak_signal:
                MetricsCollector.get_global().increment("doubt_weak_signals_total")
            else:
                MetricsCollector.get_global().increment("doubt_strength_signals_total")
        except Exception:
            logging.exception("doubt_metrics_emit_failed")

        # Persist doubt and update mastery for tagged concepts (best-effort)
        user_id = _resolve_user_id(payload)
        if user_id:
            try:
                conn = get_db_conn()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO user_doubt (user_id, question, concepts, weak_signal)
                            VALUES (%s::uuid, %s, %s::text[], %s)
                            """,
                            (user_id, question, concepts or [], weak_signal),
                        )
                        if concepts:
                            step_correct = _env_float("MASTERY_STEP_CORRECT", 0.1)
                            step_wrong = _env_float("MASTERY_STEP_WRONG", -0.05)
                            if step_wrong > 0:
                                step_wrong = -abs(step_wrong)
                            for concept_name in concepts[:5]:
                                attempt_inc = 1
                                correct_inc = 0 if weak_signal else 1
                                delta = step_wrong if weak_signal else step_correct
                                initial_mastery = step_correct if not weak_signal else 0.0
                                cur.execute(
                                    """
                                    INSERT INTO user_concept_mastery (user_id, concept, mastery, last_seen, attempts, correct)
                                    VALUES (%s::uuid, %s, %s, now(), %s, %s)
                                    ON CONFLICT (user_id, concept) DO UPDATE
                                      SET last_seen = now(),
                                          attempts = user_concept_mastery.attempts + %s,
                                          correct = user_concept_mastery.correct + %s,
                                          mastery = LEAST(1.0, GREATEST(0.0, user_concept_mastery.mastery + %s))
                                    """,
                                    (
                                        user_id,
                                        concept_name,
                                        max(initial_mastery, 0.0),
                                        attempt_inc,
                                        correct_inc,
                                        attempt_inc,
                                        correct_inc,
                                        delta,
                                    ),
                                )
                    conn.commit()
                finally:
                    conn.close()
            except Exception:
                logging.exception("doubt_log_insert_failed")
                try:
                    MetricsCollector.get_global().increment("doubt_log_failures_total")
                except Exception:
                    pass

        return result
    except Exception:
        logging.exception("doubt_agent_llm_failed")
        default_citations = []
        if consolidated:
            for g in consolidated[:3]:
                cid = (g.get("chunk_ids") or [None])[0]
                default_citations.append({"chunk_id": cid, "page": g.get("page_number"), "snippet": g.get("snippet")})
        return {"answer": "Sorry, I couldn't generate an answer right now.", "citations": default_citations, "expanded_steps": None}


