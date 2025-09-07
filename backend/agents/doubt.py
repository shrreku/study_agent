from typing import List, Dict, Any
import os
import logging
from llm import call_llm_for_tagging
from .retrieval import hybrid_search, fetch_chunks_by_ids, search_chunks_simple


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


def doubt_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    question = payload.get("question") or payload.get("q") or ""
    context_chunk_ids = payload.get("context_chunk_ids") or []

    chunks: List[Dict[str, Any]] = []
    if context_chunk_ids:
        chunks = fetch_chunks_by_ids(context_chunk_ids)
    if not chunks and question:
        try:
            chunks = hybrid_search(question, 5)
        except Exception:
            chunks = search_chunks_simple(question, 5)

    context_lines = []
    for idx, ch in enumerate(chunks, start=1):
        context_lines.append(f"[{idx}] {ch.get('snippet','')[:280]} (chunk_id:{ch.get('id')}, page:{ch.get('page_number')})")
    context_text = "\n".join(context_lines)

    prompt = (
        "You are a concise math tutor. Use the context snippets to answer the question.\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Return ONLY JSON: {\"answer\": \"...\", \"citations\":[{\"chunk_id\":\"...\", \"page\": 1, \"snippet\": \"...\"}], \"expanded_steps\": \"...\"}"
    )

    try:
        out = call_llm_for_tagging(prompt)
        citations = out.get("citations") or []
        if not citations and chunks:
            citations = [{"chunk_id": chunks[0].get("id"), "page": chunks[0].get("page_number"), "snippet": chunks[0].get("snippet") }]
        return {
            "answer": out.get("answer", ""),
            "citations": citations,
            "expanded_steps": out.get("expanded_steps"),
        }
    except Exception:
        logging.exception("doubt_agent_llm_failed")
        default_citations = []
        if chunks:
            default_citations = [{"chunk_id": chunks[0].get("id"), "page": chunks[0].get("page_number"), "snippet": chunks[0].get("snippet")}]
        return {"answer": "Sorry, I couldn't generate an answer right now.", "citations": default_citations, "expanded_steps": None}


