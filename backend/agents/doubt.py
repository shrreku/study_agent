from typing import List, Dict, Any
import os
import logging
from llm import call_llm_json
from .retrieval import hybrid_search, fetch_chunks_by_ids, search_chunks_simple, consolidate_adjacent_microchunks, dedup_by_id


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
    # Accept multiple aliases for question to remain compatible with UI
    question = payload.get("question") or payload.get("question_text") or payload.get("q") or ""
    context_chunk_ids = payload.get("context_chunk_ids") or []

    chunks: List[Dict[str, Any]] = []
    if context_chunk_ids:
        chunks = fetch_chunks_by_ids(context_chunk_ids)
    if not chunks and question:
        try:
            chunks = hybrid_search(question, 12)
        except Exception:
            chunks = search_chunks_simple(question, 12)

    # Micro-chunk adaptation: deduplicate and consolidate adjacent items for context
    chunks = dedup_by_id(chunks)
    consolidated = consolidate_adjacent_microchunks(chunks, window=2)

    context_lines = []
    for idx, group in enumerate(consolidated[:6], start=1):
        # show first chunk id as representative, include list in citations
        rep_id = group.get("chunk_ids")[0] if group.get("chunk_ids") else None
        context_lines.append(f"[{idx}] {group.get('snippet','')[:320]} (chunk_id:{rep_id}, page:{group.get('page_number')})")
    context_text = "\n".join(context_lines)

    prompt = (
        "You are a concise math tutor. Use the provided context snippets to answer the question in a grounded,\n"
        "helpful way. Keep it short and focused.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Return ONLY JSON with keys: {\"answer\":string, \"citations\":[{\"chunk_id\":string, \"page\":number, \"snippet\":string}], \"expanded_steps\":string}."
    )

    try:
        default_citations = []
        if consolidated:
            for g in consolidated[:3]:
                cid = (g.get("chunk_ids") or [None])[0]
                default_citations.append({"chunk_id": cid, "page": g.get("page_number"), "snippet": g.get("snippet")})
        default = {"answer": "", "citations": default_citations, "expanded_steps": ""}
        out = call_llm_json(prompt, default)
        citations = out.get("citations") or default_citations
        if not citations and consolidated:
            # map consolidated groups to citation objects (first id + snippet)
            citations = []
            for g in consolidated[:6]:
                cid = (g.get("chunk_ids") or [None])[0]
                citations.append({"chunk_id": cid, "page": g.get("page_number"), "snippet": g.get("snippet")})
        return {
            "answer": out.get("answer", ""),
            "citations": citations,
            "expanded_steps": out.get("expanded_steps"),
        }
    except Exception:
        logging.exception("doubt_agent_llm_failed")
        default_citations = []
        if consolidated:
            for g in consolidated[:3]:
                cid = (g.get("chunk_ids") or [None])[0]
                default_citations.append({"chunk_id": cid, "page": g.get("page_number"), "snippet": g.get("snippet")})
        return {"answer": "Sorry, I couldn't generate an answer right now.", "citations": default_citations, "expanded_steps": None}


