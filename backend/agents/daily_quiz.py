from typing import Dict, Any, Optional
import os
import logging
from llm import call_llm_for_tagging
from .retrieval import hybrid_search


def _retrieve_chunk_for_concept(concept: str) -> Optional[Dict[str, Any]]:
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
                    "SELECT id::text, LEFT(full_text,800) AS snippet FROM chunk WHERE full_text ILIKE %s LIMIT 1",
                    (f"%{concept}%",),
                )
                r = cur.fetchone()
                return r
        finally:
            conn.close()
    except Exception:
        logging.exception("retrieve_chunk_failed")
        return None


def daily_quiz_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    concepts = payload.get("concepts") or []
    count = int(payload.get("count") or 1)
    items = []

    for c in concepts:
        # First try hybrid search with the concept as query
        chunk = None
        try:
            results = hybrid_search(c, k=1)
            if results:
                chunk = {"id": results[0].get("id"), "snippet": results[0].get("snippet")}
        except Exception:
            logging.exception("hybrid_search_failed_for_quiz")
            chunk = _retrieve_chunk_for_concept(c)
        prompt_text = f"Generate one MCQ for the concept: {c}."
        if chunk and chunk.get("snippet"):
            prompt_text += "\nContext snippet:\n" + chunk["snippet"]

        try:
            resp = call_llm_for_tagging(prompt_text + "\nReturn JSON with keys question, options, answer_index, source_chunk_id")
            if "source_chunk_id" not in resp and chunk:
                resp["source_chunk_id"] = chunk.get("id")
            items.append(resp)
        except Exception:
            logging.exception("mcq_generation_failed")
            items.append({"question": f"What is {c}?", "options": ["A","B","C","D"], "answer_index": 0, "source_chunk_id": chunk.get("id") if chunk else None})
        if len(items) >= count:
            break
    return {"quiz": items}


