from typing import Dict, Any, Optional
import os
import logging
from llm import call_llm_json
from .retrieval import hybrid_search, diversify_by_page


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
            results = hybrid_search(c, k=8)
            results = diversify_by_page(results, per_page=1)
            if results:
                r0 = results[0]
                chunk = {"id": r0.get("id"), "snippet": r0.get("snippet")}
        except Exception:
            logging.exception("hybrid_search_failed_for_quiz")
            chunk = _retrieve_chunk_for_concept(c)
        # Build a strict JSON prompt for MCQ generation
        prompt_lines = [
            "Create ONE multiple-choice question grounded in the academic context.",
            f"Concept: {c}",
        ]
        if chunk and chunk.get("snippet"):
            prompt_lines.append("Context snippet:\n" + str(chunk.get("snippet"))[:800])
        prompt_lines.append(
            "Return ONLY JSON with keys: {\"question\":string, \"options\":[string,string,string,string], \"answer_index\":number (0-3), \"explanation\":string}."
        )
        prompt = "\n\n".join(prompt_lines)

        try:
            default = {
                "question": f"What best describes {c}?",
                "options": ["Definition", "Example", "Property", "None of the above"],
                "answer_index": 0,
                "explanation": "",
            }
            resp = call_llm_json(prompt, default)
            # attach source and references for grounding
            if chunk:
                resp["source_chunk_id"] = chunk.get("id")
                resp["references"] = [{"chunk_id": chunk.get("id"), "snippet": (chunk.get("snippet") or "")[:200]}]
            items.append(resp)
        except Exception:
            logging.exception("mcq_generation_failed")
            items.append({
                "question": f"What is {c}?",
                "options": ["A", "B", "C", "D"],
                "answer_index": 0,
                "explanation": "",
                "source_chunk_id": chunk.get("id") if chunk else None,
            })
        if len(items) >= count:
            break
    # Return only 'quiz' (UI can adapt); 'items' deprecated
    return {"quiz": items}


