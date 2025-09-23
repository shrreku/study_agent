"""Helpers for retrieval maintenance tasks.

Includes a helper to recompute `search_tsv` from `full_text` which is
useful when chunking strategy changes and short snippets need better
full-text indexing.
"""
import os
import logging

def _get_real_dict_cursor():
    try:
        import psycopg2
    except Exception:
        return None
    # try normal import path
    try:
        from psycopg2.extras import RealDictCursor  # type: ignore
        return RealDictCursor
    except Exception:
        # fallback: maybe tests injected a fake module with attribute 'extras'
        extras = getattr(psycopg2, "extras", None)
        if extras and hasattr(extras, "RealDictCursor"):
            return extras.RealDictCursor
    return None


def recompute_search_tsv_for_all_chunks(batch_size: int = 500):
    try:
        import psycopg2
    except Exception:
        logging.exception("psycopg2_missing")
        return 0
    RealDictCursor = _get_real_dict_cursor()

    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "app")
    dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"

    updated = 0
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, full_text FROM chunk")
            rows = cur.fetchall()
            for r in rows:
                cur.execute("UPDATE chunk SET search_tsv = to_tsvector('english', %s) WHERE id = %s", (r["full_text"], r["id"]))
                updated += 1
        conn.commit()
    finally:
        conn.close()
    return updated

"""Shared retrieval helpers for agents.

Provides simple Postgres ILIKE-based retrieval and a hybrid search that blends
pgvector similarity with full-text ranking (BM25-like via ts_rank).
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import logging
import time
import embed as embed_service


def get_db_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return dsn
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "app")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def fetch_chunks_by_ids(ids: List[str]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    try:
        import psycopg2
        conn = psycopg2.connect(get_db_dsn())
        try:
            with conn.cursor(cursor_factory=_get_real_dict_cursor()) as cur:
                cur.execute(
                    "SELECT id::text, resource_id::text, page_number, source_offset, LEFT(full_text, 300) AS snippet FROM chunk WHERE id = ANY(%s::uuid[])",
                    (ids,),
                )
                return cur.fetchall()
        finally:
            conn.close()
    except Exception:
        logging.exception("fetch_chunks_by_ids_failed")
        return []


def search_chunks_simple(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    try:
        import psycopg2
        conn = psycopg2.connect(get_db_dsn())
        try:
            with conn.cursor(cursor_factory=_get_real_dict_cursor()) as cur:
                cur.execute(
                    "SELECT id::text, resource_id::text, page_number, source_offset, LEFT(full_text, 300) AS snippet FROM chunk WHERE full_text ILIKE %s LIMIT %s",
                    (f"%{query}%", limit),
                )
                return cur.fetchall()
        finally:
            conn.close()
    except Exception:
        logging.exception("search_chunks_failed")
        return []


def _register_pgvector_adapter(conn) -> None:
    try:
        from pgvector.psycopg2 import register_vector  # type: ignore
        register_vector(conn)
    except Exception:
        pass


def _embed_query(text: str) -> List[float]:
    # Use existing embedding service; returns 384-dim list[float]
    return embed_service.embed_text(text)


def hybrid_search(query: str, k: int = 10, sim_weight: Optional[float] = None, bm25_weight: Optional[float] = None, resource_boost: Optional[float] = None, page_proximity_boost: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Blend pgvector similarity with full-text rank. Falls back gracefully.

    Returns list of {id, resource_id, page_number, snippet, score} ordered by score desc.
    """
    # Read env defaults for weights and boosts
    try:
        sim_w = float(os.getenv("RETRIEVAL_SIM_WEIGHT", "0.7")) if sim_weight is None else float(sim_weight)
    except Exception:
        sim_w = 0.7
    try:
        bm25_w = float(os.getenv("RETRIEVAL_BM25_WEIGHT", "0.3")) if bm25_weight is None else float(bm25_weight)
    except Exception:
        bm25_w = 0.3
    try:
        res_boost = float(os.getenv("RETRIEVAL_RESOURCE_BOOST", "1.0")) if resource_boost is None else float(resource_boost)
    except Exception:
        res_boost = 1.0
    try:
        page_prox = os.getenv("RETRIEVAL_PAGE_PROXIMITY", "false").lower() in ("1", "true", "yes") if page_proximity_boost is None else bool(page_proximity_boost)
    except Exception:
        page_prox = False

    # Compute query embedding; if it fails, fall back to simple search
    try:
        qvec = _embed_query(query)
    except Exception:
        logging.exception("hybrid_embed_failed_fallback_simple")
        return search_chunks_simple(query, k)

    try:
        import psycopg2
        conn = psycopg2.connect(get_db_dsn())
        _register_pgvector_adapter(conn)
        try:
            with conn.cursor(cursor_factory=_get_real_dict_cursor()) as cur:
                # Compute similarity and text rank; coalesce NULLs to zero
                # Optionally boost by resource-level weight or page proximity (simple heuristic)
                base_query = """
                    SELECT
                      id::text,
                      resource_id::text,
                      page_number,
                      source_offset,
                      LEFT(full_text, 300) AS snippet,
                      COALESCE(1 - (embedding <=> %s::vector), 0.0) AS sim,
                      COALESCE(ts_rank_cd(search_tsv, plainto_tsquery('english', %s)), 0.0) AS bm25
                    FROM chunk
                """

                # Fetch candidates first (k * 3) to allow re-ranking with boosts locally
                # Ensure the query embedding is sent as a vector literal so that pgvector can apply <=> correctly
                try:
                    qvec_lit = "[" + ",".join(f"{float(x):.6f}" for x in qvec) + "]"
                except Exception:
                    # If anything odd, fall back to simple search rather than erroring
                    logging.exception("hybrid_qvec_literal_build_failed")
                    return search_chunks_simple(query, k)

                cur.execute(base_query + " LIMIT %s", (qvec_lit, query, max(50, k * 5)))
                candidates = cur.fetchall()

                # Compute combined score in Python to allow flexible fusion and boosts
                results: List[Dict[str, Any]] = []
                for r in candidates:
                    sim = float(r.get("sim") or 0.0)
                    bm25 = float(r.get("bm25") or 0.0)
                    score = sim * sim_w + bm25 * bm25_w
                    # resource boost: simple heuristic multiply if resource id matches useful pattern
                    try:
                        score *= res_boost
                    except Exception:
                        pass
                    # page proximity boost: favor lower page numbers as they often contain summaries
                    if page_prox and r.get("page_number") is not None:
                        # pages closer to 1 get small boost
                        page = int(r.get("page_number") or 0)
                        proximity_boost = max(1.0, 1.0 + max(0, (10 - page)) * 0.02)
                        score *= proximity_boost
                    results.append({"id": r["id"], "resource_id": r["resource_id"], "page_number": r["page_number"], "source_offset": r.get("source_offset"), "snippet": r["snippet"], "sim": sim, "bm25": bm25, "score": score})

                # sort and return top-k
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:k]
        finally:
            conn.close()
    except Exception:
        logging.exception("hybrid_search_failed_fallback_simple")
        return search_chunks_simple(query, k)



# --- Micro-chunk utilities ---
def dedup_by_id(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for c in chunks:
        cid = c.get("id")
        if cid and cid not in seen:
            seen.add(cid)
            out.append(c)
    return out


def diversify_by_page(chunks: List[Dict[str, Any]], per_page: int = 1) -> List[Dict[str, Any]]:
    page_counts: Dict[Tuple[str, int], int] = {}
    diversified: List[Dict[str, Any]] = []
    for c in chunks:
        key = (str(c.get("resource_id")), int(c.get("page_number") or 0))
        cnt = page_counts.get(key, 0)
        if cnt < per_page:
            diversified.append(c)
            page_counts[key] = cnt + 1
    return diversified


def consolidate_adjacent_microchunks(chunks: List[Dict[str, Any]], window: int = 2) -> List[Dict[str, Any]]:
    """Group adjacent micro-chunks per resource/page into small windows.

    If source_offset is available, use it for ordering; otherwise preserve input order.
    Returns list of dicts: {chunk_ids, resource_id, page_number, snippet}
    """
    if not chunks:
        return []
    # group by (resource_id, page_number)
    from collections import defaultdict

    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        key = (str(c.get("resource_id")), int(c.get("page_number") or 0))
        groups[key].append(c)

    consolidated: List[Dict[str, Any]] = []
    for key, items in groups.items():
        # sort by source_offset if available
        items.sort(key=lambda x: (x.get("source_offset") is None, int(x.get("source_offset") or 0)))
        i = 0
        while i < len(items):
            win = items[i : i + window]
            chunk_ids = [w.get("id") for w in win if w.get("id")]
            snippet = " ".join((w.get("snippet") or "") for w in win)[:600]
            consolidated.append({
                "chunk_ids": chunk_ids,
                "resource_id": key[0],
                "page_number": key[1],
                "snippet": snippet,
            })
            i += window
    # preserve roughly the original ordering by page then by presence in input
    consolidated.sort(key=lambda x: (x.get("resource_id"), int(x.get("page_number") or 0)))
    return consolidated

