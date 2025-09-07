"""Shared retrieval helpers for agents.

Provides simple Postgres ILIKE-based retrieval and a hybrid search that blends
pgvector similarity with full-text ranking (BM25-like via ts_rank).
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import logging
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
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(get_db_dsn())
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


def search_chunks_simple(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(get_db_dsn())
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id::text, resource_id::text, page_number, LEFT(full_text, 300) AS snippet FROM chunk WHERE full_text ILIKE %s LIMIT %s",
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


def hybrid_search(query: str, k: int = 10, sim_weight: float = 0.7, bm25_weight: float = 0.3) -> List[Dict[str, Any]]:
    """Blend pgvector similarity with full-text rank. Falls back gracefully.

    Returns list of {id, resource_id, page_number, snippet, score} ordered by score desc.
    """
    # Compute query embedding; if it fails, fall back to simple search
    try:
        qvec = _embed_query(query)
    except Exception:
        logging.exception("hybrid_embed_failed_fallback_simple")
        return search_chunks_simple(query, k)

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(get_db_dsn())
        _register_pgvector_adapter(conn)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Compute similarity and text rank; coalesce NULLs to zero
                cur.execute(
                    """
                    SELECT
                      id::text,
                      resource_id::text,
                      page_number,
                      LEFT(full_text, 300) AS snippet,
                      COALESCE(1 - (embedding <=> %s), 0.0) AS sim,
                      COALESCE(ts_rank_cd(search_tsv, plainto_tsquery('english', %s)), 0.0) AS bm25,
                      (COALESCE(1 - (embedding <=> %s), 0.0) * %s + COALESCE(ts_rank_cd(search_tsv, plainto_tsquery('english', %s)), 0.0) * %s) AS score
                    FROM chunk
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (qvec, query, qvec, sim_weight, query, bm25_weight, k),
                )
                rows = cur.fetchall()
                return rows
        finally:
            conn.close()
    except Exception:
        logging.exception("hybrid_search_failed_fallback_simple")
        return search_chunks_simple(query, k)


