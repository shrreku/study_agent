from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import time
import logging
from psycopg2.extras import RealDictCursor

from core.auth import require_auth
from core.db import get_db_conn
from ingestion import embed as embed_service

router = APIRouter()


class UpsertRequest(BaseModel):
    chunk_ids: Optional[List[str]] = None
    texts: Optional[List[str]] = None
    embedding_version: Optional[str] = None


@router.post("/api/embeddings/upsert")
async def embeddings_upsert(req: UpsertRequest, token: str = Depends(require_auth)):
    if not req.texts and not req.chunk_ids:
        raise HTTPException(status_code=400, detail="provide texts or chunk_ids to embed")

    conn = get_db_conn()
    try:
        logging.info(
            "emb_upsert_start texts=%d chunk_ids=%d version=%s",
            len(req.texts or []),
            len(req.chunk_ids or []),
            (req.embedding_version or os.getenv("EMBED_VERSION", "all-MiniLM-L6-v2-2025-09")),
        )
        if req.texts:
            texts = req.texts
            t0 = time.time()
            vecs = embed_service.embed_texts(texts)
            with conn.cursor() as cur:
                for i, v in enumerate(vecs):
                    cur.execute("SELECT uuid_generate_v4()::text")
                    new_id = cur.fetchone()[0]
                    vec_lit = "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"
                    cur.execute(
                        "INSERT INTO chunk (id, full_text, embedding, embedding_version, created_at) VALUES (%s,%s,%s::vector,%s,now())",
                        (new_id, texts[i], vec_lit, req.embedding_version or os.getenv("EMBED_VERSION", "all-MiniLM-L6-v2-2025-09")),
                    )
            conn.commit()
            t_ms = int((time.time() - t0) * 1000)
            logging.info("emb_upsert_texts inserted=%d took_ms=%d", len(vecs), t_ms)
            return {"inserted": len(vecs)}

        if req.chunk_ids:
            normalized_ids: List[str] = []
            for cid in req.chunk_ids:
                try:
                    normalized_ids.append(str(uuid.UUID(str(cid))))
                except Exception:
                    continue
            if not normalized_ids:
                logging.warning("emb_upsert_no_valid_chunk_ids provided=%d", len(req.chunk_ids or []))
                raise HTTPException(status_code=400, detail="no valid chunk_ids provided")

            target_ver = req.embedding_version or os.getenv("EMBED_VERSION", "all-MiniLM-L6-v2-2025-09")
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, full_text FROM chunk WHERE id = ANY(%s::uuid[]) AND (embedding IS NULL OR embedding_version IS DISTINCT FROM %s)",
                    (normalized_ids, target_ver),
                )
                rows = cur.fetchall()
            texts = [r["full_text"] for r in rows]
            ids = [str(r["id"]) for r in rows]
            t0 = time.time()
            if not texts:
                logging.info("emb_upsert_chunks nothing_to_update target_ver=%s", target_ver)
                return {"updated": 0}
            vecs = embed_service.embed_texts(texts)
            with conn.cursor() as cur:
                for cid, vec in zip(ids, vecs):
                    vec_lit = "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"
                    cur.execute(
                        "UPDATE chunk SET embedding=%s::vector, embedding_version=%s, updated_at=now() WHERE id=%s::uuid",
                        (vec_lit, target_ver, cid),
                    )
            conn.commit()
            t_ms = int((time.time() - t0) * 1000)
            logging.info("emb_upsert_chunks updated=%d fetched=%d target_ver=%s took_ms=%d", len(vecs), len(rows), target_ver, t_ms)
            return {"updated": len(vecs)}
    finally:
        conn.close()
