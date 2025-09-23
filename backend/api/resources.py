from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from typing import Optional, List, Dict, Any
import os
import io
import uuid
import tempfile
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

from core.auth import require_auth
from core.db import get_db_conn
from core.storage import get_minio_client
from core.kg import merge_concepts_in_neo4j
from metrics import MetricsCollector
from llm import tag_and_extract
import embed as embed_service
from chunker import structural_chunk_resource

router = APIRouter()


@router.post("/api/resources/upload")
async def upload_resource(file: UploadFile = File(...), title: str = "", token: str = Depends(require_auth)):
    MAX_BYTES = 100 * 1024 * 1024
    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")

    minio_client = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET", "resources")
    try:
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)
    except Exception:
        pass

    object_name = f"{uuid.uuid4()}_{file.filename}"
    try:
        minio_client.put_object(bucket, object_name, data=io.BytesIO(contents), length=len(contents), content_type=file.content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store object: {e}")

    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO resource (id, title, filename, content_type, size_bytes, storage_path, created_at) VALUES (%s,%s,%s,%s,%s,%s,now()) RETURNING id, title, filename, size_bytes",
                (str(uuid.uuid4()), title or file.filename, file.filename, file.content_type, len(contents), f"{bucket}/{object_name}")
            )
            row = cur.fetchone()
            conn.commit()
    finally:
        conn.close()

    # enqueue parse job placeholder (best-effort) and return resource details
    job_id = None
    try:
        from redis import Redis  # type: ignore
        from rq import Queue  # type: ignore
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        redis = Redis.from_url(redis_url)
        q = Queue("parse", connection=redis)
        job_id = str(uuid.uuid4())
        payload = {"resource_id": row["id"], "storage_path": f"{bucket}/{object_name}"}

        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO job (id, resource_id, type, status, payload, created_at, updated_at) VALUES (%s,%s,%s,%s,%s,now(),now())",
                    (job_id, row["id"], "parse", "queued", psycopg2.extras.Json(payload)),
                )
                conn.commit()
        finally:
            conn.close()

        q.enqueue_call(func="backend.worker.process_parse_job", args=(job_id, row["id"], payload["storage_path"]))
    except Exception:
        job_id = None

    return {"resource_id": row["id"], "title": row["title"], "size": row["size_bytes"], "job_id": job_id}


@router.post("/api/resources/{resource_id}/reindex")
async def reindex_resource(resource_id: str, token: str = Depends(require_auth)):
    """Incrementally reindex a resource by diffing structural chunks.

    - Recompute structural chunks
    - Diff against existing by (page_number, source_offset)
    - Insert new, update changed, delete removed
    - Re-run LLM tagging and update concepts
    - Update embeddings for new/changed
    """
    if not resource_id or not resource_id.strip():
        raise HTTPException(status_code=400, detail="resource_id required")

    # Resolve storage path for resource
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT storage_path FROM resource WHERE id=%s::uuid", (resource_id,))
            r = cur.fetchone()
            if not r:
                raise HTTPException(status_code=404, detail="resource not found")
            storage = r["storage_path"]
    finally:
        conn.close()

    # Locate or download file
    local_path = None
    sample_dir = os.path.join(os.getcwd(), "sample")
    if os.path.isdir(sample_dir):
        fname = storage.split("/")[-1]
        for root, _, files in os.walk(sample_dir):
            if fname in files:
                local_path = os.path.join(root, fname)
                break
    if not local_path and os.path.exists(storage):
        local_path = storage

    tmp_download_path = None
    if not local_path:
        try:
            minio_client = get_minio_client()
            bucket, obj = storage.split("/", 1)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(obj)[1] or "")
            tf.close()
            tmp_download_path = tf.name
            minio_client.fget_object(bucket, obj, tmp_download_path)
            local_path = tmp_download_path
        except Exception as e:
            if tmp_download_path and os.path.exists(tmp_download_path):
                try:
                    os.unlink(tmp_download_path)
                except Exception:
                    pass
            raise HTTPException(status_code=400, detail=f"resource not available locally and MinIO download failed: {e}")

    # Compute new structural chunks
    new_chunks = structural_chunk_resource(local_path)

    def key_of(c: Dict[str, Any]) -> str:
        return f"{int(c.get('page_number') or 0)}:{int(c.get('source_offset') or 0)}"

    new_map: Dict[str, Dict[str, Any]] = {key_of(c): c for c in new_chunks}

    # Fetch existing for resource
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id::text, page_number, source_offset, full_text
                FROM chunk
                WHERE resource_id=%s::uuid
                """,
                (resource_id,),
            )
            existing_rows = cur.fetchall()
    finally:
        conn.close()

    existing_map: Dict[str, Dict[str, Any]] = {}
    for row in existing_rows:
        k = f"{int(row.get('page_number') or 0)}:{int(row.get('source_offset') or 0)}"
        if k not in existing_map:
            existing_map[k] = row

    to_insert_keys = [k for k in new_map.keys() if k not in existing_map]
    to_delete_keys = [k for k in existing_map.keys() if k not in new_map]
    to_update_keys: List[str] = []
    unchanged = 0
    for k in new_map.keys():
        if k in existing_map:
            if (existing_map[k].get("full_text") or "") != (new_map[k].get("full_text") or ""):
                to_update_keys.append(k)
            else:
                unchanged += 1

    to_insert = [new_map[k] for k in to_insert_keys]
    to_update = [(existing_map[k]["id"], new_map[k]) for k in to_update_keys]
    to_delete_ids = [existing_map[k]["id"] for k in to_delete_keys]

    inserted = updated = deleted = 0

    # Deletes
    if to_delete_ids:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunk WHERE id = ANY(%s::uuid[])", (to_delete_ids,))
            conn.commit()
            deleted = len(to_delete_ids)
        finally:
            conn.close()

    def _tag(text: str) -> Dict[str, Any]:
        try:
            return tag_and_extract(text)
        except Exception:
            return {"chunk_type": None, "concepts": [], "math_expressions": []}

    # Inserts
    if to_insert:
        texts = [c.get("full_text") or "" for c in to_insert]
        tags_list = [_tag(t) for t in texts]
        vecs = embed_service.embed_texts(texts)
        embed_version = os.getenv("EMBED_VERSION", "all-MiniLM-L6-v2-2025-09")
        conn = get_db_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for c, tags, vec in zip(to_insert, tags_list, vecs):
                    cur.execute(
                        """
                        INSERT INTO chunk (id, resource_id, page_number, source_offset, full_text,
                                           chunk_type, concepts, math_expressions, embedding, embedding_version, created_at, updated_at)
                        VALUES (uuid_generate_v4(), %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, now(), now())
                        RETURNING id::text
                        """,
                        (
                            resource_id,
                            c.get("page_number"),
                            c.get("source_offset"),
                            c.get("full_text"),
                            tags.get("chunk_type"),
                            tags.get("concepts"),
                            tags.get("math_expressions"),
                            vec,
                            embed_version,
                        ),
                    )
                    new_id = cur.fetchone()["id"]
                    try:
                        concepts = tags.get("concepts") or []
                        if concepts:
                            merge_concepts_in_neo4j(concepts, new_id, (c.get("full_text") or "")[:160], resource_id)
                    except Exception:
                        logging.exception("kg_merge_failed")
            conn.commit()
            inserted = len(to_insert)
        finally:
            conn.close()

    # Updates
    if to_update:
        texts_upd = [c.get("full_text") or "" for (_id, c) in to_update]
        tags_upd = [_tag(t) for t in texts_upd]
        vecs_upd = embed_service.embed_texts(texts_upd)
        embed_version = os.getenv("EMBED_VERSION", "all-MiniLM-L6-v2-2025-09")
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                for (chunk_id, c), tags, vec in zip(to_update, tags_upd, vecs_upd):
                    cur.execute(
                        """
                        UPDATE chunk
                        SET full_text=%s, chunk_type=%s, concepts=%s, math_expressions=%s,
                            embedding=%s, embedding_version=%s, updated_at=now()
                        WHERE id=%s::uuid
                        """,
                        (
                            c.get("full_text"),
                            tags.get("chunk_type"),
                            tags.get("concepts"),
                            tags.get("math_expressions"),
                            vec,
                            embed_version,
                            chunk_id,
                        ),
                    )
                    try:
                        concepts = tags.get("concepts") or []
                        if concepts:
                            merge_concepts_in_neo4j(concepts, str(chunk_id), (c.get("full_text") or "")[:160], resource_id)
                    except Exception:
                        logging.exception("kg_merge_failed")
            conn.commit()
            updated = len(to_update)
        finally:
            conn.close()

    # Cleanup temp download
    if tmp_download_path:
        try:
            os.unlink(tmp_download_path)
        except Exception:
            pass

    # Metrics
    try:
        mc = MetricsCollector.get_global()
        mc.increment("reindex_calls")
        mc.timing("reindex_changes", inserted + updated + deleted)
    except Exception:
        logging.exception("metrics_record_failed")

    return {
        "resource_id": resource_id,
        "inserted": inserted,
        "updated": updated,
        "deleted": deleted,
        "unchanged": unchanged,
        "total_new": len(new_chunks),
        "total_existing": len(existing_rows),
    }




@router.post("/api/resources/{resource_id}/chunk")
async def create_chunks(resource_id: str, force: bool = False, token: str = Depends(require_auth)):
    if not resource_id or not resource_id.strip():
        raise HTTPException(status_code=400, detail="resource_id required")

    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) FROM chunk WHERE resource_id=%s::uuid", (resource_id,))
            existing_count = int(cur.fetchone()["count"]) if cur.description else 0
        if existing_count > 0 and not force:
            logging.info("create_chunks_skip existing=%d resource_id=%s", existing_count, resource_id)
            return {"chunks_created": 0, "skipped": True, "existing": existing_count}
    finally:
        conn.close()

    # fetch resource storage_path from DB
    conn = get_db_conn()
    storage = None
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT storage_path FROM resource WHERE id=%s", (resource_id,))
            r = cur.fetchone()
            if not r:
                raise HTTPException(status_code=404, detail="resource not found")
            storage = r["storage_path"]
    finally:
        conn.close()

    # resolve local path (check sample/ or absolute path), else download from MinIO
    local_path = None
    sample_dir = os.path.join(os.getcwd(), "sample")
    if os.path.isdir(sample_dir):
        fname = storage.split("/")[-1]
        for root, _, files in os.walk(sample_dir):
            if fname in files:
                local_path = os.path.join(root, fname)
                break
    if not local_path and os.path.exists(storage):
        local_path = storage

    tmp_download_path = None
    if not local_path:
        try:
            minio_client = get_minio_client()
            bucket, obj = storage.split("/", 1)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(obj)[1] or "")
            tf.close()
            tmp_download_path = tf.name
            minio_client.fget_object(bucket, obj, tmp_download_path)
            local_path = tmp_download_path
        except Exception as e:
            if tmp_download_path and os.path.exists(tmp_download_path):
                try:
                    os.unlink(tmp_download_path)
                except Exception:
                    pass
            raise HTTPException(status_code=400, detail=f"resource not available locally and MinIO download failed: {e}")

    chunks = structural_chunk_resource(local_path)

    # metrics
    try:
        mc = MetricsCollector.get_global()
        mc.increment(f"resource_{resource_id}_chunks_created", len(chunks))
        mc.increment("total_chunk_jobs")
        mc.timing("last_chunk_job_chunks", len(chunks))
    except Exception:
        logging.exception("metrics_record_failed")

    conn = get_db_conn()
    inserted = 0
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for c in chunks:
                try:
                    tags = tag_and_extract(c["full_text"])  # may raise; handled below
                except Exception:
                    tags = {"chunk_type": None, "concepts": [], "math_expressions": []}

                chunk_type = tags.get("chunk_type")
                concepts = tags.get("concepts")
                math_expressions = tags.get("math_expressions")

                cur.execute(
                    "INSERT INTO chunk (id, resource_id, page_number, source_offset, full_text, chunk_type, concepts, math_expressions, created_at) VALUES (uuid_generate_v4(),%s,%s,%s,%s,%s,%s,%s,now()) RETURNING id",
                    (resource_id, c["page_number"], c["source_offset"], c["full_text"], chunk_type, concepts, math_expressions)
                )
                new_id = cur.fetchone()["id"]
                try:
                    if concepts:
                        merge_concepts_in_neo4j(concepts, new_id, c["full_text"][:160], resource_id)
                except Exception:
                    logging.exception("kg_merge_failed")
                inserted += 1
        conn.commit()
    finally:
        conn.close()

    if tmp_download_path:
        try:
            os.unlink(tmp_download_path)
        except Exception:
            pass

    return {"chunks_created": inserted}


@router.post("/api/resources/{resource_id}/parse", status_code=202)
async def start_parse_job(resource_id: str, ocr: bool = False, token: str = Depends(require_auth)):
    if not resource_id or not resource_id.strip():
        raise HTTPException(status_code=400, detail="resource_id required")

    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT storage_path FROM resource WHERE id=%s::uuid", (resource_id,))
            r = cur.fetchone()
            if not r:
                raise HTTPException(status_code=404, detail="resource not found")
            storage_path = r["storage_path"]
    finally:
        conn.close()

    try:
        from redis import Redis  # type: ignore
        from rq import Queue  # type: ignore
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        redis = Redis.from_url(redis_url)
        q = Queue("parse", connection=redis)
        job_id = str(uuid.uuid4())
        payload = {"resource_id": resource_id, "storage_path": storage_path, "ocr": bool(ocr)}

        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO job (id, resource_id, type, status, payload, created_at, updated_at) VALUES (%s,%s,%s,%s,%s,now(),now())",
                    (job_id, resource_id, "parse", "queued", psycopg2.extras.Json(payload)),
                )
                conn.commit()
        finally:
            conn.close()

        q.enqueue_call(func="backend.worker.process_parse_job", args=(job_id, resource_id, storage_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed_to_enqueue_parse: {e}")

    return {"job_id": job_id}


@router.get("/api/resources/{resource_id}/chunks")
async def list_chunks(resource_id: str, limit: int = 25, offset: int = 0, token: str = Depends(require_auth)):
    if not resource_id or not resource_id.strip():
        raise HTTPException(status_code=400, detail="resource_id required")
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id::text, page_number, chunk_type, concepts, math_expressions,
                       LEFT(full_text, 240) AS snippet, (embedding IS NOT NULL) AS has_embedding,
                       embedding_version, created_at
                FROM chunk
                WHERE resource_id=%s::uuid
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (resource_id, limit, offset),
            )
            rows = cur.fetchall()
        return {"chunks": rows, "limit": limit, "offset": offset}
    finally:
        conn.close()


@router.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str, token: str = Depends(require_auth)):
    if not job_id or not job_id.strip():
        raise HTTPException(status_code=400, detail="job_id required")
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id::text AS job_id, status, payload, created_at, updated_at
                FROM job
                WHERE id=%s::uuid
                """,
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="job not found")
            return row
    finally:
        conn.close()
