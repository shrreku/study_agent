from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor

from core.auth import require_auth
from core.db import get_db_conn
from core.storage import (
    generate_presigned_upload_url,
    get_notes_bucket,
    notes_raw_object_name,
    upload_bytes_to_object,
)
from api.resources import ReindexModels, reindex_resource


router = APIRouter()


class NotesUploadUrlRequest(BaseModel):
    filename: str
    size_bytes: int
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class NotesUploadUrlResponse(BaseModel):
    resource_id: str
    upload_url: str
    status: Optional[str]
    original_filename: str
    gcs_path_raw: str
    metadata: Optional[Dict[str, Any]] = None


@router.post("/api/notes/upload", response_model=NotesUploadUrlResponse)
async def upload_notes_file(
    file: UploadFile = File(...),
    user_id: str = Depends(require_auth),
) -> NotesUploadUrlResponse:
    MAX_BYTES = 100 * 1024 * 1024
    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")

    try:
        storage_path = upload_bytes_to_object(
            contents,
            file.filename,
            file.content_type or "application/octet-stream",
        )
    except Exception as e:
        logging.exception("notes_upload_failed")
        raise HTTPException(status_code=500, detail=f"failed_to_store_file: {e}")

    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            resource_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO resource (id, user_id, title, filename, content_type, size_bytes, storage_path, status, error_message, metadata, created_at)
                VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, now())
                RETURNING id, filename, storage_path, status, metadata
                """,
                (
                    resource_id,
                    user_id,
                    file.filename,
                    file.filename,
                    file.content_type,
                    len(contents),
                    storage_path,
                    "queued",
                    None,
                    None,
                ),
            )
            row = cur.fetchone()
            conn.commit()
    finally:
        conn.close()

    return NotesUploadUrlResponse(
        resource_id=str(row["id"]),
        upload_url="",  # not used in direct upload mode
        status=row.get("status"),
        original_filename=row.get("filename"),
        gcs_path_raw=row.get("storage_path"),
        metadata=row.get("metadata"),
    )


class NotesResource(BaseModel):
    resource_id: str
    original_filename: str
    gcs_path_raw: str
    status: Optional[str]
    error_message: Optional[str]
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


def _row_to_notes_resource(row: Dict[str, Any]) -> NotesResource:
    return NotesResource(
        resource_id=str(row["id"]),
        original_filename=row.get("filename") or row.get("title") or "",
        gcs_path_raw=row.get("storage_path") or "",
        status=row.get("status"),
        error_message=row.get("error_message"),
        metadata=row.get("metadata"),
        created_at=row.get("created_at").isoformat() if row.get("created_at") is not None else None,
    )


@router.post("/api/notes/upload-url", response_model=NotesUploadUrlResponse)
async def create_notes_upload_url(
    payload: NotesUploadUrlRequest,
    user_id: str = Depends(require_auth),
) -> NotesUploadUrlResponse:
    MAX_BYTES = 100 * 1024 * 1024
    if payload.size_bytes > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")

    file_id = str(uuid.uuid4())
    object_name = notes_raw_object_name(user_id, file_id)

    try:
        upload_url = generate_presigned_upload_url(object_name, content_type=payload.content_type)
    except Exception as e:
        logging.exception("failed_to_generate_presigned_url")
        raise HTTPException(status_code=500, detail=f"failed_to_generate_upload_url: {e}")

    storage_path = f"{get_notes_bucket()}/{object_name}"

    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            metadata_val: Optional[Any]
            if payload.metadata is not None:
                metadata_val = psycopg2.extras.Json(payload.metadata)
            else:
                metadata_val = None

            resource_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO resource (id, user_id, title, filename, content_type, size_bytes, storage_path, status, error_message, metadata, created_at)
                VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, now())
                RETURNING id, filename, storage_path, status, metadata
                """,
                (
                    resource_id,
                    user_id,
                    payload.filename,
                    payload.filename,
                    payload.content_type,
                    payload.size_bytes,
                    storage_path,
                    "queued",
                    None,
                    metadata_val,
                ),
            )
            row = cur.fetchone()
            conn.commit()
    finally:
        conn.close()

    return NotesUploadUrlResponse(
        resource_id=str(row["id"]),
        upload_url=upload_url,
        status=row.get("status"),
        original_filename=row.get("filename"),
        gcs_path_raw=row.get("storage_path"),
        metadata=row.get("metadata"),
    )


@router.post("/api/notes/{resource_id}/ingest")
async def ingest_notes_resource(
    resource_id: str,
    user_id: str = Depends(require_auth),
):
    if not resource_id or not resource_id.strip():
        raise HTTPException(status_code=400, detail="resource_id required")

    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, user_id, status
                FROM resource
                WHERE id=%s::uuid AND user_id=%s::uuid
                """,
                (resource_id, user_id),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="resource not found")

            cur.execute(
                "UPDATE resource SET status=%s, error_message=NULL WHERE id=%s::uuid",
                ("parsing", resource_id),
            )
            conn.commit()
    finally:
        conn.close()

    ingest_model = os.getenv("INGEST_LLM_MODEL") or os.getenv("INGEST_MODEL_HINT")
    models: Optional[ReindexModels] = None
    if ingest_model:
        models = ReindexModels(ingest_model=ingest_model)

    try:
        await reindex_resource(resource_id, models=models, token=user_id)
    except HTTPException as e:
        logging.exception(
            "notes_ingest_failed_http",
            extra={"resource_id": resource_id, "status_code": getattr(e, "status_code", None)},
        )
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                message = getattr(e, "detail", None)
                if not isinstance(message, str):
                    message = str(message)
                cur.execute(
                    "UPDATE resource SET status=%s, error_message=%s WHERE id=%s::uuid",
                    ("failed", (message or "")[:500], resource_id),
                )
                conn.commit()
        finally:
            conn.close()
        raise
    except Exception as e:
        logging.exception("notes_ingest_failed", extra={"resource_id": resource_id})
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE resource SET status=%s, error_message=%s WHERE id=%s::uuid",
                    ("failed", str(e)[:500], resource_id),
                )
                conn.commit()
        finally:
            conn.close()
        raise HTTPException(status_code=500, detail="ingestion_failed")

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE resource SET status=%s WHERE id=%s::uuid",
                ("ready", resource_id),
            )
            conn.commit()
    finally:
        conn.close()

    return {"resource_id": resource_id, "status": "ready"}


@router.get("/api/notes/{resource_id}", response_model=NotesResource)
async def get_notes_resource(
    resource_id: str,
    user_id: str = Depends(require_auth),
) -> NotesResource:
    if not resource_id or not resource_id.strip():
        raise HTTPException(status_code=400, detail="resource_id required")

    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, filename, storage_path, status, error_message, metadata, created_at
                FROM resource
                WHERE id=%s::uuid AND user_id=%s::uuid
                """,
                (resource_id, user_id),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="resource not found")
            return _row_to_notes_resource(row)
    finally:
        conn.close()


@router.get("/api/notes", response_model=List[NotesResource])
async def list_notes_resources(user_id: str = Depends(require_auth)) -> List[NotesResource]:
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, filename, storage_path, status, error_message, metadata, created_at
                FROM resource
                WHERE user_id=%s::uuid
                ORDER BY created_at DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
            return [_row_to_notes_resource(r) for r in rows]
    finally:
        conn.close()


@router.delete("/api/notes/{resource_id}")
async def delete_notes_resource(
    resource_id: str,
    user_id: str = Depends(require_auth),
) -> Dict[str, Any]:
    if not resource_id or not resource_id.strip():
        raise HTTPException(status_code=400, detail="resource_id required")

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "UPDATE job SET resource_id=NULL WHERE resource_id=%s::uuid",
                    (resource_id,),
                )
            except Exception:
                logging.exception("notes_delete_clear_jobs_failed", extra={"resource_id": resource_id})

            cur.execute(
                "DELETE FROM resource WHERE id=%s::uuid AND user_id=%s::uuid",
                (resource_id, user_id),
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="resource not found")
            conn.commit()
    finally:
        conn.close()

    return {"resource_id": resource_id, "deleted": True}
