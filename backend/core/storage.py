"""Storage utilities (MinIO and optional GCS helpers).

Exposes:
- get_minio_client(): returns configured Minio client
- is_gcs_enabled(): whether NOTES_GCS_BUCKET is configured
- get_notes_bucket(): active bucket name for notes (GCS or MinIO fallback)
- upload_bytes_to_object(): upload to active storage backend and return storage_path
- download_object_to_path(): download from active storage backend to a local path
"""
from __future__ import annotations
import os
import io
from datetime import timedelta
from typing import Tuple


def get_minio_client():
    try:
        from minio import Minio  # lazy import to avoid test-time dependency
    except Exception as e:
        raise RuntimeError("minio_dependency_missing") from e
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    access_key = os.getenv("MINIO_ROOT_USER", "minioadmin")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
    secure = os.getenv("MINIO_SECURE", "false").lower() in ("1", "true", "yes")
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def is_gcs_enabled() -> bool:
    """Return True if NOTES_GCS_BUCKET is configured.

    In GCP/Cloud Run, this should be set and the environment configured
    for Application Default Credentials so the GCS client can authenticate.
    """

    return bool(os.getenv("NOTES_GCS_BUCKET"))


def get_notes_bucket() -> str:
    """Get the bucket name to use for notes/resources.

    Prefers NOTES_GCS_BUCKET when set (GCS in prod) and falls back to
    MINIO_BUCKET for local/dev.
    """

    if is_gcs_enabled():
        return os.getenv("NOTES_GCS_BUCKET", "study-agent-notes")
    return os.getenv("MINIO_BUCKET", "resources")


def _get_gcs_client():
    """Return a google.cloud.storage.Client instance.

    Lazy import so local/dev without GCS does not require the dependency.
    """

    try:
        from google.cloud import storage  # type: ignore
    except Exception as e:
        raise RuntimeError("gcs_dependency_missing") from e
    return storage.Client()


def upload_bytes_to_object(data: bytes, filename: str, content_type: str | None = None) -> str:
    """Upload raw bytes to the active storage backend and return storage_path.

    storage_path uses the form "bucket/object_name" so existing DB and
    worker code can split it into (bucket, object).
    """

    bucket = get_notes_bucket()
    object_name = f"{os.getenv('STORAGE_PREFIX', '')}{os.urandom(8).hex()}_{filename}"

    if is_gcs_enabled():
        client = _get_gcs_client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(object_name)
        blob.upload_from_string(data, content_type=content_type or "application/octet-stream")
    else:
        # Direct upload to MinIO using the internal endpoint
        minio_client = get_minio_client()
        try:
            if not minio_client.bucket_exists(bucket):
                minio_client.make_bucket(bucket)
        except Exception:
            # best-effort bucket creation
            pass
        data_stream = io.BytesIO(data)
        minio_client.put_object(
            bucket,
            object_name,
            data=data_stream,
            length=len(data),
            content_type=content_type,
        )

    return f"{bucket}/{object_name}"


def generate_presigned_upload_url(object_name: str, content_type: str | None = None, expires_seconds: int = 3600) -> str:
    """Generate a presigned upload URL for the given object name.

    Uses NOTES_GCS_BUCKET when GCS is enabled, otherwise falls back to MinIO.
    """

    bucket = get_notes_bucket()

    if is_gcs_enabled():
        client = _get_gcs_client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(object_name)
        url = blob.generate_signed_url(
            version="v4",
            method="PUT",
            expiration=timedelta(seconds=expires_seconds),
            content_type=content_type or "application/octet-stream",
        )
        return url

    # For MinIO, allow overriding the endpoint used for signing so that
    # signatures match the hostname the browser will call (e.g. localhost:9000).
    presign_endpoint = os.getenv("MINIO_PRESIGN_ENDPOINT")
    if presign_endpoint:
        try:
            from minio import Minio  # type: ignore
        except Exception as e:
            raise RuntimeError("minio_dependency_missing") from e
        access_key = os.getenv("MINIO_ROOT_USER", "minioadmin")
        secret_key = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
        secure = os.getenv("MINIO_SECURE", "false").lower() in ("1", "true", "yes")
        minio_client = Minio(presign_endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    else:
        minio_client = get_minio_client()

    # Ensure bucket exists (best-effort) before generating presigned PUT URL
    try:
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)
    except Exception:
        # If this fails, presigned URL generation may still work; PUT will surface errors
        pass

    # MinIO presigned PUT; expires is a timedelta in recent clients
    url = minio_client.presigned_put_object(
        bucket,
        object_name,
        expires=timedelta(seconds=expires_seconds),
    )

    return url


def download_object_to_path(storage_path: str, dest_path: str) -> None:
    """Download an object referenced by storage_path to dest_path.

    storage_path is expected to be "bucket/object".
    """

    if "/" not in storage_path:
        raise ValueError("storage_path must be of form 'bucket/object'")
    bucket, obj = storage_path.split("/", 1)

    if is_gcs_enabled():
        client = _get_gcs_client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(obj)
        blob.download_to_filename(dest_path)
    else:
        minio_client = get_minio_client()
        minio_client.fget_object(bucket, obj, dest_path)


def notes_raw_object_name(user_id: str, file_id: str) -> str:
    """Return the recommended object name for raw user uploads.

    Layout: user_{user_id}/raw/{file_id}
    """

    return f"user_{user_id}/raw/{file_id}"


def notes_processed_object_name(user_id: str, resource_id: str) -> str:
    """Return the recommended object name for processed artifacts.

    Layout: user_{user_id}/processed/{resource_id}
    """

    return f"user_{user_id}/processed/{resource_id}"
