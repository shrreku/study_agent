"""Storage utilities (MinIO client).

Exposes:
- get_minio_client(): returns configured Minio client
"""
from __future__ import annotations
import os


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
