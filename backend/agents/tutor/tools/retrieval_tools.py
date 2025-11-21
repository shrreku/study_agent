from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..constants import logger
from .. import retrieval as _base_retrieval


def retrieve_chunks_for_tutor(
    query: str,
    resource_id: Optional[str],
    pedagogy_roles: Optional[List[str]] = None,
    k: int = 15,
) -> List[Dict[str, Any]]:
    if not query or not isinstance(query, str) or not query.strip():
        return []
    try:
        return _base_retrieval.retrieve_chunks(
            query=query,
            resource_id=resource_id,
            pedagogy_roles=pedagogy_roles,
            k=k,
        )
    except Exception:
        try:
            logger.exception("tutor_tools_retrieve_chunks_failed")
        except Exception:
            pass
        return []


def rehydrate_chunks(chunk_ids: List[str]) -> List[Dict[str, Any]]:
    if not chunk_ids:
        return []
    try:
        return _base_retrieval.rehydrate_chunks_from_ids(chunk_ids)
    except Exception:
        try:
            logger.exception("tutor_tools_rehydrate_chunks_failed")
        except Exception:
            pass
        return []
