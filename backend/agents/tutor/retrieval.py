from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..retrieval import hybrid_search, filter_relevant, diversify_by_page, _score_with_pedagogy
from .constants import logger


def retrieve_chunks(
    query: str,
    resource_id: Optional[str],
    pedagogy_roles: Optional[List[str]],
    k: int = 4,
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []

    try:
        results = hybrid_search(query, k=max(5, k), resource_id=resource_id)
        results = filter_relevant(results)
        results = _score_with_pedagogy(results, pedagogy_roles)
        results = diversify_by_page(results, per_page=1)
        return results[:k]
    except Exception:
        logger.exception("tutor_retrieval_failed")
        return []
