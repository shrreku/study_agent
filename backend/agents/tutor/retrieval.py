from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os

from ..retrieval import hybrid_search, filter_relevant, _score_with_pedagogy
from .constants import logger


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _expand_with_neighbors(chunks: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Expand top-scoring chunks with nearby neighbors on the same page.

    This keeps retrieval simple and efficient while giving the LLM more
    contiguous context from the source materials.

    Strategy (all values configurable via env):
      - Start from ranked chunks (already scored + pedagogy-boosted).
      - For each high-scoring chunk, also include up to N neighbors before
        and after it on the same (resource_id, page_number), ordered by
        source_offset.
      - Cap total chunks and per-page chunks to avoid huge contexts.
    """

    if not chunks:
        return []

    # Defaults are modest to avoid huge prompts while still improving grounding
    max_total_default = max(k * 2, 8)
    max_total = _env_int("TUTOR_RETRIEVAL_MAX_CHUNKS", max_total_default)
    window = _env_int("TUTOR_RETRIEVAL_NEIGHBOR_WINDOW", 1)
    max_per_page = _env_int("TUTOR_RETRIEVAL_MAX_PER_PAGE", 4)

    if window <= 0 or max_total <= 0:
        return chunks[:max_total]

    # Group by (resource_id, page_number) and sort within each group by source_offset
    from collections import defaultdict

    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        key = (str(c.get("resource_id")), int(c.get("page_number") or 0))
        groups[key].append(c)

    index_map: Dict[Tuple[str, int, str], int] = {}
    for key, items in groups.items():
        # sort: chunks with known source_offset first, then by offset
        items.sort(key=lambda x: (x.get("source_offset") is None, int(x.get("source_offset") or 0)))
        for idx, c in enumerate(items):
            cid = c.get("id")
            if cid:
                index_map[(key[0], key[1], str(cid))] = idx

    selected: List[Dict[str, Any]] = []
    seen_ids = set()
    per_page_counts: Dict[Tuple[str, int], int] = {}

    for center in chunks:
        if len(selected) >= max_total:
            break
        cid = center.get("id")
        key = (str(center.get("resource_id")), int(center.get("page_number") or 0))
        if not cid or key not in groups:
            continue

        base_count = per_page_counts.get(key, 0)
        if base_count >= max_per_page:
            continue

        idx = index_map.get((key[0], key[1], str(cid)))
        if idx is None:
            continue

        items = groups[key]
        start = max(0, idx - window)
        end = min(len(items), idx + window + 1)

        for j in range(start, end):
            if len(selected) >= max_total:
                break
            if per_page_counts.get(key, 0) >= max_per_page:
                break
            neighbor = items[j]
            nid = neighbor.get("id")
            if not nid or nid in seen_ids:
                continue
            selected.append(neighbor)
            seen_ids.add(nid)
            per_page_counts[key] = per_page_counts.get(key, 0) + 1

    # Fallback: if for some reason expansion selected nothing, return top-k
    if not selected:
        return chunks[:max_total]

    return selected


def retrieve_chunks(
    query: str,
    resource_id: Optional[str],
    pedagogy_roles: Optional[List[str]],
    k: int = 15,
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []

    try:
        # Fetch a larger candidate pool so we can expand with neighbors.
        base_k = max(10, k * 4)
        results = hybrid_search(query, k=base_k, resource_id=resource_id)
        filtered = filter_relevant(results)
        if not filtered and os.getenv("TUTOR_RETRIEVAL_RELAX_IF_EMPTY", "true").strip().lower() == "true":
            filtered = filter_relevant(results, min_score=0.0, min_sim=0.0, min_bm25=0.0)
        scored = _score_with_pedagogy(filtered, pedagogy_roles)
        expanded = _expand_with_neighbors(scored, k)
        return expanded
    except Exception:
        logger.exception("tutor_retrieval_failed")
        return []
