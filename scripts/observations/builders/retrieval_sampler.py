from __future__ import annotations

from typing import Any, Dict, List, Tuple
import random

from psycopg2.extras import RealDictCursor


def _infer_difficulty_band(value: int | None, thresholds: Dict[str, Any]) -> str:
    low_max = int(thresholds.get("low_max", 2))
    medium_max = int(thresholds.get("medium_max", 3))
    if value is None:
        return "beginner"
    if value <= low_max:
        return "beginner"
    if value <= medium_max:
        return "developing"
    return "proficient"


def _scenario_pedagogy_roles(scenario_type: str) -> List[str]:
    if scenario_type == "explain":
        return ["definition", "explanation"]
    if scenario_type == "worked_example":
        return ["example", "problem"]
    if scenario_type == "concept_check":
        return ["example", "problem"]
    if scenario_type == "reflection":
        return ["summary", "explanation"]
    if scenario_type == "hint":
        return ["problem", "example"]
    return []


def sample_chunks_for_concept(
    conn,
    concept: str,
    difficulty_band: str,
    scenario_type: str,
    domain_config: Dict[str, Any],
    rng: random.Random,
    *,
    min_chunks: int = 2,
    max_chunks: int = 6,
) -> Dict[str, Any]:
    """Sample retrieval chunks for a given concept.

    OBS-01 MVP:
    - Filter by concept membership in chunk.concepts (TEXT[]).
    - Prefer pedagogy_role compatible with the scenario.
    - Prefer difficulty aligned with the concept's difficulty_band.
    """

    resource_ids: List[str] = list(domain_config.get("resource_ids") or [])
    if not resource_ids:
        return {"chunk_ids": [], "chunks": [], "pedagogy_roles": []}

    thresholds: Dict[str, Any] = domain_config.get("difficulty_thresholds") or {}
    desired_roles = _scenario_pedagogy_roles(scenario_type)

    rows: List[Dict[str, Any]] = []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id::text AS id,
                page_number,
                text_snippet,
                difficulty,
                tags
            FROM chunk
            WHERE resource_id = ANY(%s::uuid[])
              AND %s = ANY(concepts)
            """,
            (resource_ids, concept),
        )
        rows = cur.fetchall() or []

    if not rows:
        return {"chunk_ids": [], "chunks": [], "pedagogy_roles": []}

    def _score_row(row: Dict[str, Any]) -> Tuple[int, int]:
        tags = row.get("tags") or {}
        role = None
        if isinstance(tags, dict):
            role = tags.get("pedagogy_role") or tags.get("content_type")
        diff_raw = row.get("difficulty")
        diff_int = None
        if diff_raw is not None:
            try:
                diff_int = int(diff_raw)
            except (TypeError, ValueError):
                pass
        band = _infer_difficulty_band(diff_int, thresholds)

        # role match score
        role_score = 0
        if desired_roles:
            if role in desired_roles:
                role_score = 2
            elif role is not None:
                role_score = 1
        # difficulty match score
        diff_score = 1 if band == difficulty_band else 0
        return role_score, diff_score

    scored = [(row, _score_row(row)) for row in rows]
    # Shuffle for some diversity, then sort by score descending
    rng.shuffle(scored)
    scored.sort(key=lambda x: x[1], reverse=True)

    selected_rows = [row for row, _ in scored[: max_chunks * 2]]

    # Prefer rows with at least some score, fall back to top ones
    positive = [r for r in selected_rows if _score_row(r) != (0, 0)]
    if len(positive) >= min_chunks:
        candidates = positive
    else:
        candidates = selected_rows

    candidates = candidates[:max_chunks]

    chunks: List[Dict[str, Any]] = []
    roles_set = []
    chunk_ids: List[str] = []

    for row in candidates:
        cid = row.get("id")
        if not cid:
            continue
        tags = row.get("tags") or {}
        role = None
        if isinstance(tags, dict):
            role = tags.get("pedagogy_role") or tags.get("content_type")
        snippet = row.get("text_snippet") or ""

        chunk_ids.append(cid)
        chunks.append(
            {
                "id": cid,
                "pedagogy_role": role,
                "snippet": snippet,
                "page_number": row.get("page_number"),
            }
        )
        if role and role not in roles_set:
            roles_set.append(role)

    return {"chunk_ids": chunk_ids, "chunks": chunks, "pedagogy_roles": roles_set}
