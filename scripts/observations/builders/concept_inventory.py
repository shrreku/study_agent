from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from psycopg2.extras import RealDictCursor


def _infer_difficulty_band(avg_difficulty: float | None, thresholds: Dict[str, Any]) -> str:
    low_max = int(thresholds.get("low_max", 2))
    medium_max = int(thresholds.get("medium_max", 3))
    if avg_difficulty is None:
        return "beginner"
    if avg_difficulty <= low_max:
        return "beginner"
    if avg_difficulty <= medium_max:
        return "developing"
    return "proficient"


def build_concept_inventory(conn, domain_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build per-domain concept inventory from the chunk table.

    OBS-01 MVP:
    - Use chunk.concepts (TEXT[]) as the primary concept source.
    - Aggregate per concept across the configured resource_ids.
    - Derive a simple difficulty band from chunk.difficulty / tags["difficulty"].
    - Optionally filter concepts by include/exclude prefixes.
    """

    resource_ids: List[str] = list(domain_config.get("resource_ids") or [])
    if not resource_ids:
        return []

    thresholds: Dict[str, Any] = domain_config.get("difficulty_thresholds") or {}

    concept_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "difficulty_values": []})

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        for rid in resource_ids:
            rid_str = str(rid).strip()
            if not rid_str:
                continue
            cur.execute(
                """
                SELECT concepts, difficulty, tags
                FROM chunk
                WHERE resource_id = %s::uuid
                """,
                (rid_str,),
            )
            rows = cur.fetchall() or []
            for row in rows:
                concepts = row.get("concepts") or []
                if not isinstance(concepts, list):
                    continue
                difficulty_value = None
                raw_diff = row.get("difficulty")
                if raw_diff is not None:
                    try:
                        difficulty_value = int(raw_diff)
                    except (TypeError, ValueError):
                        pass
                tags = row.get("tags") or {}
                if isinstance(tags, dict) and difficulty_value is None:
                    tag_diff = tags.get("difficulty")
                    if tag_diff is not None:
                        try:
                            difficulty_value = int(tag_diff)
                        except (TypeError, ValueError):
                            pass

                for concept in concepts:
                    name = (concept or "").strip()
                    if not name:
                        continue
                    stat = concept_stats[name]
                    stat["count"] += 1
                    if difficulty_value is not None:
                        stat["difficulty_values"].append(difficulty_value)

    include_prefixes = [p for p in (domain_config.get("concept_include_prefixes") or []) if p]
    exclude_prefixes = [p for p in (domain_config.get("concept_exclude_prefixes") or []) if p]

    def _keep(name: str) -> bool:
        if include_prefixes and not any(name.startswith(p) for p in include_prefixes):
            return False
        if any(name.startswith(p) for p in exclude_prefixes):
            return False
        return True

    inventory: List[Dict[str, Any]] = []
    for concept, stat in concept_stats.items():
        if not _keep(concept):
            continue
        vals = stat["difficulty_values"]
        avg_diff = float(sum(vals) / len(vals)) if vals else None
        band = _infer_difficulty_band(avg_diff, thresholds)
        inventory.append(
            {
                "concept": concept,
                "learning_path": [concept],
                "difficulty_band": band,
            }
        )

    inventory.sort(key=lambda x: x["concept"].lower())
    return inventory
