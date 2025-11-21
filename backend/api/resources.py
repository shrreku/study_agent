from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from itertools import combinations
from difflib import SequenceMatcher
import re
import os
import io
import uuid
import tempfile
import logging
import json
import psycopg2
from psycopg2.extras import RealDictCursor, Json

from core.auth import require_auth
from core.db import get_db_conn
from core.storage import get_minio_client, upload_bytes_to_object, download_object_to_path, is_gcs_enabled, get_notes_bucket
from kg_pipeline import (
    canonicalize_concept,
    merge_concepts_in_neo4j,
    merge_related_concepts,
    merge_alias,
    merge_prerequisite_edge,
    link_chunk_to_section,
    merge_chunk_figures,
    merge_chunk_formulas,
    merge_chunk_formulas_enhanced,
    merge_chunk_pedagogy_relations,
    merge_section_node,
    merge_next_chunk,
)
from kg_pipeline.enhanced_graph_builder import build_enhanced_educational_kg
from metrics import MetricsCollector
from llm import extract_pedagogy_relations, tag_and_extract, call_json_chat
from llm.common import model_override_context
from ingestion import embed as embed_service
from ingestion.chunker import structural_chunk_resource, enhanced_structural_chunk_resource
from agents.tutor.knowledge import fetch_mastery_map, fetch_prereq_chain
from agents.tutor.policy import level_for_mastery
from agents.tutor.retrieval import retrieve_chunks

router = APIRouter()


class ResourceConceptsRequest(BaseModel):
    resource_ids: List[str]


class ResourceConceptSummary(BaseModel):
    concept: str
    canonical: str
    mastery: Optional[float] = None
    level: Optional[str] = None
    occurrences: int
    resource_ids: List[str]
    pages: List[int]
    pedagogy_roles: Dict[str, int]
    path_index: Optional[int] = None
    hidden: bool = False


class ConceptSummaryRequest(BaseModel):
    concept: str
    resource_ids: List[str]
    max_chunks: Optional[int] = 6


class ConceptSummaryResponse(BaseModel):
    concept: str
    summary: str
    chunk_ids: List[str]


class ResourceConceptFeedbackRequest(BaseModel):
    resource_ids: List[str]
    concept: str
    feedback_type: str = "hide"


@router.post("/api/resources/concepts")
async def list_resource_concepts(
    body: ResourceConceptsRequest,
    user_id: str = Depends(require_auth),
) -> List[ResourceConceptSummary]:
    """Aggregate concept summaries for a set of resources owned by the current user.

    Returns per-concept coverage across resources, pedagogy-role counts,
    and mastery level for the user when available.
    """
    raw_ids = body.resource_ids or []
    resource_ids = [rid.strip() for rid in raw_ids if rid and rid.strip()]
    if not resource_ids:
        return []

    hidden_keys: set[str] = set()

    conn = get_db_conn()
    try:
        # Validate ownership and normalise resource ids
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id::text AS id, user_id::text AS user_id
                FROM resource
                WHERE id = ANY(%s::uuid[])
                """,
                (resource_ids,),
            )
            rows = cur.fetchall() or []

            owned_ids = {r["id"] for r in rows if str(r.get("user_id")) == str(user_id)}

            if not owned_ids:
                raise HTTPException(status_code=404, detail="no_resources_found")

            if len(owned_ids) != len(set(resource_ids)):
                # At least one requested resource is not owned by this user
                raise HTTPException(status_code=403, detail="forbidden_resource_access")

            cur.execute(
                """
                SELECT
                    resource_id::text AS resource_id,
                    c AS concept,
                    COUNT(*) AS occurrences,
                    array_agg(DISTINCT page_number) AS pages,
                    array_remove(array_agg(DISTINCT (tags->>'pedagogy_role')), NULL) AS roles
                FROM (
                    SELECT
                        resource_id,
                        page_number,
                        tags,
                        unnest(concepts) AS c
                    FROM chunk
                    WHERE resource_id = ANY(%s::uuid[])
                      AND concepts IS NOT NULL
                ) t
                GROUP BY resource_id, c
                """,
                (list(owned_ids),),
            )
            concept_rows = cur.fetchall() or []

            cur.execute(
                """
                SELECT concept
                FROM user_resource_concept_feedback
                WHERE user_id = %s::uuid
                  AND resource_id = ANY(%s::uuid[])
                  AND feedback_type = 'hide'
                """,
                (user_id, list(owned_ids)),
            )
            feedback_rows = cur.fetchall() or []

            for fb in feedback_rows:
                raw_fb = (fb.get("concept") or "").strip()
                if not raw_fb:
                    continue
                can_fb, _disp_fb = canonicalize_concept(raw_fb)
                key_fb = (can_fb or raw_fb).strip().lower()
                if key_fb:
                    hidden_keys.add(key_fb)

        # Fetch mastery map using a plain cursor (fetch_mastery_map expects tuple rows)
        with conn.cursor() as cur2:
            mastery_map = fetch_mastery_map(cur2, user_id)
    finally:
        conn.close()

    # Aggregate per canonical concept
    concept_map: Dict[str, Dict[str, Any]] = {}
    for row in concept_rows:
        raw = (row.get("concept") or "").strip()
        if not raw:
            continue
        try:
            occurrences = int(row.get("occurrences") or 0)
        except Exception:
            occurrences = 0
        canonical, display = canonicalize_concept(raw)
        key = (canonical or raw).strip().lower()
        display_name = display or raw

        entry = concept_map.get(key)
        if entry is None:
            entry = {
                "concept": display_name,
                "canonical": canonical or raw,
                "occurrences": 0,
                "resource_ids": set(),
                "pages": set(),
                "pedagogy_roles": {},
            }
            concept_map[key] = entry

        entry["occurrences"] += occurrences
        entry["resource_ids"].add(row.get("resource_id"))

        for p in row.get("pages") or []:
            if p is not None:
                try:
                    entry["pages"].add(int(p))
                except Exception:
                    continue

        roles = row.get("roles") or []
        if roles:
            role_counts: Dict[str, int] = entry["pedagogy_roles"]
            for role in roles:
                if not role:
                    continue
                role_counts[role] = role_counts.get(role, 0) + 1

    if not concept_map:
        return []

    # Compute learning path ordering (optional, best effort).
    # We normalise names so that minor casing or spacing differences between
    # Neo4j and our canonical labels do not break the mapping.
    path_inputs = []
    for entry in concept_map.values():
        cid = entry.get("canonical") or entry.get("concept")
        if cid:
            path_inputs.append(str(cid))
    path_inputs = list(dict.fromkeys(path_inputs))  # de-duplicate while preserving order

    path_index_map: Dict[str, int] = {}
    if path_inputs:
        try:
            ordered = fetch_prereq_chain(path_inputs)
            tmp: Dict[str, int] = {}
            for idx, name in enumerate(ordered or []):
                norm = (str(name) if name is not None else "").strip().lower()
                if norm and norm not in tmp:
                    tmp[norm] = idx
            path_index_map = tmp
        except Exception:
            path_index_map = {}

    # Build response objects with mastery and level
    summaries: List[ResourceConceptSummary] = []
    for entry in concept_map.values():
        canonical_name = entry.get("canonical") or entry.get("concept")
        display_name = entry.get("concept") or canonical_name or ""

        mastery: Optional[float] = None
        level: Optional[str] = None
        if canonical_name:
            info = mastery_map.get(canonical_name) or mastery_map.get(display_name)
            if info is not None:
                try:
                    mastery = float(info.get("mastery")) if info.get("mastery") is not None else None
                except Exception:
                    mastery = None
        if mastery is not None:
            level = level_for_mastery(mastery)

        cid_for_path = str(canonical_name) if canonical_name is not None else display_name
        norm_cid = (cid_for_path or "").strip().lower()
        path_idx = path_index_map.get(norm_cid)

        key_for_hidden = str(canonical_name or display_name).strip().lower() if (canonical_name or display_name) else ""
        is_hidden = key_for_hidden in hidden_keys

        summaries.append(
            ResourceConceptSummary(
                concept=display_name,
                canonical=str(canonical_name or display_name),
                mastery=mastery,
                level=level,
                occurrences=int(entry.get("occurrences") or 0),
                resource_ids=sorted(rid for rid in entry.get("resource_ids") or [] if rid),
                pages=sorted(int(p) for p in entry.get("pages") or []),
                pedagogy_roles=entry.get("pedagogy_roles") or {},
                path_index=path_idx,
                hidden=is_hidden,
            )
        )

    # Sort by path_index (if available), then by ascending mastery, then by name
    def _sort_key(s: ResourceConceptSummary):
        has_path = 0 if s.path_index is not None else 1
        path_val = s.path_index if s.path_index is not None else 0
        mastery_val = s.mastery if s.mastery is not None else 1.0
        return (has_path, path_val, mastery_val, s.concept.lower())

    summaries.sort(key=_sort_key)
    return summaries


@router.post("/api/resources/concepts/feedback")
async def submit_resource_concept_feedback(
    body: ResourceConceptFeedbackRequest,
    user_id: str = Depends(require_auth),
):
    raw_ids = body.resource_ids or []
    resource_ids = [rid.strip() for rid in raw_ids if rid and rid.strip()]
    concept = (body.concept or "").strip()
    feedback_type = (body.feedback_type or "hide").strip() or "hide"

    if not resource_ids:
        raise HTTPException(status_code=400, detail="resource_ids_required")
    if not concept:
        raise HTTPException(status_code=400, detail="concept_required")

    canonical, _display = canonicalize_concept(concept)
    concept_key = (canonical or concept).strip()
    if not concept_key:
        raise HTTPException(status_code=400, detail="concept_required")

    conn = get_db_conn()
    updated = 0
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id::text AS id, user_id::text AS user_id
                FROM resource
                WHERE id = ANY(%s::uuid[])
                """,
                (resource_ids,),
            )
            rows = cur.fetchall() or []

            owned_ids = [r["id"] for r in rows if str(r.get("user_id")) == str(user_id)]
            if not owned_ids:
                raise HTTPException(status_code=404, detail="no_resources_found")

            if len(owned_ids) != len(set(resource_ids)):
                raise HTTPException(status_code=403, detail="forbidden_resource_access")

            for rid in owned_ids:
                cur.execute(
                    """
                    INSERT INTO user_resource_concept_feedback (user_id, resource_id, concept, feedback_type)
                    VALUES (%s::uuid, %s::uuid, %s, %s)
                    ON CONFLICT (user_id, resource_id, concept, feedback_type) DO NOTHING
                    """,
                    (user_id, rid, concept_key, feedback_type),
                )
            conn.commit()
            updated = len(owned_ids)
    finally:
        conn.close()

    return {"ok": True, "updated": updated}


@router.post("/api/resources/concepts/summary")
async def summarize_concept_from_resources(
    body: ConceptSummaryRequest,
    user_id: str = Depends(require_auth),
) -> ConceptSummaryResponse:
    """Summarise a concept using chunks drawn from the user's resources.

    This is a best-effort helper used by the Tutor concepts sidebar for
    expanded concept summaries. It relies on the same retrieval pipeline
    as the tutor agent.
    """

    concept = (body.concept or "").strip()
    if not concept:
        raise HTTPException(status_code=400, detail="concept_required")

    raw_ids = body.resource_ids or []
    resource_ids = [rid.strip() for rid in raw_ids if rid and rid.strip()]
    if not resource_ids:
        raise HTTPException(status_code=400, detail="resource_ids_required")

    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id::text AS id, user_id::text AS user_id
                FROM resource
                WHERE id = ANY(%s::uuid[])
                """,
                (resource_ids,),
            )
            rows = cur.fetchall() or []

            owned_ids = {r["id"] for r in rows if str(r.get("user_id")) == str(user_id)}

            if not owned_ids:
                raise HTTPException(status_code=404, detail="no_resources_found")

            if len(owned_ids) != len(set(resource_ids)):
                # At least one requested resource is not owned by this user
                raise HTTPException(status_code=403, detail="forbidden_resource_access")
    finally:
        conn.close()

    # Retrieve relevant chunks per resource and merge
    max_chunks = body.max_chunks or 6
    pedagogy_roles = ["definition", "example", "application"]

    all_chunks: List[Dict[str, Any]] = []
    for rid in owned_ids:
        try:
            chunks = retrieve_chunks(concept, rid, pedagogy_roles, k=max_chunks)
        except Exception:
            chunks = []
        if chunks:
            all_chunks.extend(chunks)

    # De-duplicate by chunk id and keep highest-scoring first
    dedup: Dict[str, Dict[str, Any]] = {}
    for c in all_chunks:
        cid = str(c.get("id")) if c.get("id") is not None else None
        if not cid:
            continue
        prev = dedup.get(cid)
        if prev is None:
            dedup[cid] = c
            continue
        try:
            prev_score = float(prev.get("score") or 0.0)
        except Exception:
            prev_score = 0.0
        try:
            new_score = float(c.get("score") or 0.0)
        except Exception:
            new_score = 0.0
        if new_score > prev_score:
            dedup[cid] = c

    unique_chunks = list(dedup.values())
    if not unique_chunks:
        return ConceptSummaryResponse(concept=concept, summary="", chunk_ids=[])

    try:
        unique_chunks.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    except Exception:
        pass

    chosen = unique_chunks[:max_chunks]
    chunk_ids: List[str] = [str(c.get("id")) for c in chosen if c.get("id")]

    # Build context for LLM summarisation
    context_lines: List[str] = []
    for idx, c in enumerate(chosen, start=1):
        snippet = (c.get("snippet") or "").strip()
        if not snippet:
            continue
        page = c.get("page_number")
        rid = c.get("resource_id")
        if page is not None:
            label = f"Resource {rid}, page {page}"
        else:
            label = f"Resource {rid}"
        context_lines.append(f"[{idx}] ({label}) {snippet}")

    context_text = "\n\n".join(context_lines) if context_lines else concept

    default_payload: Dict[str, Any] = {"summary": "", "confidence": 0.5}
    user_prompt = (
        f"You are helping a university student revise the concept '{concept}'.\n\n"
        "Using ONLY the context chunks below, write a concise explanation (3-5 sentences) that would help them revise.\n"
        "Focus on the core idea and how it is used in this course. Avoid mentioning chunk IDs.\n\n"
        "Context chunks:\n"
        f"{context_text}\n\n"
        "Return ONLY minified JSON with fields: summary (string), confidence (0.0-1.0)."
    )

    result = call_json_chat(
        user_prompt,
        default=default_payload,
        system_prompt=(
            "You are an educational assistant for engineering students. "
            "Return ONLY minified JSON for the requested concept summary. No markdown."
        ),
        max_tokens=400,
        text_field="summary",
        allow_text_fallback=True,
    )

    summary_text = ""
    try:
        summary_text = str((result or {}).get("summary") or "").strip()
    except Exception:
        summary_text = ""

    return ConceptSummaryResponse(concept=concept, summary=summary_text, chunk_ids=chunk_ids)


def _get_chunker():
    """Choose between enhanced and legacy chunker based on feature flag."""
    if os.getenv("ENHANCED_CHUNKING_ENABLED", "false").lower() in ("true", "1", "yes"):
        return enhanced_structural_chunk_resource
    return structural_chunk_resource


@router.post("/api/resources/upload")
async def upload_resource(file: UploadFile = File(...), title: str = "", token: str = Depends(require_auth)):
    MAX_BYTES = 100 * 1024 * 1024
    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 100MB)")
    try:
        storage_path = upload_bytes_to_object(contents, file.filename, file.content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store object: {e}")

    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO resource (id, title, filename, content_type, size_bytes, storage_path, created_at) VALUES (%s,%s,%s,%s,%s,%s,now()) RETURNING id, title, filename, size_bytes",
                (str(uuid.uuid4()), title or file.filename, file.filename, file.content_type, len(contents), storage_path)
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
        payload = {"resource_id": row["id"], "storage_path": storage_path}

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


class ReindexModels(BaseModel):
    ingest_model: Optional[str] = None
    ingest_tag_model: Optional[str] = None
    pedagogy_model: Optional[str] = None


@router.post("/api/resources/{resource_id}/reindex")
async def reindex_resource(
    resource_id: str,
    models: Optional[ReindexModels] = None,
    token: str = Depends(require_auth),
):
    """Incrementally reindex a resource by diffing structural chunks.

    - Recompute structural chunks
    - Diff against existing by (page_number, source_offset)
    - Insert new, update changed, delete removed
    - Re-run LLM tagging and update concepts
    - Update embeddings for new/changed
    """
    if not resource_id or not resource_id.strip():
        raise HTTPException(status_code=400, detail="resource_id required")

    # Resolve storage path for resource and mark status as chunking
    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT storage_path FROM resource WHERE id=%s::uuid", (resource_id,))
            r = cur.fetchone()
            if not r:
                raise HTTPException(status_code=404, detail="resource not found")
            storage = r["storage_path"]
            logging.info("reindex_start", extra={"resource_id": resource_id, "storage": storage})

            try:
                cur.execute(
                    "UPDATE resource SET status=%s, error_message=NULL WHERE id=%s::uuid",
                    ("chunking", resource_id),
                )
                conn.commit()
            except Exception:
                logging.exception("resource_status_update_failed", extra={"resource_id": resource_id, "phase": "chunking"})
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
            bucket, obj = storage.split("/", 1)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(obj)[1] or "")
            tf.close()
            tmp_download_path = tf.name
            download_object_to_path(storage, tmp_download_path)
            local_path = tmp_download_path
        except Exception as e:
            if tmp_download_path and os.path.exists(tmp_download_path):
                try:
                    os.unlink(tmp_download_path)
                except Exception:
                    pass
            raise HTTPException(status_code=400, detail=f"resource not available locally and storage download failed: {e}")

    # Resolve per-request model overrides with env-based fallbacks
    models = models or ReindexModels()
    ingest_model = models.ingest_model or os.getenv("INGEST_MODEL_HINT") or None
    ingest_tag_model = models.ingest_tag_model or os.getenv("INGEST_TAG_MODEL_HINT") or ingest_model
    pedagogy_model = models.pedagogy_model or os.getenv("PEDAGOGY_MODEL_HINT") or None

    # Compute new structural chunks
    chunker_fn = _get_chunker()
    logging.info("chunker_selected", extra={"fn": getattr(chunker_fn, "__name__", str(chunker_fn))})
    if ingest_model:
        with model_override_context(ingest_model):
            new_chunks = chunker_fn(local_path)
    else:
        new_chunks = chunker_fn(local_path)
    logging.info("chunker_output", extra={"chunks": len(new_chunks)})

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
    logging.info(
        "reindex_diff",
        extra={
            "insert": len(to_insert),
            "update": len(to_update),
            "delete": len(to_delete_ids),
            "unchanged": unchanged,
            "total_new": len(new_chunks),
            "total_existing": len(existing_rows),
        },
    )

    inserted = updated = deleted = 0

    # Before DB writes, mark phase as embedding (chunk-level operations)
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "UPDATE resource SET status=%s WHERE id=%s::uuid",
                    ("embedding", resource_id),
                )
                conn.commit()
            except Exception:
                logging.exception("resource_status_update_failed", extra={"resource_id": resource_id, "phase": "embedding"})
    finally:
        conn.close()

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

    def _tag(text: str, hint: Optional[str] = None) -> Dict[str, Any]:
        try:
            if ingest_tag_model:
                with model_override_context(ingest_tag_model):
                    data = tag_and_extract(text)
            else:
                data = tag_and_extract(text)
            if hint and not data.get("chunk_type"):
                data["chunk_type"] = hint
            return data
        except Exception:
            return {"chunk_type": hint or None, "concepts": [], "math_expressions": []}

    def _is_alias_candidate(primary: str, candidate: str) -> bool:
        primary = (primary or "").strip()
        candidate = (candidate or "").strip()
        if not primary or not candidate:
            return False
        if primary.lower() == candidate.lower():
            return False
        primary_norm = re.sub(r"[^a-z0-9]", "", primary.lower())
        candidate_norm = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if primary_norm and primary_norm == candidate_norm:
            return True
        ratio = SequenceMatcher(None, primary.lower(), candidate.lower()).ratio()
        return ratio >= 0.78

    def _update_kg_relations(
        concepts: List[str],
        chunk_id: str,
        snippet: str,
        resource_id: str,
        chunk_meta: Dict[str, Any],
    ) -> tuple[List[str], List[str]]:
        if not concepts:
            return [], []

        unique: List[str] = []
        seen = set()
        for c in concepts:
            c_clean = (c or "").strip()
            if not c_clean:
                continue
            key = c_clean.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(c_clean)

        if not unique:
            return [], []

        canonical_unique: List[str] = []
        canonical_seen: set[str] = set()
        for label in unique:
            canonical, _display = canonicalize_concept(label)
            if canonical and canonical not in canonical_seen:
                canonical_seen.add(canonical)
                canonical_unique.append(canonical)

        merge_concepts_in_neo4j(unique, chunk_id, snippet, resource_id, chunk_meta)

        alias_merges = 0
        alias_suppressed = 0
        processed_alias_pairs: set[tuple[str, str, str]] = set()

        def _record_alias(alias: str, target: str, method: str) -> None:
            nonlocal alias_merges, alias_suppressed
            alias_norm = (alias or "").strip()
            target_norm = (target or "").strip()
            if not alias_norm or not target_norm:
                return
            if alias_norm.lower() == target_norm.lower():
                return
            key = (alias_norm.lower(), target_norm.lower(), method)
            if key in processed_alias_pairs:
                alias_suppressed += 1
                return
            processed_alias_pairs.add(key)
            try:
                merge_alias(
                    alias_norm,
                    target_norm,
                    method=method,
                    evidence_chunk_id=chunk_id,
                )
                alias_merges += 1
            except Exception:
                logging.exception("kg_alias_merge_failed", extra={"alias": alias_norm, "target": target_norm, "method": method})

        if len(unique) >= 2:
            pairs = []
            for a, b in combinations(unique, 2):
                if a.lower() == b.lower():
                    continue
                pairs.append((a, b, 1.0))
            if pairs:
                merge_related_concepts(
                    pairs,
                    method="chunk_cooccurrence",
                    evidence_chunk_id=chunk_id,
                )
                try:
                    mc = MetricsCollector.get_global()
                    mc.increment("kg_related_pairs", len(pairs))
                except Exception:
                    pass

            for a, b, _ in pairs:
                if _is_alias_candidate(a, b):
                    # choose the canonical target via normalization; default to length heuristic
                    can_a, _ = canonicalize_concept(a)
                    can_b, _ = canonicalize_concept(b)
                    if can_a == can_b:
                        target = a if len(a) <= len(b) else b
                        alias = b if target == a else a
                    else:
                        target = a
                        alias = b
                    _record_alias(alias, target, method="heuristic_alias")

        chunk_type = (chunk_meta or {}).get("chunk_type") or ""
        if chunk_type in {"definition", "theorem", "procedure"} and len(unique) >= 2:
            target = unique[0]
            for prereq in unique[1:]:
                merge_prerequisite_edge(
                    prereq,
                    target,
                    confidence=0.6,
                    evidence_chunk_id=chunk_id,
                    method=f"{chunk_type}_context",
                )

            if chunk_type == "definition":
                for alias in unique[1:]:
                    if _is_alias_candidate(target, alias):
                        _record_alias(alias, target, method="definition_alias")

        if alias_merges or alias_suppressed:
            try:
                mc = MetricsCollector.get_global()
                if alias_merges:
                    mc.increment("kg_alias_merges", alias_merges)
                if alias_suppressed:
                    mc.increment("kg_alias_suppressed", alias_suppressed)
            except Exception:
                pass
            logging.debug(
                "kg_alias_summary",
                extra={
                    "chunk_id": chunk_id,
                    "alias_merges": alias_merges,
                    "alias_suppressed": alias_suppressed,
                    "concepts": unique,
                },
            )
        pedagogy_payload = {}
        enable_pedagogy = os.getenv("PEDAGOGY_LLM_ENABLE", "0").lower() in {"1", "true", "yes"}
        if enable_pedagogy:
            try:
                pedagogy_payload = extract_pedagogy_relations(
                    chunk_meta.get("full_text") or snippet,
                    {
                        "chunk_type": chunk_type,
                        "title": chunk_meta.get("section_title"),
                        "resource_id": resource_id,
                    },
                    model_hint=pedagogy_model,
                )
            except Exception:
                logging.exception("pedagogy_llm_failed", extra={"chunk_id": chunk_id})

        pedagogy_result = {}
        if enable_pedagogy and pedagogy_payload:
            try:
                pedagogy_result = merge_chunk_pedagogy_relations(
                    chunk_id,
                    resource_id,
                    pedagogy_payload,
                    chunk_type=chunk_type,
                    method="llm_pedagogy",
                )
            except Exception:
                logging.exception("pedagogy_merge_failed", extra={"chunk_id": chunk_id})

        if enable_pedagogy:
            try:
                mc = MetricsCollector.get_global()
                mc.increment("pedagogy_llm_requests")
                if pedagogy_payload:
                    mc.increment("pedagogy_llm_payload_nonempty")
                    for key in ("defines", "explains", "exemplifies", "proves", "derives", "figure_links", "prereqs", "evidence"):
                        items = pedagogy_payload.get(key) or []
                        if items:
                            mc.increment(f"pedagogy_llm_{key}_count", len(items))
                    merged = (pedagogy_result or {}).get("concept_canonicals") or []
                    if merged:
                        mc.increment("pedagogy_llm_concepts_merged", len(merged))
            except Exception:
                pass

        return unique, canonical_unique

    def _infer_prereqs_from_sequence(resource_id: str, summaries: List[Dict[str, Any]]) -> None:
        if not summaries:
            return
        summaries_sorted = sorted(
            summaries,
            key=lambda s: (
                s.get("page_number") if s.get("page_number") is not None else 0,
                s.get("source_offset") if s.get("source_offset") is not None else 0,
            ),
        )
        prev_primary: Optional[str] = None
        prev_chunk: Optional[str] = None
        for summary in summaries_sorted:
            concepts_unique = summary.get("concepts_unique") or []
            if not concepts_unique:
                if prev_chunk and summary.get("chunk_id"):
                    merge_next_chunk(prev_chunk, summary.get("chunk_id"), resource_id)
                    prev_chunk = summary.get("chunk_id")
                continue
            primary = concepts_unique[0]
            if prev_primary and primary and primary.lower() != prev_primary.lower():
                merge_prerequisite_edge(
                    prev_primary,
                    primary,
                    confidence=0.4,
                    evidence_chunk_id=summary.get("chunk_id"),
                    method="chunk_order",
                )
            prev_primary = primary
            if prev_chunk and summary.get("chunk_id"):
                merge_next_chunk(prev_chunk, summary.get("chunk_id"), resource_id)
            prev_chunk = summary.get("chunk_id")
        # ensure final chunk still updates chain if gaps existed
        if prev_chunk and summaries_sorted:
            merge_next_chunk(prev_chunk, None, resource_id)

    # Inserts
    if to_insert:
        texts = [c.get("full_text") or "" for c in to_insert]
        tags_list = [_tag(t, c.get("chunk_type_hint")) for c, t in zip(to_insert, texts)]
        vecs = embed_service.embed_texts(texts)
        embed_version = os.getenv("EMBED_VERSION", "all-MiniLM-L6-v2-2025-09")
        conn = get_db_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sequence_summaries: List[Dict[str, Any]] = []
                for c, tags, vec in zip(to_insert, tags_list, vecs):
                    section_title = c.get("section_title") or ""
                    section_number = c.get("section_number") or ""
                    section_path = c.get("section_path") or []
                    section_level = c.get("section_level")
                    page_start = c.get("page_start") or c.get("page_number")
                    page_end = c.get("page_end") or c.get("page_number")
                    token_count = c.get("token_count") or len((c.get("full_text") or "").split())
                    has_figure = bool(c.get("has_figure"))
                    has_equation = bool(c.get("has_equation"))
                    figure_labels = c.get("figure_labels") or []
                    equation_labels = c.get("equation_labels") or []
                    caption = c.get("caption")
                    
                    # Build tags JSONB with pedagogy_role
                    tags_json = {}
                    if c.get("pedagogy_role"):
                        tags_json["pedagogy_role"] = c.get("pedagogy_role")
                    if c.get("content_type"):
                        tags_json["content_type"] = c.get("content_type")
                    if c.get("difficulty"):
                        tags_json["difficulty"] = c.get("difficulty")
                    if c.get("cognitive_level"):
                        tags_json["cognitive_level"] = c.get("cognitive_level")
                    for _fld in ("domain", "topic", "subtopic", "key_concepts", "prerequisites", "learning_objectives"):
                        _val = c.get(_fld)
                        if _val is not None and _val != "":
                            tags_json[_fld] = _val
                    if isinstance(c.get("tags"), dict):
                        tags_json.update(c.get("tags"))

                    tags_db_val = Json(tags_json)
                    
                    heading_text = " ".join(filter(None, [section_number, section_title]))
                    # tags_text for search - use old tags list if it exists
                    old_tags_list = c.get("tags") if isinstance(c.get("tags"), list) else []
                    tags_text = " ".join(old_tags_list)
                    full_text = c.get("full_text") or ""
                    text_snippet = c.get("text_snippet") or full_text[:300]
                    chunk_type = tags.get("chunk_type") or c.get("chunk_type_hint")
                    chunk_meta = {
                        "full_text": full_text,
                        "chunk_type": chunk_type,
                        "section_path": section_path,
                        "section_title": section_title,
                        "section_number": section_number,
                        "section_level": section_level,
                        "page_number": c.get("page_number"),
                    }
                    cur.execute(
                        """
                        INSERT INTO chunk (
                            id, resource_id, page_number, source_offset, full_text,
                            chunk_type, concepts, math_expressions, embedding, embedding_version,
                            created_at, updated_at,
                            section_title, section_number, section_path, section_level,
                            page_start, page_end, token_count, has_figure, has_equation,
                            figure_labels, equation_labels, caption, tags,
                            text_snippet,
                            heading_tsv, body_tsv, search_tsv
                        )
                        VALUES (
                            uuid_generate_v4(), %s::uuid, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            now(), now(),
                            %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s,
                            to_tsvector('english', coalesce(%s, '')),
                            to_tsvector('english', %s),
                            setweight(to_tsvector('english', coalesce(%s, '')), 'A')
                                || setweight(to_tsvector('english', %s), 'B')
                                || setweight(to_tsvector('english', coalesce(%s, '')), 'C')
                        )
                        RETURNING id::text
                        """,
                        (
                            resource_id,
                            c.get("page_number"),
                            c.get("source_offset"),
                            full_text,
                            chunk_type,
                            tags.get("concepts"),
                            tags.get("math_expressions"),
                            vec,
                            embed_version,
                            section_title,
                            section_number,
                            section_path,
                            section_level,
                            page_start,
                            page_end,
                            token_count,
                            has_figure,
                            has_equation,
                            figure_labels,
                            equation_labels,
                            caption,
                            tags_db_val,
                            text_snippet,
                            heading_text,
                            full_text,
                            heading_text,
                            full_text,
                            tags_text,
                        ),
                    )
                    new_id = cur.fetchone()["id"]
                    try:
                        concepts = tags.get("concepts") or []
                        concepts_unique, concepts_canonical = _update_kg_relations(
                            concepts,
                            str(new_id),
                            text_snippet,
                            resource_id,
                            chunk_meta,
                        )
                        if concepts_unique:
                            sequence_summaries.append(
                                {
                                    "chunk_id": str(new_id),
                                    "concepts_unique": concepts_unique,
                                    "page_number": c.get("page_number"),
                                    "source_offset": c.get("source_offset"),
                                }
                            )
                        link_chunk_to_section(
                            str(new_id),
                            resource_id,
                            section_path,
                            section_title,
                            section_number,
                            section_level,
                        )
                        merge_chunk_figures(
                            str(new_id),
                            resource_id,
                            figure_labels,
                            concept_canonicals=concepts_canonical,
                        )
                        # Use INGEST-04 enhanced formulas if available, otherwise fall back to old tags
                        if c.get('formulas'):
                            merge_chunk_formulas_enhanced(
                                str(new_id),
                                resource_id,
                                c.get('formulas'),
                                concept_canonicals=concepts_canonical,
                            )
                        else:
                            merge_chunk_formulas(
                                str(new_id),
                                resource_id,
                                tags.get("math_expressions"),
                                concept_canonicals=concepts_canonical,
                            )
                        # Optionally build enhanced educational KG (LLM-based) for this chunk
                        try:
                            if os.getenv("KG_ENHANCED_EXTRACTION_ENABLED", "false").lower() in ("true", "1", "yes"):
                                build_enhanced_educational_kg(
                                    full_text,
                                    str(new_id),
                                    resource_id,
                                    {
                                        "title": section_title,
                                        "section_title": section_title,
                                        "chunk_type": chunk_type,
                                    },
                                )
                        except Exception:
                            logging.exception("enhanced_kg_build_failed", extra={"chunk_id": str(new_id)})
                    except Exception:
                        logging.exception("kg_merge_failed")
            conn.commit()
            inserted = len(to_insert)
        finally:
            conn.close()
        _infer_prereqs_from_sequence(resource_id, sequence_summaries)

    # Updates
    if to_update:
        texts_upd = [c.get("full_text") or "" for (_id, c) in to_update]
        tags_upd = [_tag(t, c.get("chunk_type_hint")) for ( _id, c), t in zip(to_update, texts_upd)]
        vecs_upd = embed_service.embed_texts(texts_upd)
        embed_version = os.getenv("EMBED_VERSION", "all-MiniLM-L6-v2-2025-09")
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                sequence_summaries_upd: List[Dict[str, Any]] = []
                for (chunk_id, c), tags, vec in zip(to_update, tags_upd, vecs_upd):
                    section_title = c.get("section_title") or ""
                    section_number = c.get("section_number") or ""
                    section_path = c.get("section_path") or []
                    section_level = c.get("section_level")
                    page_start = c.get("page_start") or c.get("page_number")
                    page_end = c.get("page_end") or c.get("page_number")
                    token_count = c.get("token_count") or len((c.get("full_text") or "").split())
                    has_figure = bool(c.get("has_figure"))
                    has_equation = bool(c.get("has_equation"))
                    figure_labels = c.get("figure_labels") or []
                    equation_labels = c.get("equation_labels") or []
                    caption = c.get("caption")
                    
                    # Build tags JSONB with pedagogy_role
                    tags_json = {}
                    if c.get("pedagogy_role"):
                        tags_json["pedagogy_role"] = c.get("pedagogy_role")
                    if c.get("content_type"):
                        tags_json["content_type"] = c.get("content_type")
                    if c.get("difficulty"):
                        tags_json["difficulty"] = c.get("difficulty")
                    if c.get("cognitive_level"):
                        tags_json["cognitive_level"] = c.get("cognitive_level")
                    for _fld in ("domain", "topic", "subtopic", "key_concepts", "prerequisites", "learning_objectives"):
                        _val = c.get(_fld)
                        if _val is not None and _val != "":
                            tags_json[_fld] = _val
                    # Merge any additional tags from chunk
                    if isinstance(c.get("tags"), dict):
                        tags_json.update(c.get("tags"))
                    
                    heading_text = " ".join(filter(None, [section_number, section_title]))
                    # tags_text for search - use old tags list if it exists
                    old_tags_list = c.get("tags") if isinstance(c.get("tags"), list) else []
                    tags_text = " ".join(old_tags_list)
                    full_text = c.get("full_text") or ""
                    text_snippet = c.get("text_snippet") or full_text[:300]
                    chunk_type = tags.get("chunk_type") or c.get("chunk_type_hint")
                    chunk_meta = {
                        "full_text": full_text,
                        "chunk_type": chunk_type,
                        "section_path": section_path,
                        "section_title": section_title,
                        "section_number": section_number,
                        "section_level": section_level,
                        "page_number": c.get("page_number"),
                    }
                    cur.execute(
                        """
                        UPDATE chunk
                        SET full_text=%s, chunk_type=%s, concepts=%s, math_expressions=%s,
                            embedding=%s, embedding_version=%s, updated_at=now(),
                            section_title=%s, section_number=%s, section_path=%s, section_level=%s,
                            page_start=%s, page_end=%s, token_count=%s,
                            has_figure=%s, has_equation=%s, figure_labels=%s, equation_labels=%s,
                            caption=%s, tags=%s, text_snippet=%s,
                            heading_tsv=to_tsvector('english', coalesce(%s, '')),
                            body_tsv=to_tsvector('english', %s),
                            search_tsv=
                                setweight(to_tsvector('english', coalesce(%s, '')), 'A')
                                || setweight(to_tsvector('english', %s), 'B')
                                || setweight(to_tsvector('english', coalesce(%s, '')), 'C')
                        WHERE id=%s::uuid
                        """,
                        (
                            full_text,
                            chunk_type,
                            tags.get("concepts"),
                            tags.get("math_expressions"),
                            vec,
                            embed_version,
                            section_title,
                            section_number,
                            section_path,
                            section_level,
                            page_start,
                            page_end,
                            token_count,
                            has_figure,
                            has_equation,
                            figure_labels,
                            equation_labels,
                            caption,
                            tags_json,
                            text_snippet,
                            heading_text,
                            full_text,
                            heading_text,
                            full_text,
                            tags_text,
                            chunk_id,
                        ),
                    )
                    try:
                        concepts_canonical = None
                        concepts = tags.get("concepts") or []
                        concepts_unique, concepts_canonical = _update_kg_relations(
                            concepts,
                            str(chunk_id),
                            text_snippet,
                            resource_id,
                            chunk_meta,
                        )
                        if concepts_unique:
                            sequence_summaries_upd.append(
                                {
                                    "chunk_id": str(chunk_id),
                                    "concepts_unique": concepts_unique,
                                    "page_number": c.get("page_number"),
                                    "source_offset": c.get("source_offset"),
                                }
                            )
                        link_chunk_to_section(
                            str(chunk_id),
                            resource_id,
                            section_path,
                            section_title,
                            section_number,
                            section_level,
                        )
                        merge_chunk_figures(
                            str(chunk_id),
                            resource_id,
                            figure_labels,
                            concept_canonicals=concepts_canonical,
                        )
                        # Use INGEST-04 enhanced formulas if available, otherwise fall back to old tags
                        if c.get('formulas'):
                            merge_chunk_formulas_enhanced(
                                str(chunk_id),
                                resource_id,
                                c.get('formulas'),
                                concept_canonicals=concepts_canonical,
                            )
                        else:
                            merge_chunk_formulas(
                                str(chunk_id),
                                resource_id,
                                tags.get("math_expressions"),
                                concept_canonicals=concepts_canonical,
                            )
                        # Optionally build enhanced educational KG (LLM-based) for this updated chunk
                        try:
                            if os.getenv("KG_ENHANCED_EXTRACTION_ENABLED", "false").lower() in ("true", "1", "yes"):
                                build_enhanced_educational_kg(
                                    full_text,
                                    str(chunk_id),
                                    resource_id,
                                    {
                                        "title": section_title,
                                        "section_title": section_title,
                                        "chunk_type": chunk_type,
                                    },
                                )
                        except Exception:
                            logging.exception("enhanced_kg_build_failed", extra={"chunk_id": str(chunk_id)})
                    except Exception:
                        logging.exception("kg_merge_failed")
            conn.commit()
            updated = len(to_update)
        finally:
            conn.close()
        _infer_prereqs_from_sequence(resource_id, sequence_summaries_upd)

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
            bucket, obj = storage.split("/", 1)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(obj)[1] or "")
            tf.close()
            tmp_download_path = tf.name
            download_object_to_path(storage, tmp_download_path)
            local_path = tmp_download_path
        except Exception as e:
            if tmp_download_path and os.path.exists(tmp_download_path):
                try:
                    os.unlink(tmp_download_path)
                except Exception:
                    pass
            raise HTTPException(status_code=400, detail=f"resource not available locally and storage download failed: {e}")

    chunker_fn = _get_chunker()
    chunks = chunker_fn(local_path)

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
                
                # Build tags JSONB with pedagogy_role
                tags_json = {}
                if c.get("pedagogy_role"):
                    tags_json["pedagogy_role"] = c.get("pedagogy_role")
                if c.get("content_type"):
                    tags_json["content_type"] = c.get("content_type")
                if c.get("difficulty"):
                    tags_json["difficulty"] = c.get("difficulty")
                if c.get("cognitive_level"):
                    tags_json["cognitive_level"] = c.get("cognitive_level")
                for _fld in ("domain", "topic", "subtopic", "key_concepts", "prerequisites", "learning_objectives"):
                    _val = c.get(_fld)
                    if _val is not None and _val != "":
                        tags_json[_fld] = _val

                cur.execute(
                    "INSERT INTO chunk (id, resource_id, page_number, source_offset, full_text, chunk_type, concepts, math_expressions, tags, created_at) VALUES (uuid_generate_v4(),%s,%s,%s,%s,%s,%s,%s,%s,now()) RETURNING id",
                    (resource_id, c["page_number"], c["source_offset"], c["full_text"], chunk_type, concepts, math_expressions, json.dumps(tags_json))
                )
                new_id = cur.fetchone()["id"]
                try:
                    concepts_canonical = None
                    # Use INGEST-04 enhanced formulas if available, otherwise fall back to old tags
                    if c.get('formulas'):
                        merge_chunk_formulas_enhanced(
                            str(new_id),
                            resource_id,
                            c.get('formulas'),
                            concept_canonicals=concepts_canonical,
                        )
                    else:
                        merge_chunk_formulas(
                            str(new_id),
                            resource_id,
                            tags.get("math_expressions"),
                            concept_canonicals=concepts_canonical,
                        )
                except Exception:
                    logging.exception("kg_merge_failed")
                inserted += 1
        conn.commit()
    finally:
        conn.close()

    try:
        summaries = []
        for c in chunks:
            if "_kg_unique" in c:
                summaries.append(
                    {
                        "chunk_id": c.get("_kg_chunk_id"),
                        "concepts_unique": c.get("_kg_unique"),
                        "page_number": c.get("page_number"),
                        "source_offset": c.get("source_offset"),
                    }
                )
        if "_infer_prereqs_from_sequence" in globals():
            _infer_prereqs_from_sequence(resource_id, summaries)
    except Exception:
        logging.exception("kg_sequence_prereq_failed")

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
