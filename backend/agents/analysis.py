from __future__ import annotations
"""Analysis Agent — strengths/weaknesses with references and summary.

Implements `analysis_agent(payload)` which reads per-concept mastery for a user
and returns:
- user_id
- strengths: list[{concept, mastery, attempts, correct_rate}]
- weaknesses: list[{concept, mastery, attempts, correct_rate}]
- next_concepts: list[str]
- references: list[{concept, refs:[{chunk_id, snippet}]}]
- summary: str

Thresholds configurable via env:
- ANALYSIS_WEAK_MASTERY_LT (default 0.4)
- ANALYSIS_WEAK_RATE_LT (default 0.6)
- ANALYSIS_STRONG_MASTERY_GTE (default 0.7)
- ANALYSIS_STRONG_RATE_GTE (default 0.7)

This agent uses `retrieval.hybrid_search` and `retrieval.diversify_by_page` to find
reference chunks for the weakest concepts, and `llm.call_llm_json` to synthesize a
short narrative summary. Metrics are handled by the shared agent endpoint which
publishes `agent_analysis_calls` and `agent_analysis_elapsed_ms` automatically.
"""
from typing import Dict, Any, List
import os
import logging
import math
from datetime import datetime, timezone
from uuid import UUID

from core.db import get_db_conn
from llm import call_llm_json
from prompts import get as prompt_get, render as prompt_render
from .retrieval import hybrid_search, diversify_by_page


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def analysis_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compute strengths/weaknesses and next steps for a user.

    Expected payload keys:
    - user_id: str (required)
    - top_n: int (optional; cap strengths/weaknesses lists; default 5)
    """
    user_id_raw = (payload or {}).get("user_id")
    if not user_id_raw:
        # mirror orchestrator validation style (raises ValueError upstream as 404)
        raise ValueError("invalid payload, missing keys: ['user_id']")
    try:
        user_uuid = UUID(str(user_id_raw))
        user_id = str(user_uuid)
    except Exception:
        raise ValueError("invalid payload, user_id must be UUID string")

    try:
        top_n = int((payload or {}).get("top_n") or 5)
    except Exception:
        top_n = 5

    weak_m_lt = _env_float("ANALYSIS_WEAK_MASTERY_LT", 0.4)
    weak_r_lt = _env_float("ANALYSIS_WEAK_RATE_LT", 0.6)
    strong_m_ge = _env_float("ANALYSIS_STRONG_MASTERY_GTE", 0.7)
    strong_r_ge = _env_float("ANALYSIS_STRONG_RATE_GTE", 0.7)

    # 1) Fetch mastery rows
    rows: List[Dict[str, Any]] = []
    decay_lambda = _env_float("MASTERY_DECAY_LAMBDA", 0.0)

    def _days_since(ts) -> float:
        if not ts:
            return 0.0
        if not isinstance(ts, datetime):
            return 0.0
        try:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = now - ts
            return max(delta.total_seconds() / 86400.0, 0.0)
        except Exception:
            return 0.0
    resource_id = (payload or {}).get("resource_id")

    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT concept, mastery, last_seen, attempts, correct
                    FROM user_concept_mastery
                    WHERE user_id=%s::uuid
                    """,
                    (user_id,),
                )
                for (concept, mastery, last_seen, attempts, correct) in cur.fetchall() or []:
                    attempts_i = int(attempts or 0)
                    correct_i = int(correct or 0)
                    rate = (correct_i / attempts_i) if attempts_i > 0 else 0.0
                    mastery_val = float(mastery or 0.0)
                    if decay_lambda > 0.0:
                        days = _days_since(last_seen)
                        mastery_val = mastery_val * math.exp(-decay_lambda * days)
                    rows.append(
                        {
                            "concept": str(concept),
                            "mastery": round(mastery_val, 4),
                            "attempts": attempts_i,
                            "correct_rate": round(rate, 4),
                        }
                    )
        finally:
            conn.close()
    except Exception:
        logging.exception("analysis_fetch_mastery_failed")
        # Fall through with empty rows

    if not rows:
        return {
            "user_id": user_id,
            "strengths": [],
            "weaknesses": [],
            "next_concepts": [],
            "references": [],
            "summary": "",
        }

    # 2) Partition strengths and weaknesses
    strengths: List[Dict[str, Any]] = []
    weaknesses: List[Dict[str, Any]] = []
    for r in rows:
        m = float(r.get("mastery") or 0.0)
        cr = float(r.get("correct_rate") or 0.0)
        if m >= strong_m_ge and cr >= strong_r_ge:
            strengths.append(r)
        elif m < weak_m_lt or cr < weak_r_lt:
            weaknesses.append(r)

    # Sort results for deterministic output
    strengths.sort(key=lambda x: (-(x.get("mastery") or 0.0), -(x.get("correct_rate") or 0.0), str(x.get("concept") or "")))
    weaknesses.sort(key=lambda x: ((x.get("mastery") or 0.0), (x.get("correct_rate") or 0.0), str(x.get("concept") or "")))

    strengths = strengths[:top_n]
    weaknesses = weaknesses[:top_n]

    # 3) Next concepts: names of weaknesses
    next_concepts = [w.get("concept") for w in weaknesses if w.get("concept")]

    # 4) References for up to 3 weakest concepts
    references: List[Dict[str, Any]] = []
    for w in weaknesses[:3]:
        cname = str(w.get("concept"))
        try:
            hits = hybrid_search(cname, k=6, resource_id=resource_id)
            hits = diversify_by_page(hits, per_page=1)
            # take 1–2 references
            pick = hits[:2]
            refs = [
                {
                    "chunk_id": h.get("id"),
                    "snippet": (h.get("snippet") or "")[:200],
                }
                for h in pick
                if h.get("id")
            ]
            references.append({"concept": cname, "refs": refs})
        except Exception:
            logging.exception("analysis_references_failed")
            references.append({"concept": cname, "refs": []})

    # 5) Summary via LLM (safe default when mocked)
    def _names(lst: List[Dict[str, Any]]) -> List[str]:
        return [str(x.get("concept")) for x in lst if x.get("concept")]

    strengths_names = _names(strengths)
    weakness_names = _names(weaknesses)

    summary_default = {
        "summary": (
            "Strengths: "
            + (", ".join(strengths_names) if strengths_names else "none")
            + "; Weaknesses: "
            + (", ".join(weakness_names) if weakness_names else "none")
            + ". Next: "
            + (", ".join(next_concepts) if next_concepts else "none")
        )
    }

    # Build summary prompt via registry
    tmpl = prompt_get("analysis.summary")
    prompt = prompt_render(
        tmpl,
        {
            "strengths": ", ".join(strengths_names) if strengths_names else "none",
            "weaknesses": ", ".join(weakness_names) if weakness_names else "none",
        },
    )
    try:
        j = call_llm_json(prompt, summary_default)
        summary_text = str(j.get("summary") or summary_default["summary"]).strip()
    except Exception:
        logging.exception("analysis_summary_llm_failed")
        summary_text = summary_default["summary"]

    return {
        "user_id": user_id,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "next_concepts": next_concepts,
        "references": references,
        "summary": summary_text,
    }
