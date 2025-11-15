"""Pedagogy extraction helper using the shared JSON chat client."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .common import call_json_chat
from kg_pipeline.utils import canonicalize_concept

PEDAGOGY_SCHEMA_EMPTY: Dict[str, List[Any]] = {
    "defines": [],
    "explains": [],
    "exemplifies": [],
    "formulas": [],
    "prerequisites": [],
}


def _default_payload() -> Dict[str, List[Any]]:
    return {k: [] for k in PEDAGOGY_SCHEMA_EMPTY}


def _serialize_context(meta: Dict[str, Any]) -> str:
    title = meta.get("title") or meta.get("section_title") or ""
    chunk_type = meta.get("chunk_type") or ""
    section_path = meta.get("section_path") or []
    path_str = " > ".join(str(p) for p in section_path if p)
    context = {
        "title": title,
        "chunk_type": chunk_type,
        "section_path": path_str,
    }
    return json.dumps(context, ensure_ascii=False)


def _build_prompt(text: str, meta: Dict[str, Any]) -> str:
    """Build a clear, focused prompt for educational content extraction."""
    title = meta.get("title") or meta.get("section_title") or ""
    chunk_type = meta.get("chunk_type") or ""
    
    prompt = f"""Extract educational knowledge from this {chunk_type or 'content'} text.

Identify:
1. DEFINES: Terms and concepts that are formally defined or introduced
2. EXPLAINS: Concepts that are explained or elaborated on
3. EXEMPLIFIES: Concepts illustrated with examples
4. FORMULAS: Mathematical equations with description of what they represent
5. PREREQUISITES: Concepts that must be understood first (as prerequisite -> target concept pairs)

Return JSON with these exact keys:
- defines: list of term names (strings)
- explains: list of concept names (strings)
- exemplifies: list of concept names (strings)  
- formulas: list of objects with "equation" and "about" (concept it relates to)
- prerequisites: list of objects with "from" (prerequisite) and "to" (target concept)

Only include concepts you're confident about. Use specific technical terms from the text.

{f'Title: {title}' if title else ''}

Text:
{text}

Return valid JSON only."""
    return prompt


def _confidence_threshold() -> float:
    try:
        return float(os.getenv("PEDAGOGY_LLM_MIN_CONF", "0.5"))
    except Exception:
        return 0.5


def _canon(term: Optional[str]) -> Tuple[str, str]:
    return canonicalize_concept(term or "")


def _normalize_list(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    normalized: List[str] = []
    for item in items or []:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(cleaned)
    return normalized


def _normalize_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and canonicalize the LLM output into our internal format."""
    result = _default_payload()
    if not isinstance(raw, dict):
        return result

    min_conf = _confidence_threshold()

    # Process DEFINES - terms that are defined in the text
    defines: List[Dict[str, Any]] = []
    for item in raw.get("defines", []) or []:
        # Handle both string and object formats
        if isinstance(item, str):
            term = item
            aliases = []
        elif isinstance(item, dict):
            term = item.get("term")
            aliases = _normalize_list(item.get("aliases", []) or [])
        else:
            continue
            
        canonical, display = _canon(term)
        if canonical:
            defines.append({"term": display, "canonical": canonical, "aliases": aliases})
    result["defines"] = defines

    # Process EXPLAINS and EXEMPLIFIES - simple concept lists
    for key in ("explains", "exemplifies"):
        normalized: List[Dict[str, Any]] = []
        for item in raw.get(key, []) or []:
            # Handle both string and object formats
            concept = item if isinstance(item, str) else (item.get("term") if isinstance(item, dict) else None)
            if not concept:
                continue
            canonical, display = _canon(concept)
            if canonical:
                normalized.append({"term": display, "canonical": canonical})
        result[key] = normalized

    # Process FORMULAS - equations with their associated concepts
    formulas: List[Dict[str, Any]] = []
    for entry in raw.get("formulas", []) or []:
        if not isinstance(entry, dict):
            continue
        equation = str(entry.get("equation") or "").strip()
        about = entry.get("about")
        if not equation:
            continue
        canonical, display = _canon(about) if about else (None, None)
        formula_entry = {"equation": equation}
        if canonical:
            formula_entry["about"] = display
            formula_entry["canonical"] = canonical
        formulas.append(formula_entry)
    result["formulas"] = formulas

    # Process PREREQUISITES - concept dependencies
    prerequisites: List[Dict[str, Any]] = []
    for entry in raw.get("prerequisites", []) or []:
        if not isinstance(entry, dict):
            continue
        from_term = entry.get("from")
        to_term = entry.get("to")
        from_can, from_disp = _canon(from_term)
        to_can, to_disp = _canon(to_term)
        if from_can and to_can and from_can != to_can:
            conf = entry.get("confidence", min_conf)
            try:
                conf_val = float(conf) if conf is not None else min_conf
            except Exception:
                conf_val = min_conf
            prerequisites.append({
                "from": from_disp,
                "from_canonical": from_can,
                "to": to_disp,
                "to_canonical": to_can,
                "confidence": conf_val,
            })
    result["prerequisites"] = prerequisites

    return result


def extract_pedagogy_relations(text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    if not text:
        return _default_payload()

    try:
        max_chars = int(os.getenv("PEDAGOGY_SECTION_MAX_CHARS", os.getenv("PEDAGOGY_LLM_MAX_CHARS", "8000")))
    except Exception:
        max_chars = 8000
    truncated = (text or "")[:max_chars]

    prompt = _build_prompt(truncated, meta)
    pedagogy_model = (
        os.getenv("PEDAGOGY_MODEL_HINT")
        or os.getenv("LLM_MODEL_MINI")
        or os.getenv("LLM_MODEL_NANO")
        or "gpt-4o-mini"
    )
    raw = call_json_chat(
        prompt,
        default=_default_payload(),
        system_prompt="You are an expert in extracting structured educational knowledge from text. Always respond with valid JSON following the requested schema.",
        retry_suffix='{"defines":[],"explains":[],"exemplifies":[],"formulas":[],"prerequisites":[]}',
        max_tokens=int(os.getenv("PEDAGOGY_LLM_MAX_TOKENS", os.getenv("LLM_PREVIEW_MAX_TOKENS", "2000"))),
        model_hint=pedagogy_model,
    )

    return _normalize_output(raw)
