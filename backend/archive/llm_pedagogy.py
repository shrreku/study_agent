"""Minimal LLM helper for pedagogy extraction used by the KG pipeline.

- Keeps dependencies light (requests only)
- Robust JSON handling with sentinel tags and a retry path
- Normalizes output to the expected pedagogy schema
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from core.kg import canonicalize_concept


PEDAGOGY_DEFAULT_OUTPUT: Dict[str, list] = {
    "defines": [],
    "explains": [],
    "exemplifies": [],
    "derives": [],
    "proves": [],
    "figure_links": [],
    "prereqs": [],
    "evidence": [],
}


def _pedagogy_default() -> Dict[str, list]:
    return {k: [] for k in PEDAGOGY_DEFAULT_OUTPUT.keys()}


def _extract_json_blob(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r"BEGIN_STRICT_JSON\s*(\{[\s\S]*?\})\s*END_STRICT_JSON", s)
    if m:
        return m.group(1)
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE | re.MULTILINE)
    m2 = re.search(r"\{[\s\S]*?\}", s)
    return m2.group(0) if m2 else s


def _build_pedagogy_prompt(text: str, meta: Dict[str, Any]) -> str:
    title = meta.get("title") or meta.get("section_title") or ""
    chunk_type = meta.get("chunk_type") or ""
    parts = [
        "You are an educational content analyst. Extract pedagogical relations and respond with STRICT JSON only.",
        "The JSON object must have keys: defines, explains, exemplifies, derives, proves, figure_links, prereqs, evidence.",
        "Each field is an array. Schema:",
        'defines:[{"term":str, "aliases":[str]}], explains:[str], exemplifies:[str],',
        'derives:[{"about":str, "formula_latex":str}], proves:[str], figure_links:[{"label":str, "concepts":[str]}],',
        'prereqs:[{"from":str, "to":str, "confidence":float}], evidence:[{"edge":str, "sentences":[str], "confidence":float}].',
        "Confidence must be between 0 and 1. Omit fields or entries you are unsure about.",
        f"Context title={json.dumps(str(title))}, chunk_type={json.dumps(str(chunk_type))}.",
        "Wrap your JSON strictly between the tags BEGIN_STRICT_JSON and END_STRICT_JSON.",
        "Example minimal object:",
        'BEGIN_STRICT_JSON {"defines":[],"explains":[],"exemplifies":[],"derives":[],"proves":[],"figure_links":[],"prereqs":[],"evidence":[]} END_STRICT_JSON',
    ]
    instructions = " ".join(parts)
    return f"{instructions} Return ONLY JSON. Text follows:\n{text}"


def _normalize_pedagogy_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    result = _pedagogy_default()
    if not isinstance(raw, dict):
        return result

    try:
        min_conf = float(os.getenv("PEDAGOGY_LLM_MIN_CONF", "0.7"))
    except Exception:
        min_conf = 0.7

    def _canon(term: Optional[str]) -> Tuple[str, str]:
        return canonicalize_concept(term or "")

    def _gate(conf: Optional[float]) -> bool:
        if conf is None:
            return True
        try:
            return float(conf) >= min_conf
        except Exception:
            return False

    defines: List[Dict[str, Any]] = []
    for entry in raw.get("defines", []) or []:
        if not isinstance(entry, dict):
            continue
        term = entry.get("term")
        canonical, display = _canon(term)
        if not canonical:
            continue
        aliases: List[str] = []
        for alias in entry.get("aliases", []) or []:
            alias_can, alias_disp = _canon(alias)
            if alias_can and alias_can != canonical:
                aliases.append(alias_disp)
        defines.append({"term": display, "canonical": canonical, "aliases": aliases})
    result["defines"] = defines

    for key in ("explains", "exemplifies", "proves"):
        vals: List[Dict[str, Any]] = []
        for item in raw.get(key, []) or []:
            canonical, display = _canon(item)
            if canonical:
                vals.append({"term": display, "canonical": canonical})
        result[key] = vals

    derives: List[Dict[str, Any]] = []
    for entry in raw.get("derives", []) or []:
        if not isinstance(entry, dict):
            continue
        about = entry.get("about")
        canonical, display = _canon(about)
        if not canonical:
            continue
        formula = str(entry.get("formula_latex") or "").strip()
        if not formula:
            continue
        derives.append({"about": display, "canonical": canonical, "formula_latex": formula})
    result["derives"] = derives

    figure_links: List[Dict[str, Any]] = []
    for entry in raw.get("figure_links", []) or []:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "").strip()
        if not label:
            continue
        concepts: List[Dict[str, Any]] = []
        for concept in entry.get("concepts", []) or []:
            canonical, display = _canon(concept)
            if canonical:
                concepts.append({"term": display, "canonical": canonical})
        figure_links.append({"label": label, "concepts": concepts})
    result["figure_links"] = figure_links

    prereqs: List[Dict[str, Any]] = []
    for entry in raw.get("prereqs", []) or []:
        if not isinstance(entry, dict):
            continue
        pf = entry.get("from")
        pt = entry.get("to")
        conf = entry.get("confidence")
        if not _gate(conf):
            continue
        p_can, p_disp = _canon(pf)
        t_can, t_disp = _canon(pt)
        if not p_can or not t_can or p_can == t_can:
            continue
        prereqs.append({
            "from": p_disp,
            "from_canonical": p_can,
            "to": t_disp,
            "to_canonical": t_can,
            "confidence": float(conf) if conf is not None else min_conf,
        })
    result["prereqs"] = prereqs

    evidence: List[Dict[str, Any]] = []
    for entry in raw.get("evidence", []) or []:
        if not isinstance(entry, dict):
            continue
        edge = str(entry.get("edge") or "").strip()
        if not edge:
            continue
        conf = entry.get("confidence")
        if not _gate(conf):
            continue
        sentences = [str(s).strip() for s in entry.get("sentences", []) or [] if str(s).strip()]

        relation = None
        relation_target = None
        relation_from = None
        relation_to = None

        parsed = edge
        if ":" in edge:
            relation, remainder = edge.split(":", 1)
            relation = relation.strip().upper()
            remainder = remainder.strip()
        else:
            relation = edge.strip().upper()
            remainder = ""

        if relation == "PREREQ":
            if "->" in remainder:
                left, right = remainder.split("->", 1)
                relation_from = left.strip()
                relation_to = right.strip()
        else:
            relation_target = remainder or None

        ev: Dict[str, Any] = {
            "edge": edge,
            "relation": relation,
            "sentences": sentences,
            "confidence": float(conf) if conf is not None else min_conf,
        }
        if relation_target:
            target_can, target_disp = _canon(relation_target)
            if target_can:
                ev["target_canonical"] = target_can
                ev["target"] = target_disp
        if relation_from:
            from_can, from_disp = _canon(relation_from)
            if from_can:
                ev["from_canonical"] = from_can
                ev["from"] = from_disp
        if relation_to:
            to_can, to_disp = _canon(relation_to)
            if to_can:
                ev["to_canonical"] = to_can
                ev["to"] = to_disp
        evidence.append(ev)
    result["evidence"] = evidence

    return result


def call_llm_json(prompt: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if os.getenv("USE_LLM_MOCK", "0").lower() in {"1", "true", "yes"}:
            return default
    except Exception:
        pass

    base = (
        os.getenv("OPENAI_API_BASE")
        or os.getenv("AIMLAPI_BASE_URL")
        or os.getenv("AIML_BASE_URL")
        or os.getenv("AIMLAPI_URL")
    )
    key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("AIMLAPI_API_KEY")
        or os.getenv("AIML_KEY")
        or os.getenv("AIMLAPI_KEY")
    )
    model = os.getenv("LLM_MODEL_MINI", os.getenv("LLM_MODEL_NANO", "gpt-4o-mini"))
    if not base or not key:
        return default

    base_clean = base.rstrip("/")
    endpoint = "/chat/completions" if base_clean.endswith("/v1") else "/v1/chat/completions"
    url = base_clean + endpoint

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    use_json_mode = (os.getenv("LLM_RESPONSE_FORMAT_JSON", "1").lower() in {"1", "true", "yes"})

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return ONLY minified JSON. No markdown."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": int(os.getenv("LLM_PREVIEW_MAX_TOKENS", "800")),
        "stream": False,
    }
    if use_json_mode:
        body["response_format"] = {"type": "json_object"}

    t0 = time.time()
    try:
        r = requests.post(url, headers=headers, json=body, timeout=int(os.getenv("LLM_TIMEOUT_SECS", "60")))
        if r.status_code != 200:
            return default
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        blob = _extract_json_blob(content)
        parsed = json.loads((blob or "").strip()) if blob else None
        if isinstance(parsed, dict):
            return parsed
        # Retry once with stricter sentinel hint
        retry_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return ONLY minified JSON."},
                {"role": "user", "content": "Wrap valid JSON strictly between BEGIN_STRICT_JSON and END_STRICT_JSON for: " + prompt + "\nBEGIN_STRICT_JSON {} END_STRICT_JSON"},
            ],
            "temperature": 0.0,
            "max_tokens": int(os.getenv("LLM_PREVIEW_MAX_TOKENS", "800")),
            "stream": False,
        }
        if use_json_mode:
            retry_body["response_format"] = {"type": "json_object"}
        r2 = requests.post(url, headers=headers, json=retry_body, timeout=int(os.getenv("LLM_TIMEOUT_SECS", "60")))
        if r2.status_code != 200:
            return default
        data2 = r2.json()
        content2 = data2.get("choices", [{}])[0].get("message", {}).get("content", "")
        blob2 = _extract_json_blob(content2)
        parsed2 = json.loads((blob2 or "").strip()) if blob2 else None
        return parsed2 if isinstance(parsed2, dict) else default
    except Exception:
        return default


def extract_pedagogy_relations(text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    if not text:
        return _pedagogy_default()
    try:
        max_chars = int(os.getenv("PEDAGOGY_LLM_MAX_CHARS", os.getenv("LLM_PREVIEW_MAX_CHARS", "2000")))
    except Exception:
        max_chars = 2000
    truncated = (text or "")[:max_chars]
    prompt = _build_pedagogy_prompt(truncated, meta)
    default = _pedagogy_default()
    raw = call_llm_json(prompt, default)
    return _normalize_pedagogy_output(raw)
