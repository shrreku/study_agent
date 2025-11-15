"""Prompt registry with YAML-backed prompt sets and simple rendering.

- Reads PROMPT_SET from environment (default: 'baseline')
- Loads prompts/<set>.yaml on first use and caches content; auto-reloads on mtime change
- Provides get(key_path) e.g., 'quiz.mcq' and render(template, vars)
- Falls back to built-in baseline prompts when files are missing
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import os
import time
import threading

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is optional but recommended
    yaml = None  # type: ignore


_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {"set": None, "mtime": 0.0, "prompts": {}}
_INGEST_CACHE: Dict[str, Any] = {"mtime": 0.0, "prompts": {}}


def _default_prompts() -> Dict[str, Any]:
    """Return a minimal built-in baseline prompt set to keep app runnable."""
    return {
        "quiz": {
            "mcq": (
                "Create ONE multiple-choice question grounded in the academic context.\n"
                "Concept: {{concept}}\n"
                "Context snippet:\n{{snippet}}\n"
                "Return ONLY JSON with keys: {\"question\":string, \"options\":[string,string,string,string], \"answer_index\":number, \"explanation\":string}."
            )
        },
        "doubt": {
            "answer": (
                "You are a concise tutor. Only answer using the provided context and citations.\n"
                "If the context is empty or not relevant to the question, respond exactly with:\n"
                "I don't know based on the provided materials.\n"
                "Do not use outside knowledge. Do not fabricate.\n"
                "Question: {{question}}\n\n"
                "Context:\n{{context}}\n\n"
                "Return ONLY JSON: {\"answer\":string, \"expanded_steps\":string}. Citations are handled outside of this response."
            )
        },
        "study": {
            "summary": (
                "Write a short, student-friendly summary (1-2 sentences) using the snippet.\n"
                "Topic: {{concept}}\n"
                "Snippet:\n{{snippet}}\n"
                "Return ONLY JSON: {\"summary\": string}."
            )
        },
        "analysis": {
            "summary": (
                "Provide a short summary (2-3 sentences) of student's strengths and weaknesses.\n"
                "Strengths: {{strengths}}\n"
                "Weaknesses: {{weaknesses}}\n"
                "Return ONLY JSON: {\"summary\": string}."
            )
        },
        "tutor": {
            "classify": (
                "You are classifying a student's latest message in a tutoring session.\n"
                "Return ONLY JSON with keys intent, affect, concept, confidence, needs_escalation.\n"
                "Intent options: question, answer, reflection, off_topic, greeting, unknown.\n"
                "Affect options: confused, unsure, engaged, frustrated, neutral.\n"
                "Student message: {{student_message}}\n"
                "Target concepts: {{target_concepts}}\n"
                "Last concept: {{last_concept}}"
            ),
            "explain": (
                "You are an adaptive tutor explaining a concept using only provided context.\n"
                "Return ONLY JSON: {\"response\": string, \"confidence\": number}.\n"
                "If context insufficient respond with: Let's review that from your materials first.\n"
                "Concept: {{concept}}\n"
                "Level: {{level}}\n"
                "Context:\n{{context}}"
            ),
            "ask": (
                "Generate ONE grounded formative question.\n"
                "Return ONLY JSON: {\"question\": string, \"answer\": string, \"confidence\": number, \"options\": [..]}.\n"
                "If context insufficient respond with: Let's review that from your materials first.\n"
                "Concept: {{concept}}\n"
                "Level: {{level}}\n"
                "Context:\n{{context}}"
            ),
            "hint": (
                "Provide a grounded hint without giving the full answer.\n"
                "Return ONLY JSON: {\"response\": string, \"confidence\": number}.\n"
                "If context insufficient respond with: Let's review that from your materials first.\n"
                "Concept: {{concept}}\n"
                "Level: {{level}}\n"
                "Context:\n{{context}}"
            ),
            "reflect": (
                "Lead a brief reflection grounded in context.\n"
                "Return ONLY JSON: {\"response\": string, \"confidence\": number}.\n"
                "If context insufficient respond with: Let's review that from your materials first.\n"
                "Concept: {{concept}}\n"
                "Level: {{level}}\n"
                "Context:\n{{context}}"
            ),
            "prereq_review": (
                "You are a supportive tutor helping a student prepare for a target concept by reviewing prerequisites first.\n"
                "Return ONLY JSON: {\"response\": string, \"confidence\": number}.\n"
                "Student target concept: {{target_concept}}\n"
                "Prerequisites to review first: {{prereq_names}}\n"
                "Primary prereq for a brief grounded recap: {{first_prereq}}\n"
                "Context (snippets from materials):\n{{context}}\n"
                "Write a short, encouraging message that: (1) acknowledges interest in the target concept, (2) explains that we will briefly review the prerequisite, (3) gives a concise recap grounded ONLY in the provided context, and (4) reassures we'll reach the target concept after."
            ),
        },
        "tutor_rl": {
            "critic_score": (
                "You are an independent pedagogy critic reviewing a tutor response.\n"
                "Evaluate clarity, factual accuracy, quality of grounding, and hallucination risk.\n"
                "Use the observation summary and retrieved snippets for reference.\n"
                "Output BEGIN_STRICT_JSON ... END_STRICT_JSON with keys:\n"
                "  {\n"
                '    "clarity": number 0..1,\n'
                '    "accuracy": number 0..1,\n'
                '    "support": number 0..1,\n'
                '    "hallucination_flag": boolean,\n'
                '    "notes": string (≤280 chars),\n'
                '    "confidence": number 0..1\n'
                "  }\n"
                "Observation:\n"
                "  Focus concept: {{focus_concept}}\n"
                "  Action type: {{action_type}}\n"
                "  Classifier intent: {{intent}}\n"
                "  Retrieved snippets:\n"
                "{{retrieved_context}}\n"
                "Tutor response:\n"
                "{{response}}\n"
                "BEGIN_STRICT_JSON\n"
                "{\n"
                '  "clarity": 0.8,\n'
                '  "accuracy": 0.8,\n'
                '  "support": 0.8,\n'
                '  "hallucination_flag": false,\n'
                '  "notes": "",\n'
                '  "confidence": 0.75\n'
                "}\n"
                "END_STRICT_JSON"
            ),
            "critic_preference": (
                "You are comparing multiple tutor responses to the same observation. Choose the best candidate.\n"
                "Consider clarity, accuracy, grounding, pedagogical value, and safety.\n"
                "Output BEGIN_STRICT_JSON ... END_STRICT_JSON with keys:\n"
                "  {\n"
                '    "chosen": integer index of the preferred candidate (0-based),\n'
                '    "scores": array[number] matching the number of candidates (each 0..1),\n'
                '    "confidence": number 0..1,\n'
                '    "reason": string (≤200 chars)\n'
                "  }\n"
                "Observation summary:\n"
                "  Focus concept: {{focus_concept}}\n"
                "  Action type(s): {{action_types}}\n"
                "  Classifier intent: {{intent}}\n"
                "Candidates:\n"
                "{{candidate_summaries}}\n"
                "Respond with strict JSON only.\n"
                "BEGIN_STRICT_JSON\n"
                "{\n"
                '  "chosen": 0,\n'
                '  "scores": [0.7],\n'
                '  "confidence": 0.6,\n'
                '  "reason": ""\n'
                "}\n"
                "END_STRICT_JSON"
            ),
        },
    }


def _active_set_name() -> str:
    return os.getenv("PROMPT_SET", "baseline").strip() or "baseline"


def _prompts_dir() -> str:
    # prompts/ folder at repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "prompts"))


def _load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _ensure_loaded() -> None:
    with _LOCK:
        set_name = _active_set_name()
        pdir = _prompts_dir()
        ypath = os.path.join(pdir, f"{set_name}.yaml")
        try:
            mtime = os.path.getmtime(ypath)
        except Exception:
            mtime = 0.0
        if _CACHE["set"] != set_name or _CACHE["mtime"] != mtime:
            # (re)load
            data = _load_yaml(ypath)
            if not isinstance(data, dict) or not data:
                data = _default_prompts()
            _CACHE["set"] = set_name
            _CACHE["mtime"] = mtime
            _CACHE["prompts"] = data


def get(key_path: str, default: Optional[str] = None) -> str:
    """Get a template string by dotted key path (e.g., 'quiz.mcq')."""
    parts = key_path.split(".") if key_path else []
    if parts and parts[0] == "ingest":
        pdir = _prompts_dir()
        ypath = os.path.join(pdir, "enhanced_ingest.yaml")
        try:
            mtime = os.path.getmtime(ypath)
        except Exception:
            mtime = 0.0
        if _INGEST_CACHE.get("mtime") != mtime:
            data = _load_yaml(ypath)
            if not isinstance(data, dict):
                data = {}
            _INGEST_CACHE["mtime"] = mtime
            _INGEST_CACHE["prompts"] = data
        cur_ing: Any = _INGEST_CACHE.get("prompts", {})
        for p in parts:
            if not isinstance(cur_ing, dict):
                cur_ing = None
                break
            cur_ing = cur_ing.get(p)
        if isinstance(cur_ing, str) and cur_ing.strip():
            return cur_ing
    _ensure_loaded()
    cur: Any = _CACHE.get("prompts", {})
    for p in parts:
        if not isinstance(cur, dict):
            cur = None
            break
        cur = cur.get(p)
    if isinstance(cur, str) and cur.strip():
        return cur
    # fallback to built-in defaults
    built_in = _default_prompts()
    cur2: Any = built_in
    for p in parts:
        if not isinstance(cur2, dict):
            cur2 = None
            break
        cur2 = cur2.get(p)
    if isinstance(cur2, str) and cur2.strip():
        return cur2
    return default or ""


def render(template_str: str, vars: Dict[str, Any]) -> str:
    """Very small {{var}} replacement; no logic, just string replace."""
    if not template_str:
        return ""
    out = str(template_str)
    for k, v in (vars or {}).items():
        out = out.replace("{{" + str(k) + "}}", str(v))
    return out


def active_set() -> str:
    """Expose the active prompt set name for metrics tagging."""
    return _active_set_name()
