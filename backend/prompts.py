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
    _ensure_loaded()
    parts = key_path.split(".") if key_path else []
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
