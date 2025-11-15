"""Shared LLM helpers for StudyAgent.

- Minimal dependency surface (requests only)
- Strict JSON extraction with sentinel support and a single retry
- Environmental configuration for OpenAI-compatible providers
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional
import logging

import requests

# Thread-local storage for model override
_thread_local = threading.local()

JSON_SENTINEL_PATTERN = re.compile(r"BEGIN_STRICT_JSON\s*(\{[\s\S]*?\})\s*END_STRICT_JSON", re.IGNORECASE)
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
# Match complete JSON objects with proper nesting
FIRST_JSON_PATTERN = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}")


def _extract_json_blob(text: str) -> str:
    """Extract the first JSON object, preferring sentinel tags when present."""
    cleaned = (text or "").strip()
    match = JSON_SENTINEL_PATTERN.search(cleaned)
    if match:
        return match.group(1)
    cleaned = CODE_FENCE_PATTERN.sub("", cleaned).strip()
    
    # Try to find balanced JSON braces
    start_idx = cleaned.find('{')
    if start_idx == -1:
        return cleaned
    
    depth = 0
    in_string = False
    escape = False
    
    for i in range(start_idx, len(cleaned)):
        char = cleaned[i]
        
        if escape:
            escape = False
            continue
            
        if char == '\\':
            escape = True
            continue
            
        if char == '"' and not escape:
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return cleaned[start_idx:i+1]
    
    # If we couldn't find balanced braces, return what we have
    return cleaned


def _repair_json(json_str: str) -> str:
    """Attempt to repair common JSON errors from LLM responses."""
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix missing commas between objects/values (common LLM error)
    # This is risky, so we're conservative
    json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
    json_str = re.sub(r'}\s*\n\s*{', '},\n{', json_str)
    json_str = re.sub(r']\s*\n\s*\[', '],\n[', json_str)
    
    return json_str


def _build_base_url() -> Optional[str]:
    base = (
        os.getenv("OPENAI_API_BASE")
        or os.getenv("AIMLAPI_BASE_URL")
        or os.getenv("AIML_BASE_URL")
        or os.getenv("AIMLAPI_URL")
    )
    if not base:
        return None
    base_clean = base.rstrip("/")
    if not base_clean.endswith("/v1"):
        base_clean = base_clean + "/v1"
    return base_clean


def _resolve_api_key() -> Optional[str]:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("AIMLAPI_API_KEY")
        or os.getenv("AIML_KEY")
        or os.getenv("AIMLAPI_KEY")
    )


def _should_use_json_mode() -> bool:
    try:
        return os.getenv("LLM_RESPONSE_FORMAT_JSON", "1").lower() in {"1", "true", "yes"}
    except Exception:
        return True


@contextmanager
def model_override_context(model: str):
    """Context manager to temporarily override the model for all LLM calls in this thread."""
    old_value = getattr(_thread_local, 'model_override', None)
    _thread_local.model_override = model
    try:
        yield
    finally:
        if old_value is None:
            if hasattr(_thread_local, 'model_override'):
                delattr(_thread_local, 'model_override')
        else:
            _thread_local.model_override = old_value


def _get_model_override() -> Optional[str]:
    """Get the thread-local model override if set."""
    return getattr(_thread_local, 'model_override', None)


def _timeout_seconds() -> int:
    try:
        return int(os.getenv("LLM_TIMEOUT_SECS", "60"))
    except Exception:
        return 60


def call_json_chat(
    user_prompt: str,
    *,
    default: Dict[str, Any],
    system_prompt: str = "Return ONLY minified JSON. No markdown.",
    retry_suffix: Optional[str] = None,
    max_tokens: Optional[int] = None,
    model_hint: Optional[str] = None,
    text_field: Optional[str] = "response",
    allow_text_fallback: bool = False,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible chat completion and parse strict JSON."""
    mock_mode = os.getenv("USE_LLM_MOCK", "0").lower() in {"1", "true", "yes"}
    if mock_mode:
        logging.warning("json_chat_skipped reason=USE_LLM_MOCK_enabled")
        return default

    base_url = _build_base_url()
    api_key = _resolve_api_key()
    if not base_url:
        logging.error("json_chat_skipped reason=base_url_missing env_vars_checked=OPENAI_API_BASE,AIMLAPI_BASE_URL,AIML_BASE_URL,AIMLAPI_URL")
        return default
    if not api_key:
        logging.error("json_chat_skipped reason=api_key_missing env_vars_checked=OPENAI_API_KEY,AIMLAPI_API_KEY,AIML_KEY,AIMLAPI_KEY")
        return default

    # Check thread-local override first, then model_hint, then environment variables
    model = model_hint or _get_model_override() or os.getenv("LLM_MODEL_MINI") or os.getenv("LLM_MODEL_NANO")
    if not model:
        model = "gpt-4o-mini"

    user_content = (user_prompt or "").strip()
    if not user_content:
        user_content = "Provide a valid JSON response for the requested StudyAgent prompt."

    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens or int(os.getenv("LLM_PREVIEW_MAX_TOKENS", "2000")),
        "stream": False,
    }
    if _should_use_json_mode():
        body["response_format"] = {"type": "json_object"}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url}/chat/completions"
    logging.info(
        "json_chat_request model=%s url=%s json_mode=%s max_tokens=%s",
        model,
        url,
        _should_use_json_mode(),
        body.get("max_tokens"),
    )

    def _send(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=_timeout_seconds())
            logging.info("json_chat_response status=%s", resp.status_code)
        except requests.exceptions.Timeout:
            logging.error("json_chat_timeout model=%s timeout_secs=%d", model, _timeout_seconds())
            return None
        except Exception:
            logging.exception("json_chat_http_error")
            return None
        if not (200 <= resp.status_code < 300):
            try:
                body_preview = resp.text[:256]
            except Exception:
                body_preview = ""
            logging.error("json_chat_non_2xx status=%s body=%s", resp.status_code, body_preview)
            return None
        try:
            return resp.json()
        except Exception:
            logging.exception("json_chat_invalid_json")
            return None

    t0 = time.time()
    data = _send(body)
    content = (
        data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(data, dict)
        else ""
    )

    if not content.strip() and retry_suffix:
        logging.warning("json_chat_empty_first_try; retrying")
        retry_body = dict(body)
        retry_body["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}\n{retry_suffix}"},
        ]
        data = _send(retry_body)
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(data, dict)
            else ""
        )

    if not content.strip():
        logging.error("json_chat_empty_after_retry; returning default")
        return default

    try:
        blob = _extract_json_blob(content)
        parsed = json.loads((blob or "").strip())
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        logging.exception("json_chat_parse_failed")

    # If provider ignored JSON mode but returned text, optionally wrap it
    if allow_text_fallback and content.strip() and text_field:
        logging.warning("json_chat_wrap_text_fallback")
        wrapped: Dict[str, Any] = {text_field: content.strip()}
        if isinstance(default, dict) and "confidence" in default:
            wrapped["confidence"] = default.get("confidence", 0.5)
        return wrapped

    logging.warning("json_chat_fallback_to_default")
    return default
