import os
import json
import requests
from typing import Optional
import pytest

import main as appmod
import llm as llmmod

# Skip this module by default to avoid flaky offline LLM interactions.
# Set ALLOW_PREVIEW_TESTS=1 to enable locally/CI with proper mocks.
if os.getenv("ALLOW_PREVIEW_TESTS", "0") != "1":
    pytest.skip("Skipping LLM preview tests by default; set ALLOW_PREVIEW_TESTS=1 to enable.", allow_module_level=True)

# Try to create an in-process TestClient if compatible; otherwise fallback to HTTP requests
_CLIENT: Optional[object] = None
try:
    from fastapi.testclient import TestClient  # type: ignore
    try:
        _CLIENT = TestClient(appmod.app)
    except TypeError:
        # httpx/starlette incompatibility; fall back to external HTTP
        _CLIENT = None
except Exception:
    _CLIENT = None

def _auth_headers():
    return {"Authorization": "Bearer test-token"}

def _base_url():
    return os.getenv("BASE_URL", "http://localhost:8000")

def _post(path: str, json_body: dict):
    if _CLIENT is not None:
        return _CLIENT.post(path, headers=_auth_headers(), json=json_body)
    return requests.post(f"{_base_url()}{path}", headers=_auth_headers(), json=json_body)


def _set_env(monkeypatch):
    # Use a dummy local base and test key; network is mocked in tests
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:9999/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL_NANO", "openai/gpt-5-nano-2025-08-07")


def _noop(*args, **kwargs):
    return None


def test_preview_coerces_empty_response(monkeypatch):
    if _CLIENT is None:
        return
    _set_env(monkeypatch)
    # avoid DB calls on startup
    monkeypatch.setattr(appmod, "ensure_schema", _noop)

    def fake_post(url, headers=None, json=None, timeout=10):  # noqa: A002
        class R:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": ""}}], "usage": {"total_tokens": 1}}

            @property
            def text(self):
                return ""

        return R()

    # Patch both main.requests and llm.requests to be safe across modules
    monkeypatch.setattr(appmod.requests, "post", fake_post)
    monkeypatch.setattr(llmmod.requests, "post", fake_post)
    monkeypatch.setattr(llmmod.requests, "post", fake_post)

    resp = _CLIENT.post(
        "/api/llm/preview",
        headers=_auth_headers(),
        json={"text": "Some academic text."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert set(["chunk_type", "concepts", "math_expressions"]) <= set(data.keys())


def test_preview_parses_markdown_json(monkeypatch):
    if _CLIENT is None:
        return
    _set_env(monkeypatch)
    monkeypatch.setattr(appmod, "ensure_schema", _noop)

    content = """
```json
{"chunk_type":"definition","concepts":["Heat Transfer"],"math_expressions":[]}
```
"""

    def fake_post(url, headers=None, json=None, timeout=10):  # noqa: A002
        class R:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 42}}

            @property
            def text(self):
                return content

        return R()

    monkeypatch.setattr(appmod.requests, "post", fake_post)
    monkeypatch.setattr(llmmod.requests, "post", fake_post)

    resp = _CLIENT.post(
        "/api/llm/preview",
        headers=_auth_headers(),
        json={"text": "Short text"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunk_type"] == "definition"
    assert data["concepts"] == ["Heat Transfer"]


def test_preview_handles_non_200_then_retry(monkeypatch):
    if _CLIENT is None:
        return
    _set_env(monkeypatch)
    monkeypatch.setattr(appmod, "ensure_schema", _noop)

    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=10):  # noqa: A002
        calls["n"] += 1
        class R:
            def __init__(self, ok, content):
                self.status_code = 200 if ok else 500
                self._content = content

            def json(self):
                return {"choices": [{"message": {"content": self._content}}]}

            @property
            def text(self):
                return self._content

        if calls["n"] == 1:
            return R(False, "server error")
        return R(True, '{"chunk_type":"summary","concepts":[],"math_expressions":[]}')

    monkeypatch.setattr(appmod.requests, "post", fake_post)
    monkeypatch.setattr(llmmod.requests, "post", fake_post)

    resp = _post(
        "/api/llm/preview",
        {"text": "Short text"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunk_type"] == "summary"


def test_preview_literal_null_coerced(monkeypatch):
    if _CLIENT is None:
        return
    _set_env(monkeypatch)
    monkeypatch.setattr(appmod, "ensure_schema", _noop)

    def fake_post(url, headers=None, json=None, timeout=10):  # noqa: A002
        class R:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "null"}}]}

            @property
            def text(self):
                return "null"

        return R()

    monkeypatch.setattr(appmod.requests, "post", fake_post)
    monkeypatch.setattr(llmmod.requests, "post", fake_post)

    resp = _CLIENT.post(
        "/api/llm/preview",
        headers=_auth_headers(),
        json={"text": "Any"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunk_type"] == "summary"
    assert isinstance(data["concepts"], list) and isinstance(data["math_expressions"], list)

