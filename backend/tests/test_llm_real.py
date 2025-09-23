import os
import requests
import pytest

# Only run these with explicit opt-in. Requires real LLM credentials in env.
if os.getenv("USE_REAL_LLM", "0") != "1":
    pytest.skip("Skipping real LLM tests; set USE_REAL_LLM=1 to enable.", allow_module_level=True)


def _auth_headers():
    return {"Authorization": "Bearer test-token"}


def _base_url():
    return os.getenv("BASE_URL", "http://localhost:8000")


def _get(path: str):
    return requests.get(f"{_base_url()}{path}", headers=_auth_headers())


def _post(path: str, json_body: dict):
    return requests.post(f"{_base_url()}{path}", headers=_auth_headers(), json=json_body)


def test_llm_smoke_ok():
    resp = _get("/api/llm/smoke")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("ok") is True
    assert isinstance(data.get("reply"), str)
    assert data.get("model")


def test_llm_preview_real_json_shape():
    # Requires OPENAI_API_BASE/OPENAI_API_KEY configured
    if not (os.getenv("OPENAI_API_BASE") and (os.getenv("OPENAI_API_KEY") or os.getenv("AIMLAPI_API_KEY"))):
        pytest.skip("Missing OPENAI_API_BASE or API key for real LLM test")
    r = _post("/api/llm/preview", {"text": "Heat conduction in solids. q = -k dT/dx"})
    assert r.status_code == 200
    j = r.json()
    # Loose schema: must contain these keys
    assert set(["chunk_type", "concepts", "math_expressions"]).issubset(j.keys())
    assert isinstance(j.get("concepts"), list)
    assert isinstance(j.get("math_expressions"), list)
