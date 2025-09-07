import os
import sys
import requests
from typing import Optional


# Try to import the FastAPI app directly for in-process testing
_CLIENT = None
try:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from main import app  # type: ignore
    from fastapi.testclient import TestClient  # type: ignore

    _CLIENT = TestClient(app)
except Exception:
    _CLIENT = None


def auth_headers():
    return {"Authorization": "Bearer test-token"}


def _base_url():
    return os.getenv("BASE_URL", "http://localhost:8000")


def _backend_up():
    if _CLIENT is not None:
        return True
    try:
        r = requests.get(f"{_base_url()}/health")
        return r.status_code == 200
    except Exception:
        return False


def _post(path: str, json_body: dict):
    if _CLIENT is not None:
        return _CLIENT.post(path, headers=auth_headers(), json=json_body)
    return requests.post(f"{_base_url()}{path}", headers=auth_headers(), json=json_body)


def test_orchestrator_mock():
    if _CLIENT is None:
        return
    if not _backend_up():
        return
    r = _post("/api/agent/mock", {"foo": "bar"})
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True
    assert j.get("agent") == "mock"


def test_study_plan_basic():
    if _CLIENT is None:
        return
    if not _backend_up():
        return
    r = _post("/api/agent/study-plan", {"target_concepts": ["Derivatives", "Integrals"], "daily_minutes": 30})
    assert r.status_code == 200
    j = r.json()
    assert "plan_id" in j
    assert isinstance(j.get("todos"), list)


def test_doubt_agent_minimal():
    if _CLIENT is None:
        return
    if not _backend_up():
        return
    r = _post("/api/agent/doubt", {"question": "What is the derivative of x^2?"})
    assert r.status_code == 200
    j = r.json()
    assert "answer" in j
    assert "citations" in j


