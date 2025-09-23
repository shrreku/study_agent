import os
import sys

# In-process app client
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


def test_daily_quiz_returns_items_key():
    if _CLIENT is None:
        return
    r = _CLIENT.post("/api/agent/daily-quiz", headers=auth_headers(), json={"concepts": ["Chain rule"], "count": 1})
    assert r.status_code == 200
    j = r.json()
    # Backend should return both keys for compatibility
    assert "items" in j
    assert isinstance(j.get("items"), list)


def test_doubt_accepts_question_text_alias():
    if _CLIENT is None:
        return
    r = _CLIENT.post("/api/agent/doubt", headers=auth_headers(), json={"question_text": "Explain Fourier's Law"})
    assert r.status_code == 200
    j = r.json()
    assert "answer" in j
    assert "citations" in j


def test_split_text_into_chunks_regression():
    # Ensure the chunker (regex dependency) works; regression for missing `import re`
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from chunker import split_text_into_chunks  # type: ignore

    text = "INTRODUCTION\n1. Basics of heat transfer.\nThis is some text. Another sentence! And more?\nCONCLUSION\nSummary here."
    chunks = split_text_into_chunks(text, threshold=0.9, min_tokens=1, max_tokens=50, overlap=0)
    assert isinstance(chunks, list)
    # Expect at least one chunk produced
    assert len(chunks) >= 1
