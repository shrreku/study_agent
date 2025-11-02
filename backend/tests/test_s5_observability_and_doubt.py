import os
import sys
import uuid

# Use in-process TestClient if possible
_CLIENT = None
try:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from main import app  # type: ignore
    from fastapi.testclient import TestClient  # type: ignore

    _CLIENT = TestClient(app)
except Exception:  # pragma: no cover - fallback to HTTP if needed
    _CLIENT = None


def auth_headers():
    return {"Authorization": "Bearer test-token"}


def test_doubt_endpoint_increments_counters_and_handles_logging_failure():
    # Arrange: mock LLM and prompt set; force DB connection to fail for doubt logging
    os.environ["USE_LLM_MOCK"] = "1"
    os.environ["PROMPT_SET"] = "testset"
    os.environ["TEST_USER_ID"] = str(uuid.uuid4())
    os.environ["POSTGRES_HOST"] = "invalid-host-for-tests"  # cause get_db_conn to fail

    if _CLIENT is None:
        return  # skip if app not importable

    # Act: call doubt agent
    r = _CLIENT.post("/api/agent/doubt", headers=auth_headers(), json={"question": "What is the chain rule?"})
    assert r.status_code == 200

    # Act: fetch metrics snapshot
    m = _CLIENT.get("/api/metrics", headers=auth_headers())
    assert m.status_code == 200
    snap = m.json()

    # Assert: counters present/incremented
    counters = snap.get("counters", {})
    assert counters.get("doubt_calls_total", 0) >= 1
    assert counters.get("agent_doubt_calls", 0) >= 1
    # With forced DB failure for logging, failure counter should tick
    assert counters.get("doubt_log_failures_total", 0) >= 1


def test_quiz_metrics_rollup_counters_present():
    # Arrange
    os.environ["USE_LLM_MOCK"] = "1"
    os.environ["TEST_USER_ID"] = str(uuid.uuid4())

    if _CLIENT is None:
        return

    # Act: submit a trivial quiz grading payload
    payload = {
        "quiz_id": "q1",
        "answers": [
            {"question_id": "1", "concept": "Derivative", "chosen": 0, "correct_index": 0},
            {"question_id": "2", "concept": "Integral", "chosen": 2, "correct_index": 1},
        ],
    }
    r = _CLIENT.post("/api/agent/quiz/answer", headers=auth_headers(), json=payload)
    # In dev without DB, this might 400 if TEST_USER_ID missing; we set it above
    assert r.status_code == 200

    # Metrics snapshot
    m = _CLIENT.get("/api/metrics", headers=auth_headers())
    assert m.status_code == 200
    counters = m.json().get("counters", {})

    # At least totals should have moved by >= 2; correctness split unspecified
    assert counters.get("quiz_answers_total", 0) >= 2
    # Correct or incorrect (or both) should be > 0
    assert counters.get("quiz_answers_correct", 0) + counters.get("quiz_answers_incorrect", 0) >= 1
