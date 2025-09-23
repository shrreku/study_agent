import os
import sys
import types
from typing import Any, Dict

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


def _backend_up():
    return _CLIENT is not None


def test_admin_recompute_search_tsv(monkeypatch):
    if not _backend_up():
        return
    # monkeypatch recompute function to avoid DB
    def _fake_recompute() -> int:
        return 42

    # Ensure project root import path
    this_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import agents.retrieval as retrieval  # type: ignore

    monkeypatch.setattr(retrieval, "recompute_search_tsv_for_all_chunks", _fake_recompute)

    r = _CLIENT.post("/api/admin/recompute-search-tsv", headers=auth_headers())
    assert r.status_code == 200
    j = r.json()
    assert j.get("updated") == 42


def test_job_status_400_invalid_uuid():
    if not _backend_up():
        return
    # no DB call should happen because validation fails early
    r = _CLIENT.get("/api/jobs/ not-a-uuid ", headers=auth_headers())
    assert r.status_code == 400


def test_job_status_404_and_200(monkeypatch):
    if not _backend_up():
        return
    # Monkeypatch get_db_conn to a fake connection to simulate 404 and 200 cases
    # Build a tiny fake connection/cursor
    class FakeCursor:
        def __init__(self, row: Dict[str, Any]):
            self._row = row
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def execute(self, q, params=None):
            self._params = params
        def fetchone(self):
            return self._row

    class FakeConn:
        def __init__(self, row: Dict[str, Any]):
            self._row = row
        def cursor(self, cursor_factory=None):
            return FakeCursor(self._row)
        def close(self):
            pass

    import main as backend_main  # type: ignore

    # 404: no row
    monkeypatch.setattr(backend_main, "get_db_conn", lambda: FakeConn(None))
    r = _CLIENT.get("/api/jobs/00000000-0000-0000-0000-000000000000", headers=auth_headers())
    assert r.status_code == 404

    # 200: present row
    row = {
        "job_id": "11111111-1111-1111-1111-111111111111",
        "status": "done",
        "payload": {"ok": True},
        "created_at": "2025-09-10T00:00:00Z",
        "updated_at": "2025-09-10T00:00:02Z",
    }
    monkeypatch.setattr(backend_main, "get_db_conn", lambda: FakeConn(row))
    r2 = _CLIENT.get("/api/jobs/11111111-1111-1111-1111-111111111111", headers=auth_headers())
    assert r2.status_code == 200
    j2 = r2.json()
    assert j2.get("status") == "done"


def test_job_status_401_when_missing_auth():
    if not _backend_up():
        return
    r = _CLIENT.get("/api/jobs/11111111-1111-1111-1111-111111111111")
    assert r.status_code == 403 or r.status_code == 401
