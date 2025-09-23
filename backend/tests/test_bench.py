import os
import sys
import types


def test_bench_pk_happy_path(monkeypatch):
    # Ensure imports resolve
    this_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import agents.retrieval as retrieval  # type: ignore
    import main as appmod  # type: ignore
    from fastapi.testclient import TestClient  # type: ignore

    # Avoid DB on startup
    monkeypatch.setattr(appmod, "ensure_schema", lambda: None)

    # Monkeypatch hybrid_search to deterministic rows
    def fake_hybrid_search(query: str, k: int = 5, **kwargs):
        return [
            {"id": f"{query}-1", "resource_id": "r1", "page_number": 1, "snippet": "s1", "score": 0.9},
            {"id": f"{query}-2", "resource_id": "r2", "page_number": 2, "snippet": "s2", "score": 0.8},
        ][:k]

    monkeypatch.setattr(retrieval, "hybrid_search", fake_hybrid_search)

    # Some environments have starlette/httpx incompat; skip if TestClient errors
    try:
        client = TestClient(appmod.app)
    except TypeError:
        return

    body = {"queries": ["heat flux", "boundary layer"], "k": 2, "sim_weight": 0.6, "bm25_weight": 0.4, "resource_boost": 1.0, "page_proximity_boost": False}
    r = client.post("/api/bench/pk", headers={"Authorization": "Bearer test-token"}, json=body)
    assert r.status_code == 200
    j = r.json()
    assert j.get("k") == 2
    assert isinstance(j.get("results"), list)
    assert len(j["results"]) == 2
    assert all(isinstance(it.get("ids"), list) for it in j["results"])
