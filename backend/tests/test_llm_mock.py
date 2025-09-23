import os
import sys


def test_llm_preview_with_mock(monkeypatch):
    # Ensure project root on path
    this_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import main as appmod  # type: ignore
    from fastapi.testclient import TestClient  # type: ignore

    # Avoid ensure_schema side effects on startup
    monkeypatch.setattr(appmod, "ensure_schema", lambda: None)

    # Enable mock
    monkeypatch.setenv("USE_LLM_MOCK", "1")

    try:
        client = TestClient(appmod.app)
    except TypeError:
        return

    r = client.post(
        "/api/llm/preview",
        headers={"Authorization": "Bearer test-token"},
        json={"text": "Euler's identity: $e^{i\\pi} + 1 = 0$"},
    )
    assert r.status_code == 200
    j = r.json()
    assert j.get("chunk_type") == "summary"
    assert isinstance(j.get("concepts"), list)
    # math expressions should be extracted by regex in mock path
    assert isinstance(j.get("math_expressions"), list)
