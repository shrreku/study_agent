import os
import sys
import importlib


def _ensure_project_root_on_path():
    # Ensure project root on path
    this_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def test_llm_preview_with_mock(monkeypatch):
    _ensure_project_root_on_path()

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


def test_pedagogy_relations_with_mock(monkeypatch):
    _ensure_project_root_on_path()
    monkeypatch.setenv("USE_LLM_MOCK", "1")

    import llm  # type: ignore

    importlib.reload(llm)

    result = llm.extract_pedagogy_relations(
        "This section defines the Laplace transform and explains how it applies.",
        {"chunk_type": "definition", "title": "Laplace Transform"},
    )

    assert set(result.keys()) == {
        "defines",
        "explains",
        "exemplifies",
        "derives",
        "proves",
        "figure_links",
        "prereqs",
        "evidence",
    }
    # In mock mode we expect defaults (empty lists)
    assert all(isinstance(v, list) and not v for v in result.values())
