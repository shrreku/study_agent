import sys
import types


def _make_fake_psycopg2(rows):
    mod = types.ModuleType("psycopg2")
    extras = types.SimpleNamespace(RealDictCursor=object)

    class FakeCursor:
        def __init__(self, rows):
            self._rows = rows
            self.executed = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, q, params=None):
            self.executed.append((q, params))

        def fetchall(self):
            return self._rows

    class FakeConn:
        def __init__(self, rows):
            self._rows = rows

        def cursor(self, cursor_factory=None):
            return FakeCursor(self._rows)

        def close(self):
            pass
        def commit(self):
            pass

    def connect(dsn):
        return FakeConn(rows)

    mod.extras = extras
    mod.connect = connect
    return mod


def test_recompute_search_tsv_monkeypatch(tmp_path, monkeypatch):
    # prepare fake rows
    rows = [{"id": "c1", "full_text": "This is a test"}, {"id": "c2", "full_text": "Another chunk"}]
    fake = _make_fake_psycopg2(rows)
    monkeypatch.setitem(sys.modules, "psycopg2", fake)

    # import function and run
    # ensure project root is on sys.path so `backend` package is importable
    import os
    this_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # also ensure common container working dir is present for imports
    if "/app" not in sys.path and os.path.exists("/app"):
        sys.path.insert(0, "/app")

    from agents.retrieval import recompute_search_tsv_for_all_chunks

    updated = recompute_search_tsv_for_all_chunks()
    assert updated == len(rows)


def test_hybrid_search_fusion_monkeypatch(monkeypatch):
    # monkeypatch embed query to avoid model dependency
    import os
    this_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if "/app" not in sys.path and os.path.exists("/app"):
        sys.path.insert(0, "/app")
    import agents.retrieval as retrieval

    monkeypatch.setattr(retrieval, "_embed_query", lambda q: [0.1] * 384)

    # prepare fake candidate rows returned by DB
    candidates = [
        {"id": "a", "resource_id": "r1", "page_number": 1, "snippet": "s1", "sim": 0.9, "bm25": 0.1},
        {"id": "b", "resource_id": "r2", "page_number": 5, "snippet": "s2", "sim": 0.4, "bm25": 0.8},
        {"id": "c", "resource_id": "r3", "page_number": 2, "snippet": "s3", "sim": 0.6, "bm25": 0.3},
    ]

    fake = _make_fake_psycopg2(candidates)
    monkeypatch.setitem(sys.modules, "psycopg2", fake)

    # run hybrid search
    rows = retrieval.hybrid_search("query text", k=2, sim_weight=0.6, bm25_weight=0.4, resource_boost=1.0, page_proximity_boost=True)
    assert isinstance(rows, list)
    assert len(rows) == 2
    # ensure results sorted by computed score
    assert rows[0]["score"] >= rows[1]["score"]


