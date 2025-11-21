from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List

# ensure project root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.resources import (  # type: ignore
    ResourceConceptsRequest,
    ResourceConceptFeedbackRequest,
    list_resource_concepts,
    submit_resource_concept_feedback,
)


class _FakeCursorFeedback:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        self.queries: List[Any] = []
        self._current_rows: List[Dict[str, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, q, params=None):  # pragma: no cover - trivial
        self.queries.append((q, params))
        # For this simple fake, always return the same rows for SELECT
        self._current_rows = self._rows

    def fetchall(self):
        return self._current_rows


class _FakeConnFeedback:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        self.cursors: List[_FakeCursorFeedback] = []
        self.committed = False

    def cursor(self, cursor_factory=None):  # pragma: no cover - trivial
        cur = _FakeCursorFeedback(self._rows)
        self.cursors.append(cur)
        return cur

    def close(self):  # pragma: no cover - trivial
        pass

    def commit(self):  # pragma: no cover - trivial
        self.committed = True


def test_submit_resource_concept_feedback_inserts_for_owned_resources(monkeypatch):
    import api.resources as resources  # type: ignore

    user_id = "00000000-0000-0000-0000-000000000001"
    resource_rows = [
        {"id": "11111111-1111-1111-1111-111111111111", "user_id": user_id},
        {"id": "22222222-2222-2222-2222-222222222222", "user_id": user_id},
    ]
    fake_conn = _FakeConnFeedback(resource_rows)

    monkeypatch.setattr(resources, "get_db_conn", lambda: fake_conn)

    body = ResourceConceptFeedbackRequest(
        resource_ids=["11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"],
        concept="Conduction",
        feedback_type="hide",
    )

    resp = asyncio.run(submit_resource_concept_feedback(body, user_id))

    assert resp["ok"] is True
    assert resp["updated"] == 2
    assert fake_conn.committed is True
    assert len(fake_conn.cursors) == 1
    cursor = fake_conn.cursors[0]
    insert_calls = [q for (q, _params) in cursor.queries if "INSERT INTO user_resource_concept_feedback" in q]
    assert len(insert_calls) == 2


class _SeqCursor:
    def __init__(self, sequences: List[List[Dict[str, Any]]]):
        self._sequences = sequences
        self._index = 0
        self._current_rows: List[Dict[str, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, q, params=None):  # pragma: no cover - trivial
        if self._index < len(self._sequences):
            self._current_rows = self._sequences[self._index]
        else:
            self._current_rows = []
        self._index += 1

    def fetchall(self):
        return self._current_rows


class _SeqConn:
    def __init__(self, sequences: List[List[Dict[str, Any]]]):
        self._sequences = sequences

    def cursor(self, cursor_factory=None):  # pragma: no cover - trivial
        return _SeqCursor(self._sequences)

    def close(self):  # pragma: no cover - trivial
        pass


def test_list_resource_concepts_marks_hidden_from_feedback(monkeypatch):
    import api.resources as resources  # type: ignore

    user_id = "00000000-0000-0000-0000-000000000001"

    resource_rows = [
        {"id": "11111111-1111-1111-1111-111111111111", "user_id": user_id},
    ]
    concept_rows = [
        {
            "resource_id": "11111111-1111-1111-1111-111111111111",
            "concept": "Conduction",
            "occurrences": 3,
            "pages": [1, 2],
            "roles": ["definition"],
        }
    ]
    feedback_rows = [
        {"concept": "Conduction"},
    ]

    sequences = [resource_rows, concept_rows, feedback_rows]
    fake_conn = _SeqConn(sequences)

    def fake_get_db_conn():  # pragma: no cover - trivial
        return fake_conn

    monkeypatch.setattr(resources, "get_db_conn", fake_get_db_conn)
    monkeypatch.setattr(resources, "fetch_mastery_map", lambda cur, uid: {})
    monkeypatch.setattr(resources, "fetch_prereq_chain", lambda names: names)

    body = ResourceConceptsRequest(resource_ids=["11111111-1111-1111-1111-111111111111"])

    summaries = asyncio.run(list_resource_concepts(body, user_id))
    assert isinstance(summaries, list)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.concept
    assert summary.hidden is True
