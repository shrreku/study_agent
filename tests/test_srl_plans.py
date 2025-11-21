from __future__ import annotations

import sys
import os
import unittest

# Mock external dependencies that backend imports may expect
sys.modules["psycopg2"] = type("Mock", (), {})()
mock_psycopg2_extras = type("MockPsycopg2Extras", (), {"Json": object})()
sys.modules["psycopg2.extras"] = mock_psycopg2_extras

# Stub core and core.db with a dummy get_db_conn so imports in backend.agents.* succeed
mock_core = type("MockCore", (), {})()
mock_core_db = type("MockDB", (), {"get_db_conn": lambda *args, **kwargs: None})()
sys.modules.setdefault("core", mock_core)
sys.modules.setdefault("core.db", mock_core_db)

# Add repo root to path so we can import backend.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.agents.tutor.mdp.plans import (
    SessionPlanEntry,
    SessionPlan,
    ConceptPlanStep,
    ConceptPlan,
)
from backend.agents.tutor.mdp.adapters import (
    session_plan_from_policy,
    session_plan_to_policy,
    concept_plan_from_policy,
    concept_plan_to_policy,
)
from backend.agents.tutor.state import TutorSessionPolicy


def test_session_plan_roundtrip_through_policy() -> None:
    policy = TutorSessionPolicy()
    entries = [
        SessionPlanEntry(concept_id="c1", target_mastery=0.8),
        SessionPlanEntry(concept_id="c2", target_mastery=0.7),
    ]
    plan = SessionPlan(
        strategy="learning_path",
        entries=entries,
        plan_id="test-plan",
        created_at_step=3,
        source="test",
    )

    session_plan_to_policy(plan, policy)
    recovered = session_plan_from_policy(policy)

    assert recovered is not None
    assert recovered.strategy == "learning_path"
    assert recovered.plan_id == "test-plan"
    assert [e.concept_id for e in recovered.entries] == ["c1", "c2"]


def test_session_plan_handles_legacy_shape() -> None:
    policy = TutorSessionPolicy()
    policy.session_plan = {
        "strategy": "learning_path",
        "concept_plan": ["c1", "c2"],
    }

    plan = session_plan_from_policy(policy)

    assert plan is not None
    assert plan.strategy == "learning_path"
    assert [e.concept_id for e in plan.entries] == ["c1", "c2"]


def test_concept_plan_roundtrip_through_policy() -> None:
    policy = TutorSessionPolicy()

    steps = [
        ConceptPlanStep(
            step_id="s1",
            step_type="EXPLAIN",
            instruction="Explain the basic idea",
            expected_duration_steps=1,
        ),
        ConceptPlanStep(
            step_id="s2",
            step_type="QUIZ_MCQ",
            instruction="Ask a quick quiz",
            expected_duration_steps=1,
        ),
    ]

    plan = ConceptPlan(
        plan_id="cp-1",
        concept_id="heat_transfer_1",
        steps=steps,
        created_at_step=0,
        source="test",
        initial_mastery=0.2,
        target_mastery=0.8,
    )

    concept_plan_to_policy(plan, policy)
    recovered = concept_plan_from_policy(policy)

    assert recovered is not None
    assert recovered.plan_id == "cp-1"
    assert recovered.concept_id == "heat_transfer_1"
    assert [s.step_id for s in recovered.steps] == ["s1", "s2"]
