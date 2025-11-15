from __future__ import annotations

import os
import sys

# ensure backend package root is on path when running tests directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.tutor.planning import TutorPlanner, TutorPlan  # type: ignore
from agents.tutor.responses import generate_explain_response_with_plan  # type: ignore
from agents.tutor.self_critique import SelfCritic  # type: ignore


def _sample_observation():
    return {
        "user": {"message": "Please explain conduction."},
        "classifier": {"intent": "question", "affect": "confused"},
        "tutor": {"focus_concept": "Conduction", "concept_level": "beginner"},
        "policy": {"version": 1},
    }


def _student_state():
    return {
        "mastery_map": {"Conduction": {"mastery": 0.3}},
        "learning_path": ["Conduction", "Convection"],
    }


def _dummy_chunks():
    return [
        {
            "id": "chunk-1",
            "pedagogy_role": "definition",
            "snippet": "Conduction moves heat through direct contact.",
        }
    ]


def test_planner_produces_plan_in_mock_mode(monkeypatch):
    monkeypatch.setenv("USE_LLM_MOCK", "1")
    planner = TutorPlanner(enable_srl_mode=True)
    plan = planner.generate_plan(
        observation=_sample_observation(),
        student_state=_student_state(),
        available_actions=["explain", "ask", "hint", "reflect", "review"],
    )
    assert isinstance(plan, TutorPlan)
    assert plan.intended_action in {"explain", "ask", "hint", "reflect", "review"}
    assert isinstance(plan.thinking, str)
    assert plan.confidence >= 0.0


def test_generate_explain_response_with_plan(monkeypatch):
    monkeypatch.setenv("USE_LLM_MOCK", "1")
    plan = TutorPlan(
        thinking="<think>We should give a simple explanation.</think>",
        intended_action="explain",
        action_rationale="Student asked a question",
        retrieval_query="Conduction",
        pedagogy_focus=["definition", "explanation"],
        difficulty_cap="intermediate",
        confidence=0.6,
        assumptions=["basic understanding"],
        risks=["too advanced"],
    )
    response, conf, src_ids, inferred = generate_explain_response_with_plan(
        plan=plan,
        concept="Conduction",
        level="beginner",
        chunks=_dummy_chunks(),
    )
    assert isinstance(response, str) and len(response) > 0
    assert isinstance(conf, float) and 0.0 <= conf <= 1.0
    assert src_ids and src_ids[0] == "chunk-1"
    assert inferred == "Conduction"


def test_self_critique_defaults(monkeypatch):
    monkeypatch.setenv("USE_LLM_MOCK", "1")
    critic = SelfCritic()
    plan = TutorPlan(
        thinking="<think>Plan to ask a question.</think>",
        intended_action="ask",
        action_rationale="Check understanding",
        retrieval_query="Conduction",
        pedagogy_focus=["example"],
        difficulty_cap="introductory",
        confidence=0.5,
        assumptions=[],
        risks=[],
    )
    result = critic.critique_response(plan, "What is conduction?", {"tutor": {"concept_level": "beginner"}})
    assert result.overall_quality >= 0.0
    assert isinstance(result.issues_found, list)
    assert isinstance(result.suggestions, list)
    assert result.should_revise in {True, False}
