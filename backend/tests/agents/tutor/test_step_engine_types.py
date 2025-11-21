import types

from backend.agents.tutor.decision_engine import ActionDecision
from backend.agents.tutor.state_machine import TutorState
from backend.agents.tutor.step_engine import (
    StudyPlan,
    StudyStep,
    build_study_plan_from_policy_state,
    build_study_step_from_action_decision,
    build_study_step_id,
    infer_expected_input,
    map_action_to_step_type,
    tutor_state_to_phase,
    validate_study_plan,
)


def test_tutor_state_to_phase_mapping_basic() -> None:
    assert tutor_state_to_phase(TutorState.ORIENTATION) == "orientation"
    assert tutor_state_to_phase(TutorState.TEACHING) == "teaching"
    assert tutor_state_to_phase(TutorState.ASSESSMENT) == "assessment"
    assert tutor_state_to_phase(TutorState.REVIEW) == "review"
    assert tutor_state_to_phase(TutorState.CLOSURE) == "closure"


def test_build_study_step_id_composition() -> None:
    assert build_study_step_id() == "step"
    assert build_study_step_id(session_id="s1") == "step-s1"
    assert build_study_step_id(session_id="s1", turn_index=3) == "step-s1-3"
    assert (
        build_study_step_id(session_id="s1", turn_index=3, plan_index=1)
        == "step-s1-3-1"
    )


def test_map_action_to_step_type_normalization() -> None:
    assert map_action_to_step_type("explain") == "explain"
    assert map_action_to_step_type("ask") == "ask_open"
    assert map_action_to_step_type("mcq") == "ask_mcq"
    assert map_action_to_step_type("reflect") == "reflect"
    assert map_action_to_step_type("review") == "review"
    assert map_action_to_step_type("worked_example") == "worked_example"
    assert map_action_to_step_type("preview") == "wrap_up"
    # Fallback
    assert map_action_to_step_type("unknown_action") == "explain"


def test_infer_expected_input() -> None:
    assert infer_expected_input(action="explain", is_mcq=True) == "mcq"
    assert infer_expected_input(action="ask") == "free_text"
    assert infer_expected_input(action="reflect") == "free_text"
    assert infer_expected_input(action="explain", expects_answer=True) == "free_text"
    assert infer_expected_input(action="explain") == "none"


def test_build_study_step_from_action_decision_basic() -> None:
    decision = ActionDecision(
        action="explain",
        rationale="test rationale",
        retrieval_query="convection",
        grounding_mode="llm_integrated",
        confidence=0.9,
        pedagogy_focus=["definition", "example"],
    )

    step = build_study_step_from_action_decision(
        decision=decision,
        state=TutorState.TEACHING,
        focus_concept="convection",
        session_id="sess-1",
        turn_index=5,
        plan_index=0,
        is_mcq=False,
        expects_answer=False,
    )

    assert isinstance(step, StudyStep)
    assert step.type == "explain"
    assert step.concept == "convection"
    assert step.phase == "teaching"
    assert step.expected_input == "none"
    assert step.pedagogy_focus == ["definition", "example"]
    assert step.id == "step-sess-1-5-0"
    assert step.meta["rationale"] == "test rationale"
    assert step.meta["retrieval_query"] == "convection"


def test_build_study_plan_from_policy_state_happy_path() -> None:
    policy_state = types.SimpleNamespace(
        srl_plan={
            "id": "plan-123",
            "concept": "convection",
            "high_level_summary": "summary",
            "steps": [
                {
                    "action": "explain",
                    "reasoning": "step 1",
                    "pedagogy_focus": ["definition"],
                    "expects_answer": False,
                },
                {
                    "action": "ask",
                    "reasoning": "step 2",
                    "pedagogy_focus": ["concept_check"],
                    "expects_answer": True,
                },
            ],
        },
        srl_plan_step_index=1,
    )

    plan = build_study_plan_from_policy_state(
        policy_state=policy_state,
        focus_concept="convection",
        session_id="sess-1",
    )

    assert isinstance(plan, StudyPlan)
    assert plan.id == "plan-123"
    assert plan.concept == "convection"
    assert plan.index == 1
    assert len(plan.steps) == 2

    first, second = plan.steps
    assert first.type == "explain"
    assert first.expected_input == "none"
    assert second.type == "ask_open"
    assert second.expected_input == "free_text"


def test_validate_study_plan_invariants() -> None:
    # Valid plan
    valid_plan = StudyPlan(
        id="plan-1",
        concept="convection",
        steps=[
            StudyStep(
                id="step-1",
                type="explain",
                concept="convection",
                phase="teaching",
                expected_input="none",
            )
        ],
        index=0,
        high_level_summary=None,
    )
    validate_study_plan(valid_plan)

    # Invalid index negative
    invalid_plan_neg = StudyPlan(
        id="plan-2",
        concept="convection",
        steps=valid_plan.steps,
        index=-1,
        high_level_summary=None,
    )
    try:
        validate_study_plan(invalid_plan_neg)
    except ValueError as exc:
        assert "cannot be negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative index")

    # Invalid index beyond steps
    invalid_plan_large = StudyPlan(
        id="plan-3",
        concept="convection",
        steps=valid_plan.steps,
        index=10,
        high_level_summary=None,
    )
    try:
        validate_study_plan(invalid_plan_large)
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("Expected ValueError for index out of range")
