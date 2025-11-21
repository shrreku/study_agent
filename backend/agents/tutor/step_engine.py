"""Step Engine types and interface for tutor V3.

This module defines internal dataclasses and a StepEngine interface that
produce a StudyStep / StudyPlan view over the existing unified tutor V2
runtime. Behaviour wiring is added in a later ticket; for now this module
is purely additive and safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from .state_machine import TutorState
from .config import get_tutor_config
from .decision_engine import TutorDecisionEngine, ActionDecision
from .runtime.mode_step_srl import StepByStepSRLRuntime
from .runtime.mode_auto_conversational import AutoConversationalRuntime

if TYPE_CHECKING:
    # Imported only for type checking to avoid runtime circular imports.
    from .context_model import TutorContext
    from .state_machine import TutorStateManager
    from .state import TutorSessionPolicy


Phase = Literal["orientation", "teaching", "assessment", "review", "closure"]
ControlMode = Literal["auto", "step"]


@dataclass
class StepControl:
    """Represents a control action issued by the frontend or chosen internally.

    This is a normalized view over confirmed_action, step_control,
    and action_override in the existing payloads.
    """

    type: Literal[
        "continue",
        "skip_to_quiz",
        "skip_to_next_concept",
        "end_session",
        "replan",
        "set_action",
    ]
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StudyStep:
    """Single pedagogic step decided by the tutor.

    This acts as a stable abstraction over ActionDecision and related state.
    """

    id: str
    type: Literal[
        "explain",
        "ask_open",
        "ask_mcq",
        "reflect",
        "review",
        "worked_example",
        "plan_overview",
        "quiz_summary",
        "wrap_up",
    ]
    concept: Optional[str]
    phase: Phase
    expected_input: Literal["none", "free_text", "mcq", "button_only"]
    pedagogy_focus: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StudyPlan:
    """Plan of multiple StudyStep items for a concept or small cluster.

    Backed initially by SRL planning state from policy_state / TutorPlanner.
    """

    id: str
    concept: Optional[str]
    steps: List[StudyStep]
    index: int
    high_level_summary: Optional[str] = None


@dataclass
class StepEngineResult:
    """Result returned by StepEngine for a single turn."""

    action_decision: ActionDecision
    step: StudyStep
    plan: Optional[StudyPlan]


class StepEngine:
    """Decides the next pedagogic step for the tutor.

    For now this is a thin adapter over the existing V2 decision engine
    and SRL planning. Behaviour wiring is added in
    TUTOR-ARCH-V3-02-RUNTIME_INTEGRATION_STEP_ENGINE.
    """

    def decide_step(
        self,
        *,
        tutor_context: "TutorContext",  # from context_model.py
        state_manager: "TutorStateManager",  # from state_machine.py
        control_mode: ControlMode,
        control: Optional[StepControl],
        policy_state: "TutorSessionPolicy",
        override_type: Optional[str] = None,
        override_params: Optional[Dict[str, Any]] = None,
        mode_label: Optional[str] = None,
    ) -> StepEngineResult:
        """Decide the next pedagogic step.

        This is a thin adapter over the existing V2 decision engine and
        mode-specific runtimes. It is intentionally conservative and
        preserves all existing behaviour by delegating to the same
        runtimes that the orchestrator used previously.
        """
        # Load config so we respect the same feature flags as the
        # orchestrator, but allow the effective mode label to be driven
        # per-turn (e.g., from agent_action_mode) instead of only from
        # environment.
        config = get_tutor_config()

        effective_mode = (mode_label or config.mode or "").strip().lower()
        if effective_mode not in {"simple", "intelligent", "step_by_step", "debug"}:
            effective_mode = (config.mode or "intelligent").strip().lower()

        decision_engine = TutorDecisionEngine(
            mode=effective_mode,
            enable_llm_policy=config.enable_llm_policy,
            enable_srl_planning=config.enable_srl_planning,
        )

        # Delegate to the same runtimes as before, but branch primarily on
        # the explicit control_mode ("step" vs "auto"). This keeps behaviour
        # equivalent for current configurations where TUTOR_MODE and
        # agent_action_mode are aligned, while making the ControlMode
        # contract explicit for V3-04.
        if control_mode == "step":
            runtime = StepByStepSRLRuntime.from_env(
                mode=effective_mode,
                decision_engine=decision_engine,
            )
            action_decision = runtime.decide_action(
                tutor_context,
                state_manager,
                policy_state,
                override_type=override_type,
                override_params=override_params,
            )
        else:
            auto_runtime = AutoConversationalRuntime(
                mode=effective_mode,
                decision_engine=decision_engine,
            )
            action_decision = auto_runtime.decide_action(
                tutor_context,
                state_manager,
            )

        # Build a lightweight StudyStep view over the ActionDecision using
        # existing helpers. At this stage we do not yet know whether the
        # final execution will surface an MCQ, so we conservatively treat
        # this as non-MCQ and infer expected_input from the action and
        # intent signals.
        focus_concept = (
            getattr(tutor_context, "focus_concept", None)
            or getattr(tutor_context, "inferred_concept", None)
        )
        expects_answer = bool(
            (action_decision.action or "").strip().lower() in {"ask"}
            or getattr(tutor_context, "intent", None) in {"answer", "reflection", "question"}
        )

        step = build_study_step_from_action_decision(
            decision=action_decision,
            state=state_manager.current_state,
            focus_concept=focus_concept,
            session_id=getattr(tutor_context, "session_id", None),
            turn_index=getattr(tutor_context, "turn_index", None),
            plan_index=getattr(policy_state, "srl_plan_step_index", None),
            is_mcq=False,
            expects_answer=expects_answer,
        )

        # StudyPlan projection is still built from policy_state by the
        # orchestrator for now; StepEngine does not mutate policy_state and
        # therefore returns no plan here.
        return StepEngineResult(
            action_decision=action_decision,
            step=step,
            plan=None,
        )


# --- Mapping helpers -------------------------------------------------------


def tutor_state_to_phase(state: TutorState) -> Phase:
    """Map TutorState enum to Phase literal.

    This keeps the representation stable even if TutorState acquires
    additional members in the future.
    """

    value = state.value
    if value in {"orientation", "teaching", "assessment", "review", "closure"}:
        return value  # type: ignore[return-value]
    # Fallback: treat unknown states as teaching.
    return "teaching"


def build_study_step_id(
    *,
    session_id: Optional[str] = None,
    turn_index: Optional[int] = None,
    plan_index: Optional[int] = None,
) -> str:
    """Best-effort ID builder for StudyStep.

    Does not introduce new randomness; uses existing identifiers where
    available so that tests can assert deterministic IDs.
    """

    parts: List[str] = ["step"]
    if session_id:
        parts.append(str(session_id))
    if turn_index is not None:
        parts.append(str(turn_index))
    if plan_index is not None:
        parts.append(str(plan_index))
    return "-".join(parts)


def map_action_to_step_type(action: str) -> str:
    """Normalize ActionDecision.action into StudyStep.type.

    This is kept liberal for now; later tickets can tighten the enums.
    """

    a = (action or "").strip().lower()
    if a in {"ask", "question"}:
        return "ask_open"
    if a in {"mcq", "ask_mcq"}:
        return "ask_mcq"
    if a in {"explain", "orientation_explain"}:
        return "explain"
    if a in {"reflect", "reflection"}:
        return "reflect"
    if a in {"review"}:
        return "review"
    if a in {"worked_example", "example"}:
        return "worked_example"
    if a in {"preview", "closure", "wrap_up"}:
        return "wrap_up"
    # Fallback
    return "explain"


def infer_expected_input(
    *,
    action: str,
    is_mcq: bool = False,
    expects_answer: bool = False,
) -> str:
    """Infer expected_input for StudyStep from simple signals."""

    if is_mcq:
        return "mcq"
    a = (action or "").strip().lower()
    if a in {"ask", "question"}:
        return "free_text"
    if a in {"reflect", "reflection"}:
        return "free_text"
    if expects_answer:
        return "free_text"
    return "none"


def validate_study_plan(plan: StudyPlan) -> None:
    """Basic invariant checks for StudyPlan.

    Raises ValueError if invariants are violated.
    """

    if plan.index < 0:
        raise ValueError("plan.index cannot be negative")
    if plan.index >= len(plan.steps) and plan.steps:
        raise ValueError("plan.index out of range for steps list")


def build_study_step_from_action_decision(
    *,
    decision: "ActionDecision",
    state: TutorState,
    focus_concept: Optional[str] = None,
    session_id: Optional[str] = None,
    turn_index: Optional[int] = None,
    plan_index: Optional[int] = None,
    is_mcq: bool = False,
    expects_answer: bool = False,
) -> StudyStep:
    """Construct a StudyStep view over an ActionDecision.

    This helper is pure and side-effect free; it does not mutate any
    policy state or context objects.
    """

    step_type = map_action_to_step_type(decision.action)
    phase = tutor_state_to_phase(state)
    expected_input = infer_expected_input(
        action=decision.action,
        is_mcq=is_mcq,
        expects_answer=expects_answer,
    )

    step_id = build_study_step_id(
        session_id=session_id,
        turn_index=turn_index,
        plan_index=plan_index,
    )

    pedagogy: List[str] = []
    if decision.pedagogy_focus:
        try:
            pedagogy = list(decision.pedagogy_focus)
        except Exception:
            pedagogy = []

    meta: Dict[str, Any] = {
        "rationale": decision.rationale,
        "grounding_mode": decision.grounding_mode,
        "confidence": decision.confidence,
        "retrieval_query": decision.retrieval_query,
        "cold_start": decision.cold_start,
        "retrieval_strategy": decision.retrieval_strategy,
        "max_chunks": decision.max_chunks,
    }

    return StudyStep(
        id=step_id,
        type=step_type,  # type: ignore[arg-type]
        concept=focus_concept,
        phase=phase,
        expected_input=expected_input,  # type: ignore[arg-type]
        pedagogy_focus=pedagogy,
        meta=meta,
    )


def build_study_plan_from_policy_state(
    *,
    policy_state: Any,
    focus_concept: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[StudyPlan]:
    """Best-effort StudyPlan builder from SRL plan policy state.

    Expects an object with ``srl_plan`` and ``srl_plan_step_index``
    attributes as produced by the current tutor policy state.
    Returns None if no structured plan is present.
    """

    try:
        plan_data = getattr(policy_state, "srl_plan", None)
    except Exception:
        plan_data = None

    if not isinstance(plan_data, dict):
        return None

    raw_steps = plan_data.get("steps") or []
    if not isinstance(raw_steps, list) or not raw_steps:
        return None

    try:
        raw_index = getattr(policy_state, "srl_plan_step_index", 0)
    except Exception:
        raw_index = 0

    try:
        index = int(raw_index or 0)
    except Exception:
        index = 0

    steps: List[StudyStep] = []
    for i, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            continue

        action = (raw.get("action") or "explain")
        step_type = map_action_to_step_type(action)

        expects_answer = bool(raw.get("expects_answer"))
        expected_input = infer_expected_input(
            action=action,
            is_mcq=False,
            expects_answer=expects_answer,
        )

        pedagogy_raw = raw.get("pedagogy_focus")
        pedagogy: List[str] = []
        if isinstance(pedagogy_raw, list):
            try:
                pedagogy = [str(x) for x in pedagogy_raw if x]
            except Exception:
                pedagogy = []

        # Copy-through any remaining keys as meta, excluding fields we have
        # explicitly modeled above.
        meta: Dict[str, Any] = {}
        for key, value in raw.items():
            if key in {"action", "pedagogy_focus", "expects_answer", "concept"}:
                continue
            meta[key] = value

        step_concept = focus_concept or raw.get("concept")
        step_id = build_study_step_id(
            session_id=session_id,
            plan_index=i,
        )

        steps.append(
            StudyStep(
                id=step_id,
                type=step_type,  # type: ignore[arg-type]
                concept=step_concept,
                phase="teaching",  # type: ignore[assignment]
                expected_input=expected_input,  # type: ignore[arg-type]
                pedagogy_focus=pedagogy,
                meta=meta,
            )
        )

    plan_id = plan_data.get("id") or f"plan-{session_id or 'unknown'}"
    concept = focus_concept or plan_data.get("concept")
    high_level_summary = plan_data.get("high_level_summary")

    plan = StudyPlan(
        id=str(plan_id),
        concept=concept,
        steps=steps,
        index=index,
        high_level_summary=high_level_summary,
    )

    # Validate basic invariants; allow empty plans only if index is zero.
    if steps:
        validate_study_plan(plan)
    else:
        if index != 0:
            raise ValueError("empty study plan must have index 0")

    return plan


def serialize_step(step: StudyStep) -> Dict[str, Any]:
    """Serialize StudyStep into a JSON-safe dict for responses/logs."""

    return {
        "id": step.id,
        "type": step.type,
        "concept": step.concept,
        "phase": step.phase,
        "expected_input": step.expected_input,
        "pedagogy_focus": list(step.pedagogy_focus or []),
        "meta": dict(step.meta or {}),
    }


def serialize_plan(plan: StudyPlan) -> Dict[str, Any]:
    """Serialize StudyPlan into a JSON-safe dict for responses/logs."""

    return {
        "id": plan.id,
        "concept": plan.concept,
        "index": plan.index,
        "high_level_summary": plan.high_level_summary,
        "steps": [serialize_step(s) for s in plan.steps],
    }


