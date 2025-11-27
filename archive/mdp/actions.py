from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class ConceptMDPAction(str, Enum):
    """Canonical action space for the concept-level MDP.

    These are macro-level tutor decisions for a single concept episode.
    """

    FOLLOW_PLAN_STEP = "FOLLOW_PLAN_STEP"
    JUMP_TO_ASSESSMENT = "JUMP_TO_ASSESSMENT"
    ADVANCE_CONCEPT = "ADVANCE_CONCEPT"
    TERMINATE_CONCEPT = "TERMINATE_CONCEPT"
    REPLAN_CONCEPT = "REPLAN_CONCEPT"
    LOCAL_OVERRIDE_STEP = "LOCAL_OVERRIDE_STEP"
    NOP = "NOP"  # Explicit no-op / logging-only step


def map_step_controls_to_concept_action(
    *,
    step_control_obj: Any,
    requested_override_type: Optional[str],
    control_type_label: Optional[str],
) -> Optional[ConceptMDPAction]:
    """Map step-by-step control surface + overrides into a ConceptMDPAction.

    This helper centralizes the semantics that are currently in the
    step-by-step orchestrator. For Phase 1 it is not yet wired into the
    runtime; it will be used in a later refactor.
    """

    # Highest-fidelity signal: normalized StepControl object (if present).
    if step_control_obj is not None:
        ctype = getattr(step_control_obj, "type", None)
        if ctype == "continue":
            return ConceptMDPAction.FOLLOW_PLAN_STEP
        if ctype == "skip_to_quiz":
            return ConceptMDPAction.JUMP_TO_ASSESSMENT
        if ctype == "skip_to_next_concept":
            return ConceptMDPAction.ADVANCE_CONCEPT
        if ctype == "end_session":
            return ConceptMDPAction.TERMINATE_CONCEPT

    # Next: explicit override types from payload.
    if requested_override_type:
        t = requested_override_type.strip().lower()
        if t == "step_skip_to_assessment":
            return ConceptMDPAction.JUMP_TO_ASSESSMENT
        if t == "step_next_concept":
            return ConceptMDPAction.ADVANCE_CONCEPT
        if t == "session_end":
            return ConceptMDPAction.TERMINATE_CONCEPT

    # Finally: fall back to raw control labels inferred in the orchestrator
    # (confirmed_action, step_control_type, etc.).
    label = (control_type_label or "").strip().lower()
    if label in {"continue", "next", "yes"}:
        return ConceptMDPAction.FOLLOW_PLAN_STEP
    if label == "skip_to_quiz":
        return ConceptMDPAction.JUMP_TO_ASSESSMENT
    if label in {"skip_to_next_concept", "skip_to_next"}:
        return ConceptMDPAction.ADVANCE_CONCEPT

    return None
