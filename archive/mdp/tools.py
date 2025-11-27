from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

from .plans import SessionPlan, ConceptPlan, ConceptPlanStep
from .pedagogical_tutor import PedagogicalTutorState, PedagogicalTutorAction


class SessionPlannerTool(Protocol):
    """Build a SessionPlan for a tutoring session.

    Implementations may call LLMs or heuristics to choose concepts and
    target masteries based on strategy and mastery information.
    """

    def __call__(
        self,
        *,
        user_id: str,
        session_id: str,
        strategy: str,
        target_concepts: List[str],
        mastery_map: Dict[str, Dict[str, Any]],
    ) -> SessionPlan:
        ...


class ConceptPlannerTool(Protocol):
    """Build a ConceptPlan (SRL-style plan) for a single concept episode.

    Implementations typically wrap SRL planning prompts and use TutorContext-
    like observations for history, mastery, and retrieval context.
    """

    def __call__(
        self,
        *,
        user_id: str,
        session_id: str,
        concept_id: str,
        target_mastery: Optional[float],
        context_obs: Dict[str, Any],
    ) -> ConceptPlan:
        ...


class StepExecutorTool(Protocol):
    """Execute a single ConceptPlanStep into tutor messages and UI payload.

    The returned dictionary is expected to be JSON-safe and may contain:

    - "messages": List[Dict[str, Any]] – tutor messages for the frontend.
    - "ui_mode": str – e.g. "buttons_only", "mcq", "free_text".
    - "mcq_payload": Optional[Dict[str, Any]] – if ui_mode == "mcq".
    - "debug": Dict[str, Any] – step metadata for logs.
    """

    def __call__(
        self,
        *,
        session_id: str,
        user_id: str,
        concept_id: str,
        step: ConceptPlanStep,
        context_obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...


class MasteryEstimatorTool(Protocol):
    """Estimate concept mastery change based on recent interactions.

    Returns (post_mastery, mastery_delta). Either value may be None if the
    estimator cannot produce a signal.
    """

    def __call__(
        self,
        *,
        concept_id: str,
        mastery_before: Optional[float],
        recent_interactions: List[Dict[str, Any]],
    ) -> Tuple[Optional[float], Optional[float]]:
        ...


class QuizEvaluatorTool(Protocol):
    """Evaluate a single MCQ response.

    Returns (mcq_outcome, quiz_delta) where mcq_outcome is a JSON-safe dict
    that should at least include an "answer_correct" boolean.
    """

    def __call__(
        self,
        *,
        question: Dict[str, Any],
        user_answer: Any,
        correct_answer: Any,
    ) -> Tuple[Dict[str, Any], Optional[float]]:
        ...


class PedagogicalResponseGeneratorTool(Protocol):
    """Generate tutor messages and UI payload for a pedagogical tutor action.

    Implementations may call LLMs or heuristics but must return a JSON-safe
    dictionary compatible with the main tutor runtime.
    """

    def __call__(
        self,
        *,
        user_id: str,
        session_id: str,
        concept_id: str,
        pedagogical_action: PedagogicalTutorAction,
        ped_state: PedagogicalTutorState,
        concept_plan: Optional[ConceptPlan],
        context_obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...
