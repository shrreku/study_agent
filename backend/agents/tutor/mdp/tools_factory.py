from __future__ import annotations

from typing import Any

from .tools import (
    SessionPlannerTool,
    ConceptPlannerTool,
    StepExecutorTool,
    MasteryEstimatorTool,
    QuizEvaluatorTool,
    PedagogicalResponseGeneratorTool,
)
from ..tools.session_planner import SessionPlannerLLM
from ..tools.concept_planner import ConceptPlannerLLM
from ..tools.step_executor import execute_concept_step
from ..tools.mastery_estimator import HeuristicMasteryEstimator
from ..tools.quiz_evaluator import BasicQuizEvaluator
from ..tools.pedagogical_response import DefaultPedagogicalResponseGeneratorTool


class _StepExecutorAdapter:
    """Adapter to expose execute_concept_step as a StepExecutorTool."""

    def __call__(
        self,
        *,
        session_id: str,
        user_id: str,
        concept_id: str,
        step,
        context_obs,
    ):
        return execute_concept_step(
            session_id=session_id,
            user_id=user_id,
            concept_id=concept_id,
            step=step,
            context_obs=context_obs,
        )


def make_session_planner_tool(config: Any = None) -> SessionPlannerTool:
    """Factory for SessionPlannerTool.

    For now this always returns SessionPlannerLLM with default settings.
    """

    return SessionPlannerLLM()


def make_concept_planner_tool(config: Any = None) -> ConceptPlannerTool:
    """Factory for ConceptPlannerTool.

    For now this always returns ConceptPlannerLLM with default settings.
    """

    return ConceptPlannerLLM()


def make_step_executor_tool(config: Any = None) -> StepExecutorTool:
    """Factory for StepExecutorTool.

    Wraps execute_concept_step in a callable matching the Protocol.
    """

    return _StepExecutorAdapter()


def make_mastery_estimator_tool(config: Any = None) -> MasteryEstimatorTool:
    """Factory for MasteryEstimatorTool.

    Uses HeuristicMasteryEstimator with default hyperparameters; these
    can be made configurable via TutorConfig in the future.
    """

    return HeuristicMasteryEstimator()


def make_quiz_evaluator_tool(config: Any = None) -> QuizEvaluatorTool:
    """Factory for QuizEvaluatorTool.

    Returns the basic, heuristic MCQ evaluator.
    """

    return BasicQuizEvaluator()


def make_pedagogical_response_tool(config: Any = None) -> PedagogicalResponseGeneratorTool:
    """Factory for PedagogicalResponseGeneratorTool.

    For now this always returns the default, lightweight implementation.
    """

    return DefaultPedagogicalResponseGeneratorTool(config=config)
