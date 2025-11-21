"""
Clean 3-layer MDP environment for tutor agent.

This package provides a hierarchical MDP architecture:
- SessionEnvironment: Manages overall study session and concept sequencing
- ConceptEnvironment: Handles learning for a single concept
- TutorEnvironment: Executes individual pedagogical actions

Each environment layer is responsible for its own state management,
action space, and transitions.
"""

from .base import BaseEnvironment, EnvironmentState, EnvironmentTransition
from .session_env import SessionEnvironment, SessionState, SessionAction
from .concept_env import ConceptEnvironment, ConceptState, ConceptAction
from .tutor_env import TutorEnvironment, TutorState, TutorAction
from .context import SessionContext, ConceptContext, TutorContext
from .policies import (
    SimpleSessionPolicy,
    SimpleConceptPolicy,
    SimpleTutorPolicy,
    make_session_policy,
    make_concept_policy,
    make_tutor_policy,
)
from .tools import EnvironmentStateManager, PlanCoordinator, ResponseBuilder
from .orchestrator import EnvironmentOrchestrator, run_environment_turn

__all__ = [
    # Base classes
    "BaseEnvironment",
    "EnvironmentState",
    "EnvironmentTransition",
    # Environments
    "SessionEnvironment",
    "SessionState",
    "SessionAction",
    "ConceptEnvironment",
    "ConceptState",
    "ConceptAction",
    "TutorEnvironment",
    "TutorState",
    "TutorAction",
    # Contexts
    "SessionContext",
    "ConceptContext",
    "TutorContext",
    # Policies
    "SimpleSessionPolicy",
    "SimpleConceptPolicy",
    "SimpleTutorPolicy",
    "make_session_policy",
    "make_concept_policy",
    "make_tutor_policy",
    # Tools
    "EnvironmentStateManager",
    "PlanCoordinator",
    "ResponseBuilder",
    # Orchestrator
    "EnvironmentOrchestrator",
    "run_environment_turn",
]
