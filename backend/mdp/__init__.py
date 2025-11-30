"""
MDP Module for Intelligent Tutoring System

3-Layer Architecture:
- Level 1: Session Planning (which concepts to cover)
- Level 2: Concept Planning (steps within a concept)
- Level 3: Tutor MDP (pedagogical decisions) <- TRAINABLE

Modules:
- schemas_v2: MDP state, action, reward, transition definitions
- policies_v2: Rule-based and LLM-based tutor policies
- engine_v2: Orchestrator with transition logging
- trajectory: Logging utilities for PPO training
- reward: Reward computation module
"""

import logging

logger = logging.getLogger(__name__)

# Import new V2 modules (core - always available)
from mdp.schemas_v2 import (
    # Enums
    PedagogicalAction,
    StudentIntent,
    CorrectnessLevel,
    # Core data classes
    TutorState,
    TutorObservation,
    TutorAction,
    TutorReward,
    TutorTransition,
    StudentAnalysis,
    ConversationTurn,
    # Plan structures
    StudentProfile,
    PlanStep,
    Plan,
    SessionStep,
    SessionPlan,
    # Interfaces
    ConceptPolicy,
    TutorPolicy,
)

from mdp.policies_v2 import (
    RuleBasedTutorPolicy,
    LLMTutorPolicy,
    HybridTutorPolicy,
    ConversationalTutorPolicy,  # Legacy compatibility
    SimpleTutorPolicy,  # Legacy compatibility
)

from mdp.trajectory import (
    TrajectoryLogger,
    TrajectoryDataset,
    TrajectoryStats,
)

# Optional imports that require external dependencies
TutorOrchestrator = None
LegacyOrchestrator = None

try:
    from mdp.engine_v2 import TutorOrchestrator
except ImportError as e:
    logger.debug(f"engine_v2 import skipped: {e}")

try:
    from mdp.engine import Orchestrator as LegacyOrchestrator
except ImportError as e:
    logger.debug(f"Legacy engine import skipped: {e}")

# Legacy schema imports
try:
    from mdp.schemas import (
        StudentProfile as LegacyStudentProfile,
        PlanStep as LegacyPlanStep,
        Plan as LegacyPlan,
    )
except ImportError:
    LegacyStudentProfile = StudentProfile
    LegacyPlanStep = PlanStep
    LegacyPlan = Plan

try:
    from mdp.policies import (
        SimpleTutorPolicy as LegacySimpleTutorPolicy,
        ConversationalTutorPolicy as LegacyConversationalTutorPolicy,
    )
except ImportError:
    LegacySimpleTutorPolicy = SimpleTutorPolicy
    LegacyConversationalTutorPolicy = ConversationalTutorPolicy

__all__ = [
    # V2 Core
    "PedagogicalAction",
    "StudentIntent", 
    "CorrectnessLevel",
    "TutorState",
    "TutorObservation",
    "TutorAction",
    "TutorReward",
    "TutorTransition",
    "StudentAnalysis",
    "ConversationTurn",
    # V2 Plans
    "StudentProfile",
    "PlanStep",
    "Plan",
    "SessionStep",
    "SessionPlan",
    # V2 Policies
    "RuleBasedTutorPolicy",
    "LLMTutorPolicy", 
    "HybridTutorPolicy",
    "ConversationalTutorPolicy",
    "SimpleTutorPolicy",
    # V2 Engine
    "TutorOrchestrator",
    # V2 Trajectory
    "TrajectoryLogger",
    "TrajectoryDataset",
    "TrajectoryStats",
    # Interfaces
    "ConceptPolicy",
    "TutorPolicy",
    # Legacy
    "LegacyOrchestrator",
]
