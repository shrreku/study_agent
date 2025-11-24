from typing import Any, Optional

from .orchestrator import TutorOrchestrator
from .mdp.policy import (
    make_session_policy,
    make_concept_policy,
    make_pedagogical_tutor_policy,
)
from .tools.session_planner import SessionPlannerLLM
from .tools.concept_planner import ConceptPlannerLLM
from .tools.pedagogical_response import DefaultPedagogicalResponseGeneratorTool
from .persistence import TutorStateManager

def make_orchestrator(config: Any = None, cur: Any = None) -> TutorOrchestrator:
    """
    Factory function to create a configured TutorOrchestrator.
    
    Args:
        config: Configuration object (optional)
        cur: Database cursor (optional, but recommended for real usage)
    """
    
    # Policies
    session_policy = make_session_policy(config)
    concept_policy = make_concept_policy(config)
    pedagogical_policy = make_pedagogical_tutor_policy(config)
    
    # Tools
    # In a real app, we inject clients/drivers here.
    # The planners use the 'llm' module internally, so no extra injection needed there yet.
    session_planner = SessionPlannerLLM()
    concept_planner = ConceptPlannerLLM()
    response_generator = DefaultPedagogicalResponseGeneratorTool(config)
    
    # Persistence (not directly used by orchestrator, but useful to have initialized if we change design)
    # For now, the orchestrator doesn't take the state manager directly, 
    # but the caller (run_environment_turn) will use it.
    
    return TutorOrchestrator(
        session_policy=session_policy,
        concept_policy=concept_policy,
        pedagogical_policy=pedagogical_policy,
        session_planner=session_planner,
        concept_planner=concept_planner,
        response_generator=response_generator,
    )

def make_state_manager(cur: Any) -> TutorStateManager:
    """
    Factory for the state manager.
    """
    return TutorStateManager(cur)
