"""
Simple policies for the 3-layer MDP environment.

These are MVP policies that implement basic logic for each layer.
They can be replaced with learned policies later.
"""

from __future__ import annotations

from typing import Optional

from .session_env import SessionAction, SessionState
from .concept_env import ConceptAction, ConceptState
from .tutor_env import TutorAction, TutorState
from ..mdp.plans import ConceptPlanStep


class SimpleSessionPolicy:
    """Simple policy for session-level decisions.
    
    Follows a sequential strategy: study concepts in order until all complete.
    """
    
    def decide(
        self,
        state: SessionState,
        concept_complete: bool = False,
        **kwargs
    ) -> SessionAction:
        """Decide next session action.
        
        Args:
            state: Current session state
            concept_complete: Whether current concept just finished
            **kwargs: Additional context
            
        Returns:
            Session action to take
        """
        # Check if session should end
        if state.is_session_complete():
            return SessionAction.END_SESSION
        
        # If no plan yet, this shouldn't happen but handle it
        if state.session_plan is None or not state.session_plan.entries:
            return SessionAction.END_SESSION
        
        # If concept just completed, advance
        if concept_complete:
            return SessionAction.ADVANCE_CONCEPT
        
        # If at start or after advancing, start new concept
        current_concept = state.get_current_concept_id()
        if current_concept:
            return SessionAction.START_CONCEPT
        
        return SessionAction.END_SESSION


class SimpleConceptPolicy:
    """Simple policy for concept-level decisions.
    
    Executes plan steps sequentially, requests replanning when needed.
    """
    
    def decide(
        self,
        state: ConceptState,
        step_complete: bool = False,
        **kwargs
    ) -> ConceptAction:
        """Decide next concept action.
        
        Args:
            state: Current concept state
            step_complete: Whether current step just finished
            **kwargs: Additional context
            
        Returns:
            Concept action to take
        """
        # Check if we need a plan
        if state.concept_plan is None or not state.concept_plan.steps:
            return ConceptAction.GENERATE_PLAN
        
        # If step just completed, advance
        if step_complete:
            return ConceptAction.ADVANCE_STEP
        
        # If plan exhausted, check mastery
        if state.is_plan_complete():
            if state.is_mastery_reached():
                return ConceptAction.COMPLETE_CONCEPT
            else:
                # Need more practice - replan
                return ConceptAction.REPLAN
        
        # Execute current step
        return ConceptAction.EXECUTE_STEP


class SimpleTutorPolicy:
    """Simple policy for tutor-level pedagogical actions.
    
    Maps plan step types to pedagogical actions.
    """
    
    def decide(
        self,
        state: TutorState,
        current_step: Optional[ConceptPlanStep] = None,
        **kwargs
    ) -> TutorAction:
        """Decide next tutor action.
        
        Args:
            state: Current tutor state
            current_step: Current plan step to execute
            **kwargs: Additional context
            
        Returns:
            Tutor action to take
        """
        # If no step, this is a transition
        if current_step is None:
            return TutorAction.TRANSITION
        
        # Map step type to pedagogical action
        step_type = current_step.step_type.lower()
        
        if "explain" in step_type or "introduce" in step_type:
            return TutorAction.EXPLAIN
        elif "question" in step_type or "ask" in step_type:
            return TutorAction.ASK_QUESTION
        elif "example" in step_type or "demonstrate" in step_type:
            return TutorAction.WORKED_EXAMPLE
        elif "practice" in step_type or "exercise" in step_type:
            return TutorAction.GUIDED_PRACTICE
        elif "summary" in step_type or "recap" in step_type:
            return TutorAction.SUMMARY
        elif "quiz" in step_type or "assess" in step_type:
            return TutorAction.QUIZ
        else:
            # Default to explanation
            return TutorAction.EXPLAIN


def make_session_policy() -> SimpleSessionPolicy:
    """Factory for session policy.
    
    Returns:
        Session policy instance
    """
    return SimpleSessionPolicy()


def make_concept_policy() -> SimpleConceptPolicy:
    """Factory for concept policy.
    
    Returns:
        Concept policy instance
    """
    return SimpleConceptPolicy()


def make_tutor_policy() -> SimpleTutorPolicy:
    """Factory for tutor policy.
    
    Returns:
        Tutor policy instance
    """
    return SimpleTutorPolicy()
