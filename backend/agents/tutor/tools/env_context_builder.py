"""
Context builders for the environment architecture.

Provides utilities to build lightweight context objects from various sources.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..environment.context import SessionContext, ConceptContext, TutorContext
from ..mdp.plans import ConceptPlanStep


class EnvironmentContextBuilder:
    """Builds context objects for the environment layers."""
    
    @staticmethod
    def build_session_context(
        session_id: str,
        user_id: str,
        target_concepts: List[str],
        mastery_map: Optional[Dict[str, float]] = None,
        current_index: int = 0,
        strategy: str = "sequential",
    ) -> SessionContext:
        """Build session context from basic parameters.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            target_concepts: List of concepts to study
            mastery_map: Current mastery levels
            current_index: Current position in concept sequence
            strategy: Session strategy
            
        Returns:
            SessionContext instance
        """
        return SessionContext(
            session_id=session_id,
            user_id=user_id,
            target_concepts=target_concepts,
            current_concept_index=current_index,
            mastery_map=dict(mastery_map or {}),
            session_strategy=strategy,
        )
    
    @staticmethod
    def build_concept_context(
        concept_id: str,
        current_mastery: float = 0.0,
        target_mastery: float = 0.8,
        current_step_index: int = 0,
        phase: str = "learning",
    ) -> ConceptContext:
        """Build concept context from basic parameters.
        
        Args:
            concept_id: Concept identifier
            current_mastery: Current mastery level
            target_mastery: Target mastery level
            current_step_index: Current position in plan
            phase: Learning phase
            
        Returns:
            ConceptContext instance
        """
        return ConceptContext(
            concept_id=concept_id,
            current_mastery=current_mastery,
            target_mastery=target_mastery,
            current_step_index=current_step_index,
            phase=phase,
        )
    
    @staticmethod
    def build_tutor_context_from_step(
        step: ConceptPlanStep,
        button_label: str = "Continue",
    ) -> TutorContext:
        """Build tutor context from a plan step.
        
        Args:
            step: Plan step to execute
            button_label: Label for continue button
            
        Returns:
            TutorContext instance
        """
        return TutorContext(
            step_instruction=step.instruction,
            step_type=step.step_type,
            button_label=button_label,
            awaiting_response=True,
        )
    
    @staticmethod
    def build_planning_observation(
        concept_id: str,
        current_mastery: float,
        target_mastery: float,
        session_id: str,
        user_id: str,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build observation dictionary for concept planning.
        
        This creates the context needed for LLM-based plan generation.
        
        Args:
            concept_id: Concept to plan for
            current_mastery: Starting mastery
            target_mastery: Goal mastery
            session_id: Session identifier
            user_id: User identifier
            additional_context: Extra context fields
            
        Returns:
            Planning observation dictionary
        """
        obs = {
            "concept_id": concept_id,
            "current_mastery": current_mastery,
            "target_mastery": target_mastery,
            "mastery_gap": target_mastery - current_mastery,
            "session_id": session_id,
            "user_id": user_id,
            "focus_concept": concept_id,
        }
        
        if additional_context:
            obs.update(additional_context)
        
        return obs
