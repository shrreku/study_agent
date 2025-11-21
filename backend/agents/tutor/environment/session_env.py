"""
Session Environment - Layer 1 of 3-layer MDP.

Manages the overall study session, concept sequencing, and session-level goals.

Responsibilities:
- Track which concept is currently being studied
- Decide when to move to the next concept
- Determine when the session should end
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .base import BaseEnvironment, EnvironmentState, EnvironmentTransition
from .context import SessionContext
from ..mdp.plans import SessionPlan, SessionPlanEntry


class SessionAction(str, Enum):
    """Action space for session-level decisions."""
    
    START_CONCEPT = "START_CONCEPT"
    CONTINUE_CONCEPT = "CONTINUE_CONCEPT"
    ADVANCE_CONCEPT = "ADVANCE_CONCEPT"
    END_SESSION = "END_SESSION"


@dataclass
class SessionState(EnvironmentState):
    """State for the session environment.
    
    Tracks progress through the session plan and concept mastery.
    """
    
    session_plan: Optional[SessionPlan] = None
    current_concept_index: int = 0
    concepts_completed: int = 0
    mastery_map: Dict[str, float] = field(default_factory=dict)
    turn_count: int = 0
    
    def get_current_concept_id(self) -> Optional[str]:
        """Get the ID of the concept currently being studied.
        
        Returns:
            Concept ID or None if no plan or session complete
        """
        if self.session_plan is None or not self.session_plan.entries:
            return None
        
        if 0 <= self.current_concept_index < len(self.session_plan.entries):
            return self.session_plan.entries[self.current_concept_index].concept_id
        
        return None
    
    def is_session_complete(self) -> bool:
        """Check if all planned concepts have been covered.
        
        Returns:
            True if session should end
        """
        if self.session_plan is None or not self.session_plan.entries:
            return True
        
        return self.current_concept_index >= len(self.session_plan.entries)


class SessionEnvironment(BaseEnvironment[SessionState]):
    """Environment for managing the overall study session.
    
    This is Layer 1 of the 3-layer MDP architecture. It coordinates
    concept-level episodes and maintains session-level state.
    """
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        session_plan: Optional[SessionPlan] = None,
        initial_mastery_map: Optional[Dict[str, float]] = None,
    ):
        """Initialize session environment.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            session_plan: Pre-built session plan or None
            initial_mastery_map: Current mastery levels for concepts
        """
        episode_id = f"session-{session_id}"
        
        state = SessionState(
            episode_id=episode_id,
            session_id=session_id,
            user_id=user_id,
            session_plan=session_plan,
            current_concept_index=0,
            concepts_completed=0,
            mastery_map=dict(initial_mastery_map or {}),
            turn_count=0,
            terminated=False,
        )
        
        super().__init__(state)
    
    def reset(self) -> SessionState:
        """Reset session to initial state.
        
        Returns:
            Fresh session state
        """
        self.state.current_concept_index = 0
        self.state.concepts_completed = 0
        self.state.turn_count = 0
        self.state.terminated = False
        self.state.termination_reason = None
        
        return self.state
    
    def step(
        self,
        action: SessionAction,
        concept_complete: bool = False,
        concept_mastery: Optional[float] = None,
        **kwargs
    ) -> EnvironmentTransition:
        """Execute one session-level step.
        
        Args:
            action: Session action to take
            concept_complete: Whether current concept episode ended
            concept_mastery: Updated mastery for current concept
            **kwargs: Additional context
            
        Returns:
            Transition with updated state
        """
        self.state.turn_count += 1
        
        outputs: Dict = {}
        info: Dict = {
            "action": action.value,
            "turn_count": self.state.turn_count,
        }
        
        # Update mastery if provided
        current_concept = self.state.get_current_concept_id()
        if current_concept and concept_mastery is not None:
            self.state.mastery_map[current_concept] = concept_mastery
            info["mastery_updated"] = True
        
        # Handle actions
        if action == SessionAction.END_SESSION:
            self.state.terminated = True
            self.state.termination_reason = "user_requested_end"
            outputs["session_message"] = "Session ended. Great work!"
        
        elif action == SessionAction.ADVANCE_CONCEPT:
            if concept_complete:
                self.state.concepts_completed += 1
                self.state.current_concept_index += 1
                
                info["concept_advanced"] = True
                info["new_index"] = self.state.current_concept_index
                
                # Check if session is now complete
                if self.state.is_session_complete():
                    self.state.terminated = True
                    self.state.termination_reason = "all_concepts_complete"
                    outputs["session_message"] = "All concepts covered! Session complete."
                else:
                    next_concept = self.state.get_current_concept_id()
                    outputs["transition_message"] = f"Moving to next concept: {next_concept}"
        
        elif action == SessionAction.START_CONCEPT:
            current_concept = self.state.get_current_concept_id()
            if current_concept:
                outputs["focus_concept"] = current_concept
                info["concept_started"] = True
        
        elif action == SessionAction.CONTINUE_CONCEPT:
            # No state change, just continuing current concept
            info["continuing"] = True
        
        return EnvironmentTransition(
            next_state=self.state,
            outputs=outputs,
            info=info,
            terminated=self.state.terminated,
            termination_reason=self.state.termination_reason,
        )
    
    def set_session_plan(self, plan: SessionPlan) -> None:
        """Set or update the session plan.
        
        Args:
            plan: SessionPlan to use
        """
        self.state.session_plan = plan
        self.state.current_concept_index = 0
    
    def get_current_concept(self) -> Optional[str]:
        """Get the concept that should be studied now.
        
        Returns:
            Concept ID or None
        """
        return self.state.get_current_concept_id()
    
    def get_mastery(self, concept_id: str) -> float:
        """Get mastery level for a concept.
        
        Args:
            concept_id: Concept to query
            
        Returns:
            Mastery level (0.0 to 1.0)
        """
        return self.state.mastery_map.get(concept_id, 0.0)
    
    def update_mastery(self, concept_id: str, mastery: float) -> None:
        """Update mastery for a concept.
        
        Args:
            concept_id: Concept to update
            mastery: New mastery level
        """
        self.state.mastery_map[concept_id] = max(0.0, min(1.0, mastery))
