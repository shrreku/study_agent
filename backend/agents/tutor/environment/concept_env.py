"""
Concept Environment - Layer 2 of 3-layer MDP.

Manages learning for a single concept, including plan execution and mastery tracking.

Responsibilities:
- Generate and maintain concept learning plan
- Track progress through plan steps
- Monitor mastery and decide when concept is complete
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .base import BaseEnvironment, EnvironmentState, EnvironmentTransition
from .context import ConceptContext
from ..mdp.plans import ConceptPlan, ConceptPlanStep


class ConceptAction(str, Enum):
    """Action space for concept-level decisions."""
    
    GENERATE_PLAN = "GENERATE_PLAN"
    EXECUTE_STEP = "EXECUTE_STEP"
    ADVANCE_STEP = "ADVANCE_STEP"
    REPLAN = "REPLAN"
    COMPLETE_CONCEPT = "COMPLETE_CONCEPT"


@dataclass
class ConceptState(EnvironmentState):
    """State for the concept environment.
    
    Tracks the learning plan, progress, and mastery for one concept.
    """
    
    concept_id: str = ""
    concept_plan: Optional[ConceptPlan] = None
    current_step_index: int = 0
    current_mastery: float = 0.0
    target_mastery: float = 0.8
    initial_mastery: float = 0.0
    phase: str = "learning"  # learning, assessment, complete
    steps_completed: int = 0
    plan_version: int = 0  # Increments on replan
    
    def get_current_step(self) -> Optional[ConceptPlanStep]:
        """Get the current step to execute.
        
        Returns:
            Current step or None if plan exhausted
        """
        if self.concept_plan is None or not self.concept_plan.steps:
            return None
        
        if 0 <= self.current_step_index < len(self.concept_plan.steps):
            return self.concept_plan.steps[self.current_step_index]
        
        return None
    
    def is_plan_complete(self) -> bool:
        """Check if all plan steps have been executed.
        
        Returns:
            True if at end of plan
        """
        if self.concept_plan is None or not self.concept_plan.steps:
            return True
        
        return self.current_step_index >= len(self.concept_plan.steps)
    
    def is_mastery_reached(self) -> bool:
        """Check if target mastery has been achieved.
        
        Returns:
            True if mastery goal met
        """
        return self.current_mastery >= self.target_mastery


class ConceptEnvironment(BaseEnvironment[ConceptState]):
    """Environment for managing learning of a single concept.
    
    This is Layer 2 of the 3-layer MDP architecture. It executes
    a learning plan for one concept and tracks mastery progress.
    """
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        concept_id: str,
        initial_mastery: float = 0.0,
        target_mastery: float = 0.8,
        concept_plan: Optional[ConceptPlan] = None,
    ):
        """Initialize concept environment.
        
        Args:
            session_id: Parent session ID
            user_id: User identifier
            concept_id: Concept being studied
            initial_mastery: Starting mastery level
            target_mastery: Goal mastery level
            concept_plan: Pre-built plan or None to generate
        """
        episode_id = f"concept-{session_id}-{concept_id}"
        
        state = ConceptState(
            episode_id=episode_id,
            session_id=session_id,
            user_id=user_id,
            concept_id=concept_id,
            concept_plan=concept_plan,
            current_step_index=0,
            current_mastery=initial_mastery,
            target_mastery=target_mastery,
            initial_mastery=initial_mastery,
            phase="learning",
            steps_completed=0,
            plan_version=0,
            terminated=False,
        )
        
        super().__init__(state)
    
    def reset(self) -> ConceptState:
        """Reset concept environment to initial state.
        
        Returns:
            Fresh concept state
        """
        self.state.current_step_index = 0
        self.state.current_mastery = self.state.initial_mastery
        self.state.phase = "learning"
        self.state.steps_completed = 0
        self.state.terminated = False
        self.state.termination_reason = None
        
        return self.state
    
    def step(
        self,
        action: ConceptAction,
        mastery_delta: float = 0.0,
        step_complete: bool = False,
        **kwargs
    ) -> EnvironmentTransition:
        """Execute one concept-level step.
        
        Args:
            action: Concept action to take
            mastery_delta: Change in mastery from this step
            step_complete: Whether current step finished
            **kwargs: Additional context
            
        Returns:
            Transition with updated state
        """
        outputs = {}
        info = {
            "action": action.value,
            "concept_id": self.state.concept_id,
            "step_index": self.state.current_step_index,
        }
        
        # Update mastery
        if mastery_delta != 0.0:
            old_mastery = self.state.current_mastery
            self.state.current_mastery = max(0.0, min(1.0, 
                self.state.current_mastery + mastery_delta))
            info["mastery_change"] = self.state.current_mastery - old_mastery
            info["new_mastery"] = self.state.current_mastery
        
        # Handle actions
        if action == ConceptAction.GENERATE_PLAN:
            # Plan will be set externally via set_plan()
            info["plan_generated"] = True
            outputs["needs_plan"] = True
        
        elif action == ConceptAction.EXECUTE_STEP:
            current_step = self.state.get_current_step()
            if current_step:
                outputs["current_step"] = current_step
                outputs["step_type"] = current_step.step_type
                outputs["instruction"] = current_step.instruction
                info["executing_step"] = True
            else:
                # No step available
                outputs["needs_plan"] = True
                info["no_step_available"] = True
        
        elif action == ConceptAction.ADVANCE_STEP:
            if step_complete:
                self.state.current_step_index += 1
                self.state.steps_completed += 1
                info["step_advanced"] = True
                info["new_index"] = self.state.current_step_index
                
                # Check if plan complete
                if self.state.is_plan_complete():
                    info["plan_exhausted"] = True
                    
                    # Check mastery
                    if self.state.is_mastery_reached():
                        self.state.phase = "complete"
                        outputs["concept_complete"] = True
                    else:
                        # Need more practice
                        outputs["needs_replan"] = True
        
        elif action == ConceptAction.REPLAN:
            self.state.current_step_index = 0
            self.state.plan_version += 1
            info["replanned"] = True
            outputs["needs_plan"] = True
        
        elif action == ConceptAction.COMPLETE_CONCEPT:
            self.state.terminated = True
            self.state.phase = "complete"
            
            if self.state.is_mastery_reached():
                self.state.termination_reason = "mastery_reached"
                outputs["completion_message"] = f"Mastery achieved for {self.state.concept_id}!"
            else:
                self.state.termination_reason = "manual_completion"
                outputs["completion_message"] = f"Moving on from {self.state.concept_id}"
            
            info["concept_complete"] = True
        
        return EnvironmentTransition(
            next_state=self.state,
            outputs=outputs,
            info=info,
            terminated=self.state.terminated,
            termination_reason=self.state.termination_reason,
        )
    
    def set_plan(self, plan: ConceptPlan) -> None:
        """Set or update the concept learning plan.
        
        Args:
            plan: ConceptPlan to use
        """
        self.state.concept_plan = plan
        self.state.current_step_index = 0
        
        # Update mastery targets from plan if available
        if plan.target_mastery is not None:
            self.state.target_mastery = plan.target_mastery
        if plan.initial_mastery is not None:
            self.state.initial_mastery = plan.initial_mastery
    
    def get_current_step(self) -> Optional[ConceptPlanStep]:
        """Get the current step to execute.
        
        Returns:
            Current step or None
        """
        return self.state.get_current_step()
    
    def has_plan(self) -> bool:
        """Check if a plan exists and has steps.
        
        Returns:
            True if plan is available
        """
        return (self.state.concept_plan is not None and 
                len(self.state.concept_plan.steps) > 0)
    
    def get_mastery_gap(self) -> float:
        """Calculate remaining mastery gap.
        
        Returns:
            Distance to target mastery
        """
        return self.state.target_mastery - self.state.current_mastery
    
    def advance_to_step(self, step_index: int) -> None:
        """Jump to a specific step index.
        
        Args:
            step_index: Index to jump to
        """
        if self.state.concept_plan and 0 <= step_index < len(self.state.concept_plan.steps):
            self.state.current_step_index = step_index
