"""
Tutor Environment - Layer 3 of 3-layer MDP.

Executes individual pedagogical actions and generates tutor responses.

Responsibilities:
- Execute one plan step into concrete tutor messages
- Generate appropriate UI controls (buttons)
- Handle user interactions (button clicks)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import BaseEnvironment, EnvironmentState, EnvironmentTransition
from .context import TutorContext
from ..mdp.plans import ConceptPlanStep


class TutorAction(str, Enum):
    """Action space for tutor-level pedagogical moves."""
    
    EXPLAIN = "EXPLAIN"
    ASK_QUESTION = "ASK_QUESTION"
    WORKED_EXAMPLE = "WORKED_EXAMPLE"
    GUIDED_PRACTICE = "GUIDED_PRACTICE"
    SUMMARY = "SUMMARY"
    QUIZ = "QUIZ"
    TRANSITION = "TRANSITION"  # Moving between concepts/phases


@dataclass
class TutorState(EnvironmentState):
    """State for the tutor environment.
    
    Tracks the current pedagogical action and interaction context.
    """
    
    concept_id: str = ""
    current_step: Optional[ConceptPlanStep] = None
    last_action: Optional[TutorAction] = None
    awaiting_user_input: bool = False
    button_label: str = "Continue"
    action_history: List[str] = field(default_factory=list)
    turn_in_step: int = 0


class TutorEnvironment(BaseEnvironment[TutorState]):
    """Environment for executing individual pedagogical actions.
    
    This is Layer 3 of the 3-layer MDP architecture. It translates
    plan steps into concrete tutor messages and UI elements.
    """
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        concept_id: str,
    ):
        """Initialize tutor environment.
        
        Args:
            session_id: Parent session ID
            user_id: User identifier
            concept_id: Current concept being taught
        """
        episode_id = f"tutor-{session_id}-{concept_id}"
        
        state = TutorState(
            episode_id=episode_id,
            session_id=session_id,
            user_id=user_id,
            concept_id=concept_id,
            current_step=None,
            last_action=None,
            awaiting_user_input=False,
            button_label="Continue",
            action_history=[],
            turn_in_step=0,
            terminated=False,
        )
        
        super().__init__(state)
    
    def reset(self) -> TutorState:
        """Reset tutor environment to initial state.
        
        Returns:
            Fresh tutor state
        """
        self.state.current_step = None
        self.state.last_action = None
        self.state.awaiting_user_input = False
        self.state.button_label = "Continue"
        self.state.action_history = []
        self.state.turn_in_step = 0
        self.state.terminated = False
        self.state.termination_reason = None
        
        return self.state
    
    def step(
        self,
        action: TutorAction,
        step: Optional[ConceptPlanStep] = None,
        user_clicked: bool = False,
        **kwargs
    ) -> EnvironmentTransition:
        """Execute one tutor-level step.
        
        Args:
            action: Tutor action to take
            step: Plan step to execute (if applicable)
            user_clicked: Whether user clicked button
            **kwargs: Additional context
            
        Returns:
            Transition with tutor response
        """
        self.state.turn_in_step += 1
        
        outputs: Dict[str, Any] = {
            "messages": [],
            "ui_mode": "buttons_only",
            "button_options": [],
        }
        
        info = {
            "action": action.value,
            "turn_in_step": self.state.turn_in_step,
        }
        
        # Update state
        self.state.last_action = action
        if step is not None:
            self.state.current_step = step
        
        # Record action in history
        self.state.action_history.append(action.value)
        if len(self.state.action_history) > 10:
            self.state.action_history = self.state.action_history[-10:]
        
        # Generate response based on action
        if action == TutorAction.EXPLAIN:
            message = self._generate_explanation(step, **kwargs)
            outputs["messages"].append({
                "role": "assistant",
                "content": message,
            })
            outputs["button_options"] = ["Continue"]
            self.state.button_label = "Continue"
            self.state.awaiting_user_input = True
            info["explanation_generated"] = True
        
        elif action == TutorAction.ASK_QUESTION:
            message = self._generate_question(step, **kwargs)
            outputs["messages"].append({
                "role": "assistant",
                "content": message,
            })
            outputs["button_options"] = ["Continue"]
            self.state.button_label = "Continue"
            self.state.awaiting_user_input = True
            info["question_generated"] = True
        
        elif action == TutorAction.WORKED_EXAMPLE:
            message = self._generate_worked_example(step, **kwargs)
            outputs["messages"].append({
                "role": "assistant",
                "content": message,
            })
            outputs["button_options"] = ["Continue"]
            self.state.button_label = "Continue"
            self.state.awaiting_user_input = True
            info["example_generated"] = True
        
        elif action == TutorAction.SUMMARY:
            message = self._generate_summary(step, **kwargs)
            outputs["messages"].append({
                "role": "assistant",
                "content": message,
            })
            outputs["button_options"] = ["Continue"]
            self.state.button_label = "Continue"
            self.state.awaiting_user_input = True
            info["summary_generated"] = True
        
        elif action == TutorAction.TRANSITION:
            message = self._generate_transition(**kwargs)
            outputs["messages"].append({
                "role": "assistant",
                "content": message,
            })
            outputs["button_options"] = ["Continue"]
            self.state.button_label = "Continue"
            self.state.awaiting_user_input = True
            info["transition_generated"] = True
        
        # Handle user interaction
        if user_clicked:
            self.state.awaiting_user_input = False
            info["user_clicked"] = True
            outputs["step_complete"] = True
        
        return EnvironmentTransition(
            next_state=self.state,
            outputs=outputs,
            info=info,
            terminated=self.state.terminated,
            termination_reason=self.state.termination_reason,
        )
    
    def set_current_step(self, step: ConceptPlanStep) -> None:
        """Set the current plan step to execute.
        
        Args:
            step: Plan step to work on
        """
        self.state.current_step = step
        self.state.turn_in_step = 0
    
    def _generate_explanation(
        self,
        step: Optional[ConceptPlanStep],
        **kwargs
    ) -> str:
        """Generate explanation message.
        
        Args:
            step: Plan step with instruction
            **kwargs: Additional context
            
        Returns:
            Explanation text
        """
        if step and step.instruction:
            return step.instruction
        
        return "Let me explain this concept..."
    
    def _generate_question(
        self,
        step: Optional[ConceptPlanStep],
        **kwargs
    ) -> str:
        """Generate question message.
        
        Args:
            step: Plan step with instruction
            **kwargs: Additional context
            
        Returns:
            Question text
        """
        if step and step.instruction:
            return step.instruction
        
        return "Let me ask you a question..."
    
    def _generate_worked_example(
        self,
        step: Optional[ConceptPlanStep],
        **kwargs
    ) -> str:
        """Generate worked example message.
        
        Args:
            step: Plan step with instruction
            **kwargs: Additional context
            
        Returns:
            Example text
        """
        if step and step.instruction:
            return step.instruction
        
        return "Here's an example..."
    
    def _generate_summary(
        self,
        step: Optional[ConceptPlanStep],
        **kwargs
    ) -> str:
        """Generate summary message.
        
        Args:
            step: Plan step with instruction
            **kwargs: Additional context
            
        Returns:
            Summary text
        """
        if step and step.instruction:
            return step.instruction
        
        concept_id = kwargs.get("concept_id", self.state.concept_id)
        return f"Let's summarize what we've learned about {concept_id}..."
    
    def _generate_transition(self, **kwargs) -> str:
        """Generate transition message.
        
        Args:
            **kwargs: Context including next_concept, etc.
            
        Returns:
            Transition text
        """
        next_concept = kwargs.get("next_concept")
        if next_concept:
            return f"Great! Now let's move on to {next_concept}."
        
        return "Moving to the next part..."
    
    def is_awaiting_input(self) -> bool:
        """Check if tutor is waiting for user interaction.
        
        Returns:
            True if awaiting button click or other input
        """
        return self.state.awaiting_user_input
