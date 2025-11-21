"""
Main orchestrator for the 3-layer MDP environment.

This is the entry point that coordinates all three layers:
- Session layer: Manages concept sequencing
- Concept layer: Executes learning plans
- Tutor layer: Generates pedagogical responses

The orchestrator is clean, self-contained, and doesn't depend on
legacy code structures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .session_env import SessionEnvironment, SessionAction
from .concept_env import ConceptEnvironment, ConceptAction
from .tutor_env import TutorEnvironment, TutorAction
from .policies import make_session_policy, make_concept_policy, make_tutor_policy
from .tools import EnvironmentStateManager, PlanCoordinator, ResponseBuilder
from ..mdp.tools_factory import make_session_planner_tool, make_concept_planner_tool
from ..tools.concept_planner import make_concept_planning_observation
from ..context_model import TutorContext


logger = logging.getLogger(__name__)


class EnvironmentOrchestrator:
    """Main orchestrator for the 3-layer MDP environment.
    
    Coordinates session, concept, and tutor environments to handle
    one tutoring turn.
    """
    
    def __init__(self, cur: Any):
        """Initialize orchestrator.
        
        Args:
            cur: Database cursor for persistence
        """
        self.cur = cur
        self.state_manager = EnvironmentStateManager(cur)
        
        # Initialize policies
        self.session_policy = make_session_policy()
        self.concept_policy = make_concept_policy()
        self.tutor_policy = make_tutor_policy()
        
        # Initialize planners
        session_planner = make_session_planner_tool()
        concept_planner = make_concept_planner_tool()
        self.plan_coordinator = PlanCoordinator(session_planner, concept_planner)
    
    def run_turn(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        target_concepts: Optional[List[str]] = None,
        user_clicked_button: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute one turn through all three layers.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            user_message: User's input message
            target_concepts: List of concepts to study (for new sessions)
            user_clicked_button: Whether user clicked a button
            **kwargs: Additional context
            
        Returns:
            Frontend-compatible response dictionary
        """
        logger.info(
            "environment_orchestrator_turn_start",
            extra={
                "session_id": session_id,
                "user_id": user_id,
                "has_target_concepts": target_concepts is not None,
                "user_clicked": user_clicked_button,
            }
        )
        
        # ===== Layer 1: Session Environment =====
        session_env = self._initialize_session_env(
            session_id=session_id,
            user_id=user_id,
            target_concepts=target_concepts,
        )
        
        # Check if session already complete
        if session_env.is_terminated():
            return ResponseBuilder.build_session_complete_response(session_env)
        
        # Get current concept from session
        current_concept_id = session_env.get_current_concept()
        if not current_concept_id:
            # No concept available - session should end
            session_action = SessionAction.END_SESSION
            transition = session_env.step(session_action)
            return ResponseBuilder.build_session_complete_response(session_env)
        
        logger.info(
            "environment_session_focus",
            extra={
                "session_id": session_id,
                "focus_concept": current_concept_id,
                "concept_index": session_env.state.current_concept_index,
            }
        )
        
        # ===== Layer 2: Concept Environment =====
        concept_env = self._initialize_concept_env(
            session_id=session_id,
            user_id=user_id,
            concept_id=current_concept_id,
            session_env=session_env,
        )
        
        # Ensure concept has a plan
        if not concept_env.has_plan():
            plan = self._generate_concept_plan(
                session_id=session_id,
                user_id=user_id,
                concept_id=current_concept_id,
                concept_env=concept_env,
            )
            concept_env.set_plan(plan)
            logger.info(
                "environment_plan_generated",
                extra={
                    "session_id": session_id,
                    "concept_id": current_concept_id,
                    "num_steps": len(plan.steps),
                }
            )
        
        # Decide concept-level action
        concept_action = self.concept_policy.decide(
            state=concept_env.state,
            step_complete=user_clicked_button,
        )
        
        logger.info(
            "environment_concept_action",
            extra={
                "session_id": session_id,
                "concept_id": current_concept_id,
                "action": concept_action.value,
            }
        )
        
        # Execute concept action
        concept_transition = concept_env.step(
            action=concept_action,
            step_complete=user_clicked_button,
        )
        
        # Handle concept completion
        if concept_transition.terminated:
            # Update session mastery
            session_env.update_mastery(
                concept_id=current_concept_id,
                mastery=concept_env.state.current_mastery,
            )
            
            # Advance session to next concept
            session_action = SessionAction.ADVANCE_CONCEPT
            session_transition = session_env.step(
                action=session_action,
                concept_complete=True,
                concept_mastery=concept_env.state.current_mastery,
            )
            
            # Check if session complete
            if session_transition.terminated:
                return ResponseBuilder.build_session_complete_response(session_env)
            
            # Get next concept
            next_concept_id = session_env.get_current_concept()
            return ResponseBuilder.build_concept_complete_response(
                concept_env=concept_env,
                next_concept=next_concept_id,
            )
        
        # If concept needs replanning
        if concept_transition.outputs.get("needs_replan"):
            plan = self._generate_concept_plan(
                session_id=session_id,
                user_id=user_id,
                concept_id=current_concept_id,
                concept_env=concept_env,
            )
            concept_env.set_plan(plan)
        
        # ===== Layer 3: Tutor Environment =====
        tutor_env = self._initialize_tutor_env(
            session_id=session_id,
            user_id=user_id,
            concept_id=current_concept_id,
        )
        
        # Get current step from concept plan
        current_step = concept_env.get_current_step()
        
        # Decide tutor action
        tutor_action = self.tutor_policy.decide(
            state=tutor_env.state,
            current_step=current_step,
        )
        
        logger.info(
            "environment_tutor_action",
            extra={
                "session_id": session_id,
                "concept_id": current_concept_id,
                "action": tutor_action.value,
                "step_type": current_step.step_type if current_step else None,
            }
        )
        
        # Execute tutor action
        tutor_transition = tutor_env.step(
            action=tutor_action,
            step=current_step,
            user_clicked=user_clicked_button,
        )
        
        # Build response
        response = ResponseBuilder.build_response(
            tutor_outputs=tutor_transition.outputs,
            session_env=session_env,
            concept_env=concept_env,
            tutor_env=tutor_env,
        )
        
        # Persist states
        self.state_manager.save_session_state(session_env)
        self.state_manager.save_concept_state(concept_env)
        
        logger.info(
            "environment_orchestrator_turn_complete",
            extra={
                "session_id": session_id,
                "concept_id": current_concept_id,
            }
        )
        
        return response
    
    def _initialize_session_env(
        self,
        session_id: str,
        user_id: str,
        target_concepts: Optional[List[str]] = None,
    ) -> SessionEnvironment:
        """Initialize or load session environment.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            target_concepts: Concepts for new session
            
        Returns:
            SessionEnvironment instance
        """
        # Try to load existing state
        state_data = self.state_manager.load_session_state(session_id, user_id)
        
        # Create environment
        session_env = SessionEnvironment(
            session_id=session_id,
            user_id=user_id,
            initial_mastery_map=state_data.get("mastery_map", {}),
        )
        
        # Check if we need to create a session plan
        if state_data.get("session_plan") is None and target_concepts:
            # Generate session plan
            plan = self.plan_coordinator.generate_session_plan(
                user_id=user_id,
                session_id=session_id,
                target_concepts=target_concepts,
                mastery_map=session_env.state.mastery_map,
            )
            session_env.set_session_plan(plan)
        
        return session_env
    
    def _initialize_concept_env(
        self,
        session_id: str,
        user_id: str,
        concept_id: str,
        session_env: SessionEnvironment,
    ) -> ConceptEnvironment:
        """Initialize or load concept environment.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            concept_id: Concept being studied
            session_env: Parent session environment
            
        Returns:
            ConceptEnvironment instance
        """
        # Get mastery from session
        initial_mastery = session_env.get_mastery(concept_id)
        
        # Load state if exists
        state_data = self.state_manager.load_concept_state(
            session_id, user_id, concept_id
        )
        
        # Create environment
        concept_env = ConceptEnvironment(
            session_id=session_id,
            user_id=user_id,
            concept_id=concept_id,
            initial_mastery=initial_mastery,
            target_mastery=0.8,  # Default target
        )
        
        return concept_env
    
    def _initialize_tutor_env(
        self,
        session_id: str,
        user_id: str,
        concept_id: str,
    ) -> TutorEnvironment:
        """Initialize tutor environment.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            concept_id: Current concept
            
        Returns:
            TutorEnvironment instance
        """
        return TutorEnvironment(
            session_id=session_id,
            user_id=user_id,
            concept_id=concept_id,
        )
    
    def _generate_concept_plan(
        self,
        session_id: str,
        user_id: str,
        concept_id: str,
        concept_env: ConceptEnvironment,
    ) -> Any:
        """Generate a concept learning plan using LLM.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            concept_id: Concept to plan for
            concept_env: Concept environment
            
        Returns:
            ConceptPlan
        """
        # Build context observation for planning
        context_obs = {
            "concept_id": concept_id,
            "current_mastery": concept_env.state.current_mastery,
            "target_mastery": concept_env.state.target_mastery,
            "mastery_gap": concept_env.get_mastery_gap(),
            "session_id": session_id,
            "user_id": user_id,
        }
        
        # Generate plan
        plan = self.plan_coordinator.generate_concept_plan(
            user_id=user_id,
            session_id=session_id,
            concept_id=concept_id,
            target_mastery=concept_env.state.target_mastery,
            current_mastery=concept_env.state.current_mastery,
            context_obs=context_obs,
        )
        
        return plan


def run_environment_turn(
    ctx: Any,
    cur: Any,
) -> Dict[str, Any]:
    """Main entry point for environment-based orchestration.
    
    This is the interface that matches existing runtime expectations.
    
    Args:
        ctx: TurnContext with session_id, user_id, message, etc.
        cur: Database cursor
        
    Returns:
        Frontend-compatible response
    """
    orchestrator = EnvironmentOrchestrator(cur)
    
    # Extract from context
    session_id = ctx.session_id
    user_id = ctx.user_id
    message = ctx.message
    target_concepts = getattr(ctx, "target_concepts", None)
    
    # Check if user clicked button
    # This would be parsed from payload in real implementation
    payload = getattr(ctx, "payload", {}) or {}
    user_clicked = payload.get("button_clicked", False)
    
    return orchestrator.run_turn(
        session_id=session_id,
        user_id=user_id,
        user_message=message,
        target_concepts=target_concepts,
        user_clicked_button=user_clicked,
    )
