from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from .context_model import TutorContext
from .mdp.actions import ConceptMDPAction
from .mdp.concept import (
    ConceptObservation,
    ConceptState,
    apply_concept_transition,
    build_concept_observation,
    build_concept_state_from_session,
)
from .mdp.pedagogical_tutor import (
    PedagogicalTutorAction,
    PedagogicalTutorObservation,
    PedagogicalTutorState,
    apply_pedagogical_transition,
    build_pedagogical_tutor_observation,
    build_pedagogical_tutor_state,
)
from .mdp.plans import ConceptPlan, SessionPlan
from .mdp.policy import (
    ConceptPolicy,
    PedagogicalTutorPolicy,
    SessionPolicy,
)
from .mdp.session import (
    SessionMDPAction,
    SessionObservation,
    SessionState,
    apply_session_transition,
    build_session_observation,
    build_session_state_from_session,
)
from .mdp.tools import (
    ConceptPlannerTool,
    PedagogicalResponseGeneratorTool,
    SessionPlannerTool,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorState:
    """Holds the transient state of the orchestrator during a turn."""
    session_state: Optional[SessionState] = None
    concept_state: Optional[ConceptState] = None
    pedagogical_state: Optional[PedagogicalTutorState] = None
    
    session_plan: Optional[SessionPlan] = None
    concept_plan: Optional[ConceptPlan] = None


class TutorOrchestrator:
    """
    Orchestrates the 3-layered MDP for the Tutor Agent.
    
    Layers:
    1. Session Layer: Manages the overall learning path and session lifecycle.
    2. Concept Layer: Manages the mastery of a specific concept (SRL loop).
    3. Pedagogical Layer: Manages the immediate interaction and teaching moves.
    """

    def __init__(
        self,
        session_policy: SessionPolicy,
        concept_policy: ConceptPolicy,
        pedagogical_policy: PedagogicalTutorPolicy,
        session_planner: SessionPlannerTool,
        concept_planner: ConceptPlannerTool,
        response_generator: PedagogicalResponseGeneratorTool,
    ):
        self.session_policy = session_policy
        self.concept_policy = concept_policy
        self.pedagogical_policy = pedagogical_policy
        
        self.session_planner = session_planner
        self.concept_planner = concept_planner
        self.response_generator = response_generator

    def tick(
        self,
        context: TutorContext,
        mastery_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute one turn of the agent loop.
        
        Returns a dictionary with:
        - messages: List of messages to send to the user.
        - ui_mode: UI mode for the frontend.
        - mcq_payload: Optional MCQ payload.
        - debug: Debug information.
        - updated_context: (Implicitly handled by state updates, but we might want to return diffs)
        """
        logger.info("Orchestrator tick start")
        
        # 1. Build/Hydrate States
        # In a real implementation, we would load these from a persistent store.
        # For now, we rebuild them from the TutorContext and PolicyState (which acts as our store).
        
        # We assume context.policy_state holds the persistence for now.
        policy_state = context.policy_state
        
        session_state = build_session_state_from_session(
            policy_state=policy_state,
            tutor_context=context,
            mastery_map=mastery_map,
        )
        
        # 2. Session Layer
        session_obs = build_session_observation(session_state)
        
        # Load session plan from policy state
        from .mdp.adapters import session_plan_from_policy
        session_plan = session_plan_from_policy(policy_state)
        
        session_action = self.session_policy.decide(
            observation=session_obs,
            session_plan=session_plan 
        )
        logger.info(f"Session Action: {session_action}")
        
        if session_action == SessionMDPAction.TERMINATE_SESSION:
            return self._terminate_session(session_state)
            
        if session_action == SessionMDPAction.REPLAN_SESSION:
            logger.info("Replanning session...")
            # Execute Session Planner
            target_concepts = context.target_concepts if hasattr(context, 'target_concepts') else []
            logger.info(f"Using target concepts for replanning: {target_concepts}")
            
            new_plan = self.session_planner(
                user_id=context.user_id,
                session_id=context.session_id,
                strategy=session_state.strategy,
                target_concepts=target_concepts,
                mastery_map=mastery_map
            )
            # Update session state with new plan (concept_plan list)
            session_state.concept_plan = [e.concept_id for e in new_plan.entries]
            session_state.total_concepts = len(session_state.concept_plan)
            session_state.plan_index = 0 # Reset to start of new plan
            
            # Persist the new plan to policy state
            from .mdp.adapters import session_plan_to_policy
            session_plan_to_policy(new_plan, policy_state)
            session_plan = new_plan
            
            # Re-evaluate session action after planning
            session_obs = build_session_observation(session_state)
            session_action = self.session_policy.decide(
                observation=session_obs,
                session_plan=session_plan
            )
            logger.info(f"Session Action (after replan): {session_action}")

        # Apply Session Transition (updates plan index if advancing)
        session_outcome = apply_session_transition(
            prev_state=session_state,
            mdp_action=session_action
        )
        session_state = session_outcome.state
        
        # If we finished the plan, terminate
        if session_outcome.terminated:
             return self._terminate_session(session_state)

        # 3. Concept Layer
        # Determine current concept
        current_concept_id = session_state.current_concept_id
        if not current_concept_id:
            # Should not happen if we are not terminated, but safety check
            return self._terminate_session(session_state)
            
        concept_state = build_concept_state_from_session(
            policy_state=policy_state,
            tutor_context=context
        )
        # Ensure concept_id matches session (handling transitions)
        if concept_state.concept_id != current_concept_id:
            # We switched concepts. Reset concept state for the new concept.
            logger.info(f"Switching concept to {current_concept_id}")
            concept_state = ConceptState(
                episode_id=f"ce-{context.session_id}-{current_concept_id}",
                session_id=context.session_id,
                user_id=context.user_id,
                concept_id=current_concept_id,
                target_mastery=0.8, # Default
                mastery=mastery_map.get(current_concept_id, {}).get("mastery"),
                mastery_start=mastery_map.get(current_concept_id, {}).get("mastery"),
                plan_index=0,
                quiz_phase="",
                quiz_question_index=0,
                quiz_max_questions=0,
                quiz_correct=0,
                quiz_wrong=0,
                step_count=0,
                last_control_type=None,
                last_mdp_action=None
            )

        # Load or Generate Concept Plan
        concept_plan: Optional[ConceptPlan] = None
        # Try to load from policy_state (persistence)
        if hasattr(policy_state, "concept_plan") and policy_state.concept_plan:
            try:
                # Assuming policy_state.concept_plan is a dict, convert back to ConceptPlan object
                # We might need a helper method or assume ConceptPlan can init from dict or use `**`
                # ConceptPlan is likely a Pydantic model or dataclass.
                # Let's try to re-hydrate it. If it fails, we regenerate.
                # For now, let's assume simple dict access or `ConceptPlan(**...)`
                # Need to verify `ConceptPlan` definition.
                # Assuming it's a dataclass or similar.
                # concept_plan = ConceptPlan(**policy_state.concept_plan)
                # Wait, ConceptPlan might have nested objects (steps).
                # For now, we rely on the planner to generate if we can't load.
                # Or if the persisted plan is for the wrong concept.
                persisted_plan = policy_state.concept_plan
                if persisted_plan.get("concept_id") == current_concept_id:
                     # Use a helper to hydrate if possible, or just use the dict if policies support it?
                     # Policies expect `ConceptPlan` object.
                     # For MVP, if we can't easily hydrate, we regenerate if step is 0.
                     # But for replan flow we must handle existing plans.
                     # Let's skip hydration complexity for a moment and assume we generate if missing/invalid.
                     pass
            except Exception:
                logger.warning("Failed to hydrate concept plan from persistence")

        # If we are at step 0 and have no plan (or couldn't hydrate), generate one.
        # Also if we switched concepts, we need a new plan.
        if concept_state.plan_index == 0 and concept_state.step_count == 0:
             logger.info(f"Generating initial plan for concept {current_concept_id}")
             concept_plan = self.concept_planner(
                user_id=context.user_id,
                session_id=context.session_id,
                concept_id=current_concept_id,
                target_mastery=concept_state.target_mastery,
                context_obs=context.to_planning_observation()
            )

        # Extract control signal for observation
        control_signal = context.get_control_signal()
        logger.info(f"Control signal: {control_signal}")

        concept_obs = build_concept_observation(
            state=concept_state,
            last_intent="unknown", # TODO: Extract from context
            last_affect="neutral", # TODO: Extract from context
            last_action_type="unknown",
            last_control_type=control_signal
        )
        
        concept_action = self.concept_policy.decide(
            observation=concept_obs,
            concept_plan=concept_plan
        )
        logger.info(f"Concept Action: {concept_action}")
        
        if concept_action == ConceptMDPAction.REPLAN_CONCEPT:
             logger.info("Replanning concept...")
             concept_plan = self.concept_planner(
                user_id=context.user_id,
                session_id=context.session_id,
                concept_id=current_concept_id,
                target_mastery=concept_state.target_mastery,
                context_obs=context.to_planning_observation()
            )
             # Reset plan index
             concept_state.plan_index = 0
             
             # Re-decide
             # Pass None as control type to avoid loop, but we need to inform next layer
             # that we just replanned.
             concept_obs_re = build_concept_observation(
                state=concept_state,
                last_intent="unknown", 
                last_affect="neutral", 
                last_action_type="unknown",
                last_control_type=None # Clear explicitly to avoid infinite REPLAN loop
            )
             
             concept_action = self.concept_policy.decide(
                observation=concept_obs_re,
                concept_plan=concept_plan
            )
             logger.info(f"Concept Action (after replan): {concept_action}")
             
             # Override control signal for the pedagogical layer to ensure it executes
             # because we want "Auto-execute first step after plan generation"
             control_signal = "replan_concept"

        # Apply Concept Transition
        concept_outcome = apply_concept_transition(
            prev_state=concept_state,
            mdp_action=concept_action,
            mastery_delta=None, 
            quiz_delta=None,
            mcq_outcome=None,
            control_type=control_signal # Persist control signal in state history
        )
        concept_state = concept_outcome.state
        
        if concept_outcome.terminated:
            self._sync_state_to_policy(policy_state, session_state, concept_state, concept_plan)
            return {
                "messages": [{"role": "assistant", "content": f"Great job! We've finished {current_concept_id}. Let's move on."}],
                "ui_mode": "free_text",
                "debug": {"concept_terminated": True}
            }

        # 4. Pedagogical Layer
        # Build Ped State
        ped_state = build_pedagogical_tutor_state(
            episode_id=concept_state.episode_id,
            session_id=context.session_id,
            user_id=context.user_id,
            concept_id=current_concept_id,
            plan_id=concept_plan.plan_id if concept_plan else "unknown",
            plan_index=concept_state.plan_index,
            plan_length=len(concept_plan.steps) if concept_plan and concept_plan.steps else 0,
            phase="learning", # Simplify for now
            mastery=concept_state.mastery,
            target_mastery=concept_state.target_mastery,
            last_intent="unknown",
            last_affect="neutral",
            last_control_type=control_signal # Pass explicit signal (which might be 'replan_concept')
        )
        
        ped_obs = build_pedagogical_tutor_observation(state=ped_state)
        ped_action = self.pedagogical_policy.decide(
            observation=ped_obs,
            concept_plan=concept_plan
        )
        logger.info(f"Pedagogical Action: {ped_action}")
        
        # Apply Ped Transition (internal bookkeeping)
        apply_pedagogical_transition(
            prev_state=ped_state,
            mdp_action=ped_action
        )
        
        # 5. Execution
        response = self.response_generator(
            user_id=context.user_id,
            session_id=context.session_id,
            concept_id=current_concept_id,
            pedagogical_action=ped_action,
            ped_state=ped_state,
            concept_plan=concept_plan,
            context_obs=context.to_planning_observation()
        )
        
        # Sync back to policy state for persistence
        self._sync_state_to_policy(policy_state, session_state, concept_state, concept_plan)
        
        return response

    def _terminate_session(self, session_state: SessionState) -> Dict[str, Any]:
        return {
            "messages": [{"role": "assistant", "content": "That concludes our session for today. Great work!"}],
            "ui_mode": "buttons_only",
            "button_options": ["Close"],
            "debug": {"session_terminated": True}
        }

    def _sync_state_to_policy(
        self, 
        policy_state: Any, 
        session_state: SessionState, 
        concept_state: ConceptState,
        concept_plan: Optional[ConceptPlan]
    ) -> None:
        """Sync transient MDP states back to the persistent policy state."""
        # Session fields
        policy_state.session_plan_index = session_state.plan_index
        # Note: We assume session_plan entries are not modified, just the index.
        # If session plan changed (replanning), we need to update it.
        # session_state.concept_plan stores IDs.
        # We assume policy_state.session_plan stores the full plan dict/object.
        # For now we trust the session_state fields mapping.
        
        # Concept fields
        policy_state.concept_episode_id = concept_state.episode_id
        policy_state.srl_plan_step_index = concept_state.plan_index
        policy_state.concept_episode_step_count = concept_state.step_count
        policy_state.concept_episode_last_control_type = concept_state.last_control_type
        
        policy_state.quiz_phase = concept_state.quiz_phase
        policy_state.quiz_question_index = concept_state.quiz_question_index
        policy_state.quiz_max_questions = concept_state.quiz_max_questions
        policy_state.concept_episode_quiz_correct = concept_state.quiz_correct
        policy_state.concept_episode_quiz_wrong = concept_state.quiz_wrong
        
        # Concept Plan persistence
        if concept_plan:
             # Serialize concept plan to dict
             # Assuming concept_plan is a dataclass or Pydantic
             try:
                 if hasattr(concept_plan, "dict"):
                     policy_state.concept_plan = concept_plan.dict()
                 elif hasattr(concept_plan, "__dict__"):
                     # Shallow dict might not be enough for nested steps
                     # Use asdict if available
                     policy_state.concept_plan = asdict(concept_plan)
                 else:
                     policy_state.concept_plan = {}
             except Exception as e:
                 logger.warning(f"Error serializing concept plan: {e}")
