"""
Environment-specific tools for the 3-layer MDP architecture.

These tools provide environment management functions like state persistence,
plan generation coordination, and response building.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .session_env import SessionEnvironment, SessionState
from .concept_env import ConceptEnvironment, ConceptState
from .tutor_env import TutorEnvironment, TutorState
from ..mdp.plans import SessionPlan, ConceptPlan, ConceptPlanStep


class EnvironmentStateManager:
    """Manages persistence and loading of environment states.
    
    Coordinates saving/loading states across all three layers and
    handles serialization/deserialization.
    """
    
    def __init__(self, cur: Any):
        """Initialize with database cursor.
        
        Args:
            cur: Database cursor for persistence
        """
        self.cur = cur
    
    def load_session_state(
        self,
        session_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Load session state from database.
        
        Args:
            session_id: Session to load
            user_id: User identifier
            
        Returns:
            Dictionary with session state data
        """
        from psycopg2.extras import Json
        
        # Load from tutor_session policy field
        self.cur.execute(
            """
            SELECT policy, target_concepts
            FROM tutor_session
            WHERE id = %s::uuid
            LIMIT 1
            """,
            (session_id,),
        )
        row = self.cur.fetchone()
        
        if not row:
            return {
                "session_id": session_id,
                "user_id": user_id,
                "session_plan": None,
                "current_concept_index": 0,
                "mastery_map": {},
            }
        
        policy = row[0] or {}
        target_concepts = row[1] or []
        
        # Extract environment state from policy
        env_state = policy.get("environment_state", {})
        
        # Load mastery map from user_concept_mastery table
        self.cur.execute(
            """
            SELECT concept, mastery
            FROM user_concept_mastery
            WHERE user_id = %s::uuid
            """,
            (user_id,),
        )
        mastery_rows = self.cur.fetchall() or []
        mastery_map = {row[0]: float(row[1]) for row in mastery_rows if row[0]}
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "session_plan": env_state.get("session_plan"),
            "current_concept_index": env_state.get("current_concept_index", 0),
            "mastery_map": mastery_map,
            "turn_count": env_state.get("turn_count", 0),
        }
    
    def save_session_state(
        self,
        session_env: SessionEnvironment,
    ) -> None:
        """Persist session state to database.
        
        Args:
            session_env: Session environment to save
        """
        from psycopg2.extras import Json
        
        state = session_env.get_state()
        
        # Build environment state dictionary
        env_state = {
            "current_concept_index": state.current_concept_index,
            "concepts_completed": state.concepts_completed,
            "turn_count": state.turn_count,
            "terminated": state.terminated,
        }
        
        if state.session_plan:
            env_state["session_plan"] = state.session_plan.to_dict()
        
        # Load existing policy and merge
        self.cur.execute(
            """
            SELECT policy
            FROM tutor_session
            WHERE id = %s::uuid
            LIMIT 1
            """,
            (state.session_id,),
        )
        row = self.cur.fetchone()
        policy = (row[0] or {}) if row else {}
        
        # Update environment_state in policy
        policy["environment_state"] = env_state
        
        # Save back to database
        self.cur.execute(
            """
            UPDATE tutor_session
            SET policy = %s,
                updated_at = now()
            WHERE id = %s::uuid
            """,
            (Json(policy), state.session_id),
        )
        
        # Also update mastery in user_concept_mastery table
        for concept_id, mastery in state.mastery_map.items():
            self.cur.execute(
                """
                INSERT INTO user_concept_mastery (user_id, concept, mastery, last_seen)
                VALUES (%s::uuid, %s, %s, now())
                ON CONFLICT (user_id, concept) DO UPDATE
                  SET mastery = EXCLUDED.mastery,
                      last_seen = now()
                """,
                (state.user_id, concept_id, mastery),
            )
    
    def load_concept_state(
        self,
        session_id: str,
        user_id: str,
        concept_id: str,
    ) -> Dict[str, Any]:
        """Load concept state from database.
        
        Args:
            session_id: Parent session
            user_id: User identifier
            concept_id: Concept being studied
            
        Returns:
            Dictionary with concept state data
        """
        from psycopg2.extras import Json
        
        # Load from policy field, keyed by concept
        self.cur.execute(
            """
            SELECT policy
            FROM tutor_session
            WHERE id = %s::uuid
            LIMIT 1
            """,
            (session_id,),
        )
        row = self.cur.fetchone()
        
        if not row:
            return {
                "session_id": session_id,
                "user_id": user_id,
                "concept_id": concept_id,
                "concept_plan": None,
                "current_step_index": 0,
                "current_mastery": 0.0,
            }
        
        policy = row[0] or {}
        concept_states = policy.get("concept_states", {})
        concept_state = concept_states.get(concept_id, {})
        
        # Get current mastery
        self.cur.execute(
            """
            SELECT mastery
            FROM user_concept_mastery
            WHERE user_id = %s::uuid AND concept = %s
            LIMIT 1
            """,
            (user_id, concept_id),
        )
        mastery_row = self.cur.fetchone()
        current_mastery = float(mastery_row[0]) if mastery_row else 0.0
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "concept_id": concept_id,
            "concept_plan": concept_state.get("concept_plan"),
            "current_step_index": concept_state.get("current_step_index", 0),
            "current_mastery": current_mastery,
            "phase": concept_state.get("phase", "learning"),
            "terminated": bool(concept_state.get("terminated", False)),
        }
    
    def save_concept_state(
        self,
        concept_env: ConceptEnvironment,
    ) -> None:
        """Persist concept state to database.
        
        Args:
            concept_env: Concept environment to save
        """
        from psycopg2.extras import Json
        
        state = concept_env.get_state()
        
        # Build concept state dictionary
        concept_state_dict = {
            "current_step_index": state.current_step_index,
            "current_mastery": state.current_mastery,
            "target_mastery": state.target_mastery,
            "phase": state.phase,
            "steps_completed": state.steps_completed,
            "plan_version": state.plan_version,
            "terminated": state.terminated,
            "termination_reason": state.termination_reason,
        }
        
        if state.concept_plan:
            concept_state_dict["concept_plan"] = state.concept_plan.to_dict()
        
        # Load existing policy
        self.cur.execute(
            """
            SELECT policy
            FROM tutor_session
            WHERE id = %s::uuid
            LIMIT 1
            """,
            (state.session_id,),
        )
        row = self.cur.fetchone()
        policy = (row[0] or {}) if row else {}
        
        # Update concept_states in policy
        if "concept_states" not in policy:
            policy["concept_states"] = {}
        policy["concept_states"][state.concept_id] = concept_state_dict
        
        # Save back to database
        self.cur.execute(
            """
            UPDATE tutor_session
            SET policy = %s,
                updated_at = now(),
                last_concept = %s
            WHERE id = %s::uuid
            """,
            (Json(policy), state.concept_id, state.session_id),
        )


class PlanCoordinator:
    """Coordinates plan generation across environment layers.
    
    Wraps existing planning tools and provides environment-friendly
    interfaces for requesting plans.
    """
    
    def __init__(self, session_planner, concept_planner):
        """Initialize with planner tools.
        
        Args:
            session_planner: SessionPlannerTool instance
            concept_planner: ConceptPlannerTool instance
        """
        self.session_planner = session_planner
        self.concept_planner = concept_planner
    
    def generate_session_plan(
        self,
        user_id: str,
        session_id: str,
        target_concepts: List[str],
        mastery_map: Dict[str, float],
        strategy: str = "sequential",
    ) -> SessionPlan:
        """Generate a session plan.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            target_concepts: Concepts to cover
            mastery_map: Current mastery levels
            strategy: Planning strategy
            
        Returns:
            SessionPlan with concept sequence
        """
        # For MVP, just create a simple sequential plan
        # Later this can use LLM planner
        from ..mdp.plans import SessionPlanEntry
        
        entries = [
            SessionPlanEntry(
                concept_id=cid,
                target_mastery=0.8,
            )
            for cid in target_concepts
        ]
        
        return SessionPlan(
            strategy=strategy,
            entries=entries,
            plan_id=f"sp-{session_id}",
            source="simple_sequential",
        )
    
    def generate_concept_plan(
        self,
        user_id: str,
        session_id: str,
        concept_id: str,
        target_mastery: float,
        context_obs: Dict[str, Any],
    ) -> ConceptPlan:
        """Generate a concept learning plan.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            concept_id: Concept to plan for
            target_mastery: Goal mastery level
            context_obs: Additional context for planning (may include
                current_mastery for downstream tools)
            
        Returns:
            ConceptPlan with learning steps
        """
        # Use the existing concept planner tool. The underlying
        # ConceptPlannerTool already takes ``target_mastery`` and an
        # arbitrary ``context_obs`` dict; if a downstream implementation
        # wants ``current_mastery`` it can read it from that dict.
        return self.concept_planner(
            user_id=user_id,
            session_id=session_id,
            concept_id=concept_id,
            target_mastery=target_mastery,
            context_obs=context_obs,
        )


class ResponseBuilder:
    """Builds frontend-compatible responses from environment outputs.
    
    Converts environment transition outputs into the JSON structure
    expected by the frontend.
    """
    
    @staticmethod
    def build_response(
        tutor_outputs: Dict[str, Any],
        session_env: Optional[SessionEnvironment] = None,
        concept_env: Optional[ConceptEnvironment] = None,
        tutor_env: Optional[TutorEnvironment] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build frontend response from environment outputs.
        
        Args:
            tutor_outputs: Outputs from tutor environment step
            session_env: Session environment (optional)
            concept_env: Concept environment (optional)
            tutor_env: Tutor environment (optional)
            **kwargs: Additional response fields
            
        Returns:
            Frontend-compatible response dictionary
        """
        response = {
            "messages": tutor_outputs.get("messages", []),
            "ui_mode": tutor_outputs.get("ui_mode", "buttons_only"),
            "button_options": tutor_outputs.get("button_options", ["Continue"]),
            "mcq_payload": None,
            "agent_action_mode": "environment_v1",
        }
        
        # Add debug info
        debug_info = {}
        
        if session_env:
            state = session_env.get_state()
            debug_info["session"] = {
                "current_concept_index": state.current_concept_index,
                "concepts_completed": state.concepts_completed,
                "turn_count": state.turn_count,
                "terminated": state.terminated,
            }
        
        if concept_env:
            state = concept_env.get_state()
            debug_info["concept"] = {
                "concept_id": state.concept_id,
                "current_step_index": state.current_step_index,
                "current_mastery": state.current_mastery,
                "target_mastery": state.target_mastery,
                "phase": state.phase,
            }
        
        if tutor_env:
            state = tutor_env.get_state()
            debug_info["tutor"] = {
                "last_action": state.last_action.value if state.last_action else None,
                "awaiting_user_input": state.awaiting_user_input,
                "turn_in_step": state.turn_in_step,
            }
        
        response["debug"] = debug_info
        
        # Merge any additional kwargs
        response.update(kwargs)
        
        return response
    
    @staticmethod
    def build_session_complete_response(
        session_env: SessionEnvironment,
    ) -> Dict[str, Any]:
        """Build response for session completion.
        
        Args:
            session_env: Completed session environment
            
        Returns:
            Completion response
        """
        state = session_env.get_state()
        
        message = f"Session complete! You studied {state.concepts_completed} concepts."
        
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": message,
                }
            ],
            "ui_mode": "free_text",
            "button_options": [],
            "mcq_payload": None,
            "agent_action_mode": "environment_v1",
            "debug": {
                "session_complete": True,
                "termination_reason": state.termination_reason,
                "concepts_completed": state.concepts_completed,
            }
        }
    
    @staticmethod
    def build_concept_complete_response(
        concept_env: ConceptEnvironment,
        next_concept: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build response for concept completion.
        
        Args:
            concept_env: Completed concept environment
            next_concept: Next concept ID if available
            
        Returns:
            Completion response
        """
        state = concept_env.get_state()
        
        if next_concept:
            message = f"Great work on {state.concept_id}! Moving to {next_concept}..."
        else:
            message = f"Completed {state.concept_id}!"
        
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": message,
                }
            ],
            "ui_mode": "buttons_only",
            "button_options": ["Continue"],
            "mcq_payload": None,
            "agent_action_mode": "environment_v1",
            "debug": {
                "concept_complete": True,
                "concept_id": state.concept_id,
                "final_mastery": state.current_mastery,
                "mastery_reached": state.is_mastery_reached(),
            }
        }
