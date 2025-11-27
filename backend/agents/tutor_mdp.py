import logging
import time
from typing import Dict, Any, Optional
from mdp.engine import Orchestrator
from mdp.policies import ConversationalTutorPolicy
from mdp.planning import LLMConceptPolicy
from mdp.llm_client import LLMClient
from mdp.schemas import StudentProfile

logger = logging.getLogger(__name__)

# Global in-memory session store for MVP testing
# Format: {session_id: Orchestrator}
_SESSIONS: Dict[str, Orchestrator] = {}

def tutor_mdp_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for the new MDP-based tutor agent.
    
    Payload expected:
    - user_id: str
    - session_id: str (optional, if not provided, treated as new or stateless if we were fully stateless)
    - message: str (student input)
    - button: str (optional control signal: 'continue', 'replan', 'finish')
    - concept: str (optional, for starting a new session)
    """
    user_id = payload.get("user_id")
    session_id = payload.get("session_id", "default_session")
    message = payload.get("message", "")
    button = payload.get("button")
    concept = payload.get("concept")
    
    
    logger.info(f"tutor_mdp_agent request: session={session_id} user={user_id} msg={message} btn={button} concept={concept}")
    
    # 1. Get or Create Orchestrator
    if session_id not in _SESSIONS:
        logger.info(f"Creating new Orchestrator for session {session_id}")
        llm_client = LLMClient()
        # Initialize policies
        concept_policy = LLMConceptPolicy(llm_client)
        tutor_policy = ConversationalTutorPolicy()
        
        orchestrator = Orchestrator(concept_policy, tutor_policy, llm_client)
        _SESSIONS[session_id] = orchestrator
    else:
        orchestrator = _SESSIONS[session_id]
    
    # 2. If this is a new session (no tutor_state), require concept to start
    if not orchestrator.tutor_state and not concept:
        return {"error": "New session requires 'concept' parameter"}
    
    # 3. Start session if concept provided
    if concept:
        # Mock student profile for now
        profile = StudentProfile(student_id=user_id, mastery=0.0, confidence=0.5)
        response = orchestrator.start_session(profile, concept)
        return _format_response(response, orchestrator)
    
    # 4. Handle Input for existing session
    # Mock profile - in real app, fetch from DB
    profile = StudentProfile(student_id=user_id, mastery=0.5) 
    
    if button:
        response = orchestrator.handle_button(button, profile)
    elif message:
        response = orchestrator.handle_message(message, profile)
    else:
        # Just render current state if no input (e.g. refresh)
        response = orchestrator.render_current_step()
        
    return _format_response(response, orchestrator)

def _format_response(mdp_response: Dict[str, Any], orchestrator: Orchestrator) -> Dict[str, Any]:
    """Format MDP response for the frontend."""
    
    # Extract plan for visualization
    plan = orchestrator.tutor_state.get("plan")
    plan_data = None
    if plan:
        plan_data = {
            "steps": [
                {
                    "step_id": s.step_id,
                    "concept": s.concept,
                    "pedagogy": s.pedagogy,
                    "content": s.content,
                    "status": "current" if i == orchestrator.tutor_state.get("current_step_index") else "pending" if i > orchestrator.tutor_state.get("current_step_index") else "completed"
                }
                for i, s in enumerate(plan.steps)
            ]
        }
        
    return {
        "agent": "tutor-mdp",
        "response": mdp_response.get("rendered_content"), # The text to show
        "step": mdp_response.get("step"), # Raw step data
        "debug": mdp_response.get("debug_json"),
        "status": mdp_response.get("status"), # 'rendered' or 'done'
        "plan": plan_data,
        "error": mdp_response.get("error")
    }
