from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import logging
from mdp.engine import Orchestrator
from mdp.planning import LLMConceptPolicy
from mdp.policies import SimpleTutorPolicy
from mdp.schemas import StudentProfile
from mdp.llm_client import LLMClient

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory session storage for MVP
# In a real app, use Redis or DB
SESSIONS: Dict[str, Orchestrator] = {}
PROFILES: Dict[str, StudentProfile] = {}

class StartSessionRequest(BaseModel):
    user_id: str
    concept: str
    mastery: float = 0.0

class ButtonRequest(BaseModel):
    session_id: str
    button: str  # 'continue' or 'replan'

@router.post("/api/mdp/start")
def start_session(req: StartSessionRequest):
    """Start a new MDP tutor session."""
    try:
        session_id = str(uuid.uuid4())
        
        profile = StudentProfile(student_id=req.user_id, mastery=req.mastery)
        PROFILES[session_id] = profile
        
        llm_client = LLMClient()
        concept_policy = LLMConceptPolicy(llm_client)
        tutor_policy = SimpleTutorPolicy()
        orch = Orchestrator(concept_policy, tutor_policy, llm_client)
        
        result = orch.start_session(profile, req.concept)
        SESSIONS[session_id] = orch
        
        return {"session_id": session_id, "result": result}
    except Exception as e:
        logger.exception("Failed to start MDP session")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/mdp/button")
def handle_button(req: ButtonRequest):
    """Handle student button interaction (continue/replan)."""
    try:
        session_id = req.session_id
        if session_id not in SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found")
        
        orch = SESSIONS[session_id]
        profile = PROFILES[session_id]
        
        result = orch.handle_button(req.button, profile)
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to handle button")
        raise HTTPException(status_code=500, detail=str(e))
