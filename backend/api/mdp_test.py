from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
from pydantic import BaseModel
import uuid

from mdp.engine import Orchestrator
from mdp.policies import SimpleTutorPolicy, ConversationalTutorPolicy
from mdp.planning import LLMConceptPolicy
from mdp.llm_client import LLMClient
from mdp.schemas import StudentProfile, SessionPlan, SessionStep

router = APIRouter()

# In-memory session store for prototype
# session_id -> Orchestrator
sessions: Dict[str, Orchestrator] = {}

class StartRequest(BaseModel):
    concept: str
    student_id: Optional[str] = "student_01"

class StartPlanRequest(BaseModel):
    session_plan: Dict[str, Any]
    student_id: Optional[str] = "student_01"

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ResetRequest(BaseModel):
    session_id: str

@router.post("/api/mdp/start")
async def start_session(req: StartRequest):
    session_id = str(uuid.uuid4())
    
    llm_client = LLMClient()
    concept_policy = LLMConceptPolicy(llm_client)
    
    # Use the new Conversational Policy we added
    tutor_policy = ConversationalTutorPolicy() 
    
    orchestrator = Orchestrator(concept_policy, tutor_policy, llm_client)
    
    profile = StudentProfile(student_id=req.student_id)
    
    result = orchestrator.start_session(profile, req.concept)
    
    sessions[session_id] = orchestrator
    
    return {
        "session_id": session_id,
        "result": result
    }

@router.post("/api/mdp/start_from_plan")
async def start_session_from_plan(req: StartPlanRequest):
    session_id = str(uuid.uuid4())
    
    # Convert dict to SessionPlan object
    plan_data = req.session_plan
    session_plan = SessionPlan(
        session_id=plan_data["session_id"],
        steps=[SessionStep(**step) for step in plan_data["steps"]],
        resource_ids=plan_data["resource_ids"],
        created_at=plan_data["created_at"],
        meta=plan_data.get("meta", {})
    )
    
    llm_client = LLMClient()
    concept_policy = LLMConceptPolicy(llm_client)
    tutor_policy = ConversationalTutorPolicy()
    
    orchestrator = Orchestrator(concept_policy, tutor_policy, llm_client)
    
    profile = StudentProfile(student_id=req.student_id)
    
    result = orchestrator.start_with_plan(profile, session_plan)
    
    sessions[session_id] = orchestrator
    
    return {
        "session_id": session_id,
        "result": result
    }

@router.post("/api/mdp/chat")
async def chat_session(req: ChatRequest):
    session_id = req.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    orchestrator = sessions[session_id]
    
    # Retrieve student_id from the active session state
    stored_student_id = orchestrator.tutor_state.get("student_id", "student_01")
    profile = StudentProfile(student_id=stored_student_id)
    
    result = orchestrator.handle_message(req.message, profile)
    
    return {
        "session_id": session_id,
        "result": result
    }

@router.get("/api/mdp/session/{session_id}")
async def get_session_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    orchestrator = sessions[session_id]
    # Just render the current step without advancing or processing input
    result = orchestrator.render_current_step()
    
    return {
        "session_id": session_id,
        "result": result
    }

@router.post("/api/mdp/reset")
async def reset_session(req: ResetRequest):
    if req.session_id in sessions:
        del sessions[req.session_id]
    return {"status": "ok"}
