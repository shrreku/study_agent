from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import logging
from core.auth import require_auth
from mdp.session_planning import SessionPlanner
from mdp.schemas import StudentProfile, SessionPlan

router = APIRouter()
logger = logging.getLogger(__name__)

class CreatePlanRequest(BaseModel):
    resource_ids: Optional[List[str]] = None
    concept_ids: Optional[List[str]] = None
    user_id: Optional[str] = None

@router.post("/api/session/plan")
async def create_session_plan(
    body: CreatePlanRequest,
    token: str = Depends(require_auth)
):
    try:
        if not body.resource_ids and not body.concept_ids:
            raise HTTPException(status_code=400, detail="Either resource_ids or concept_ids must be provided")

        planner = SessionPlanner()
        
        # Mock profile for now or fetch from DB
        profile = StudentProfile(
            student_id=body.user_id or "current_user",
            mastery=0.5 # TODO: Fetch real mastery from DB/Service
        )
        
        plan = planner.generate_plan(
            student_profile=profile,
            resource_ids=body.resource_ids,
            concept_ids=body.concept_ids
        )
        return plan
        
    except Exception as e:
        logger.exception("create_session_plan_failed")
        raise HTTPException(status_code=500, detail=str(e))
