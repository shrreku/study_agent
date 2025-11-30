"""
MDP v2 API Router

This router uses the new TutorOrchestrator (engine_v2) with:
- Explicit PedagogicalAction selection
- Transition logging for PPO training
- Rule-based or LLM-based policies

The flow:
1. /api/mdp/v2/start - Start session with a concept
2. /api/mdp/v2/start_from_plan - Start from session plan
3. /api/mdp/v2/chat - Handle student messages
4. /api/mdp/v2/button - Handle button clicks (continue, replan)
5. /api/mdp/v2/transitions - Get logged transitions
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# MDP v2 imports
from mdp.schemas_v2 import (
    TutorState,
    TutorObservation,
    TutorAction,
    TutorTransition,
    PedagogicalAction,
    StudentProfile,
    Plan,
    PlanStep,
    SessionPlan,
    SessionStep,
)
from mdp.policies_v2 import (
    RuleBasedTutorPolicy,
    LLMTutorPolicy,
    HybridTutorPolicy,
    UnifiedLLMPolicy,
)
from mdp.trajectory import TrajectoryLogger
from mdp.llm_client import LLMClient
from mdp.planning import LLMConceptPolicy

router = APIRouter()
logger = logging.getLogger(__name__)

# =============================================================================
# IN-MEMORY SESSION STORE
# =============================================================================

class SessionState:
    """Wrapper for session state with trajectory logging"""
    
    def __init__(
        self, 
        session_id: str,
        llm_client: LLMClient,
        policy_type: str = "rule",
    ):
        self.session_id = session_id
        self.llm = llm_client
        self.concept_policy = LLMConceptPolicy(llm_client)
        
        # Choose policy based on policy_type
        if policy_type == "unified":
            self.tutor_policy = UnifiedLLMPolicy(llm_client)
        elif policy_type == "llm":
            self.tutor_policy = LLMTutorPolicy(llm_client)
        elif policy_type == "hybrid":
            self.tutor_policy = HybridTutorPolicy(llm_client)
        else:  # "rule" or default
            self.tutor_policy = RuleBasedTutorPolicy()
        
        # State
        self.state = TutorState(session_id=session_id)
        self.transitions: List[TutorTransition] = []
        self.trajectory_logger = TrajectoryLogger(
            output_dir="data/trajectories",
            session_id=session_id,
        )
        
        logger.info(f"session_created session_id={session_id} policy={type(self.tutor_policy).__name__}")
    
    def log_transition(self, transition: TutorTransition):
        """Log transition to memory and file"""
        self.transitions.append(transition)
        self.trajectory_logger.log(transition)
        logger.info(
            f"transition_logged session={self.session_id} "
            f"turn={transition.turn_number} "
            f"action={transition.action.action.value if transition.action else None} "
            f"reward={transition.reward.compute_total() if transition.reward else None:.3f}"
        )
    
    def close(self):
        """Flush and close trajectory logger"""
        self.trajectory_logger.close()


# Global session store
_sessions: Dict[str, SessionState] = {}


def get_session(session_id: str) -> SessionState:
    """Get session or raise 404"""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return _sessions[session_id]


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class StartRequest(BaseModel):
    concept: str
    student_id: Optional[str] = "student_01"
    use_llm_policy: Optional[bool] = False  # Deprecated, use policy_type
    policy_type: Optional[str] = "rule"  # "rule", "llm", "unified", "hybrid"


class StartPlanRequest(BaseModel):
    session_plan: Dict[str, Any]
    student_id: Optional[str] = "student_01"
    use_llm_policy: Optional[bool] = False  # Deprecated, use policy_type
    policy_type: Optional[str] = "rule"  # "rule", "llm", "unified", "hybrid"


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ButtonRequest(BaseModel):
    session_id: str
    button: str  # "continue" | "replan"


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post("/api/mdp/v2/start")
async def start_session(req: StartRequest):
    """Start a new tutoring session with a concept"""
    session_id = str(uuid.uuid4())
    
    # Determine policy type (backward compat: use_llm_policy -> "llm")
    policy_type = req.policy_type or ("llm" if req.use_llm_policy else "rule")
    
    logger.info(f"starting_session concept={req.concept} student={req.student_id} policy={policy_type}")
    
    try:
        llm_client = LLMClient()
        session = SessionState(session_id, llm_client, policy_type=policy_type)
        
        # Initialize state
        session.state.student_id = req.student_id
        session.state.concept_id = req.concept
        
        # Generate concept plan
        plan = _generate_concept_plan(session, req.concept)
        session.state.plan = plan
        session.state.current_step_index = 0
        
        # Store session
        _sessions[session_id] = session
        
        # Render first step
        result = _render_current_step(session)
        
        logger.info(f"session_started session_id={session_id} plan_steps={len(plan.steps)}")
        
        return {
            "session_id": session_id,
            "result": result,
        }
        
    except Exception as e:
        logger.exception(f"start_session_failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/mdp/v2/start_from_plan")
async def start_session_from_plan(req: StartPlanRequest):
    """Start a session from an existing session plan"""
    session_id = str(uuid.uuid4())
    
    try:
        plan_data = req.session_plan
        
        # Determine policy type
        policy_type = req.policy_type or ("llm" if req.use_llm_policy else "rule")
        
        logger.info(f"starting_from_plan steps={len(plan_data.get('steps', []))} student={req.student_id} policy={policy_type}")
        
        llm_client = LLMClient()
        session = SessionState(session_id, llm_client, policy_type=policy_type)
        
        # Parse session plan
        session_plan = SessionPlan(
            session_id=plan_data["session_id"],
            steps=[SessionStep(**step) for step in plan_data["steps"]],
            resource_ids=plan_data.get("resource_ids", []),
            created_at=plan_data.get("created_at", time.time()),
            meta=plan_data.get("meta", {}),
        )
        
        if not session_plan.steps:
            raise HTTPException(status_code=400, detail="Empty session plan")
        
        # Initialize state
        session.state.session_id = session_id
        session.state.student_id = req.student_id
        session.state.session_plan = session_plan
        session.state.session_step_index = 0
        
        # Start with first concept
        first_concept = session_plan.steps[0].concept_name
        session.state.concept_id = first_concept
        
        # Generate concept plan for first concept
        plan = _generate_concept_plan(session, first_concept)
        session.state.plan = plan
        session.state.current_step_index = 0
        
        # Store session
        _sessions[session_id] = session
        
        # Render first step
        result = _render_current_step(session)
        
        logger.info(f"session_from_plan_started session_id={session_id} concept={first_concept}")
        
        return {
            "session_id": session_id,
            "result": result,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"start_from_plan_failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/mdp/v2/chat")
async def chat_session(req: ChatRequest):
    """Handle student message - main MDP step"""
    session = get_session(req.session_id)
    
    logger.info(f"chat_message session={req.session_id} msg_len={len(req.message)}")
    
    try:
        # Record state before
        mastery_before = session.state.mastery_current
        step_before = session.state.current_step_index
        
        # 1. Analyze message
        from mdp.input_analysis import InputAnalyzer
        analyzer = InputAnalyzer(session.llm)
        current_step = session.state.current_step
        
        if not current_step:
            return {
                "session_id": req.session_id,
                "result": {"status": "done", "rendered_content": "Session complete."},
            }
        
        raw_analysis = analyzer.analyze(req.message, current_step)
        
        logger.info(
            f"message_analyzed intent={raw_analysis.get('intent')} "
            f"correctness={raw_analysis.get('correctness')} "
            f"recommended={raw_analysis.get('recommended_action')}"
        )
        
        # 2. Update mastery (simple heuristic)
        correctness_score = raw_analysis.get("correctness_score", 0.0)
        mastery_delta = correctness_score * 0.1 if correctness_score > 0.5 else -0.02
        session.state.mastery_current = max(0, min(1, session.state.mastery_current + mastery_delta))
        
        # Track errors
        if raw_analysis.get("correctness") == "incorrect":
            session.state.consecutive_incorrect += 1
        else:
            session.state.consecutive_incorrect = 0
        
        # 3. Create observation
        observation = _create_observation(session, req.message, raw_analysis)
        
        logger.info(
            f"observation_created concept={observation.concept_id} "
            f"step={observation.step_index + 1}/{observation.steps_total} "
            f"intent={observation.student_intent.value}"
        )
        
        # 4. Policy selects action
        action = session.tutor_policy.select_action(observation)
        
        logger.info(
            f"policy_action action={action.action.value} "
            f"thinking_len={len(action.thinking)} "
            f"confidence={action.confidence:.2f}"
        )
        
        # 5. Execute action
        result = _execute_action(session, action, observation, raw_analysis)
        
        # 6. Update conversation history
        session.state.add_turn("student", req.message, analysis=raw_analysis, mastery_delta=mastery_delta)
        session.state.add_turn("tutor", result.get("rendered_content", "")[:200], action=action.action.value)
        
        # 7. Compute reward
        from mdp.schemas_v2 import TutorReward
        reward = TutorReward(
            mastery_delta=mastery_delta,
            pedagogical_quality=0.7 if action.action in PedagogicalAction.scaffolding_actions() else 0.5,
            scaffolding_used=action.action in PedagogicalAction.scaffolding_actions(),
        )
        
        logger.info(f"reward_computed total={reward.compute_total():.3f} mastery_delta={mastery_delta:.3f}")
        
        # 8. Log transition
        transition = TutorTransition(
            session_id=req.session_id,
            concept_id=session.state.concept_id,
            turn_number=session.state.turn_count,
            observation=observation,
            action=action,
            reward=reward,
            mastery_before=mastery_before,
            mastery_after=session.state.mastery_current,
            step_before=step_before,
            step_after=session.state.current_step_index,
            is_terminal=result.get("status") == "done",
        )
        session.log_transition(transition)
        
        return {
            "session_id": req.session_id,
            "result": result,
        }
        
    except Exception as e:
        logger.exception(f"chat_failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/mdp/v2/button")
async def button_action(req: ButtonRequest):
    """Handle button click (continue, replan)"""
    session = get_session(req.session_id)
    
    logger.info(f"button_action session={req.session_id} button={req.button}")
    
    try:
        if req.button == "continue":
            session.state.current_step_index += 1
            
            # Check if concept is done
            if session.state.plan and session.state.current_step_index >= len(session.state.plan.steps):
                # Try advance to next concept in session plan
                if _try_advance_session(session):
                    result = _render_current_step(session)
                else:
                    result = {"status": "done", "rendered_content": "Session complete!"}
            else:
                result = _render_current_step(session)
                
        elif req.button == "replan":
            # Regenerate concept plan
            plan = _generate_concept_plan(session, session.state.concept_id)
            session.state.plan = plan
            session.state.current_step_index = 0
            result = _render_current_step(session)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown button: {req.button}")
        
        return {
            "session_id": req.session_id,
            "result": result,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"button_failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/mdp/v2/session/{session_id}")
async def get_session_state(session_id: str):
    """Get current session state"""
    session = get_session(session_id)
    
    result = _render_current_step(session)
    
    return {
        "session_id": session_id,
        "result": result,
        "state": session.state.to_dict(),
    }


@router.get("/api/mdp/v2/transitions/{session_id}")
async def get_transitions(session_id: str):
    """Get logged transitions for training"""
    session = get_session(session_id)
    
    return {
        "session_id": session_id,
        "count": len(session.transitions),
        "transitions": [t.to_dict() for t in session.transitions],
        "ppo_format": [t.to_ppo_format() for t in session.transitions],
    }


@router.post("/api/mdp/v2/reset")
async def reset_session(req: ButtonRequest):
    """Reset/close a session"""
    if req.session_id in _sessions:
        session = _sessions[req.session_id]
        session.close()
        del _sessions[req.session_id]
        logger.info(f"session_reset session_id={req.session_id}")
    
    return {"status": "ok"}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _generate_concept_plan(session: SessionState, concept: str) -> Plan:
    """Generate concept plan using Level 2 policy"""
    try:
        state_c = {
            "concept_id": concept,
            "student_profile": {"student_id": session.state.student_id},
            "constraints": {"max_steps": 5},
        }
        plan = session.concept_policy.generate_plan(state_c)
        logger.info(f"plan_generated concept={concept} steps={len(plan.steps)}")
        return plan
    except Exception as e:
        logger.warning(f"plan_generation_failed: {e}, using fallback")
        # Fallback plan
        return Plan(
            plan_id=str(uuid.uuid4())[:8],
            concept=concept,
            steps=[
                PlanStep(1, concept, "explain", "Introduce the concept"),
                PlanStep(2, concept, "example", "Show an example"),
                PlanStep(3, concept, "question", "Check understanding"),
            ],
        )


def _create_observation(
    session: SessionState, 
    message: str, 
    analysis: Dict[str, Any]
) -> TutorObservation:
    """Create TutorObservation from current state"""
    from mdp.schemas_v2 import StudentIntent, CorrectnessLevel
    
    current_step = session.state.current_step
    
    # Parse intent
    intent = StudentIntent.ACKNOWLEDGE
    intent_str = analysis.get("intent", "").lower()
    if "answer" in intent_str:
        intent = StudentIntent.ANSWER
    elif "question" in intent_str:
        intent = StudentIntent.QUESTION
    elif "confusion" in intent_str:
        intent = StudentIntent.CONFUSION
    elif "continue" in intent_str:
        intent = StudentIntent.CONTINUE
    
    # Parse correctness
    correctness = CorrectnessLevel.NOT_APPLICABLE
    corr_str = analysis.get("correctness", "").lower()
    if "correct" in corr_str and "incorrect" not in corr_str:
        correctness = CorrectnessLevel.CORRECT
    elif "partial" in corr_str:
        correctness = CorrectnessLevel.PARTIAL
    elif "incorrect" in corr_str:
        correctness = CorrectnessLevel.INCORRECT
    
    # Parse recommended_action (key for flow control!)
    recommended_action = analysis.get("recommended_action", "stay").lower()
    if recommended_action not in ["advance", "reply", "stay", "replan"]:
        recommended_action = "stay"
    
    # Build recent history
    recent_history = []
    for turn in session.state.get_recent_history(5):
        recent_history.append({
            "role": turn.role,
            "content": turn.content[:200],
        })
    
    # Safely get step attributes (PlanStep may have different fields depending on source)
    step_pedagogy = getattr(current_step, 'pedagogy', 'explain') if current_step else "explain"
    step_content = getattr(current_step, 'content', '') or "" if current_step else ""
    step_subgoal = getattr(current_step, 'subgoal', '') or "" if current_step else ""
    
    return TutorObservation(
        concept_id=session.state.concept_id,
        step_pedagogy=step_pedagogy,
        step_content=step_content,
        step_subgoal=step_subgoal,
        step_index=session.state.current_step_index,
        steps_total=len(session.state.plan.steps) if session.state.plan else 0,
        student_message=message,
        student_intent=intent,
        student_correctness=correctness,
        correctness_score=analysis.get("correctness_score", 0.0),
        mastery_current=session.state.mastery_current,
        mastery_target=session.state.mastery_target,
        mastery_trend=session.state.mastery_trend,
        turn_count=session.state.turn_count,
        hints_given=session.state.hints_given,
        questions_asked=session.state.questions_asked,
        consecutive_incorrect=session.state.consecutive_incorrect,
        recent_errors=session.state.errors_tracked[-3:],
        recent_history=recent_history,
        recommended_action=recommended_action,
        feedback_hint=analysis.get("feedback", ""),
    )


def _execute_action(
    session: SessionState,
    action: TutorAction,
    observation: TutorObservation,
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the selected action and generate response"""
    
    # Flow control actions
    if action.action == PedagogicalAction.ADVANCE_STEP:
        session.state.current_step_index += 1
        if session.state.plan and session.state.current_step_index >= len(session.state.plan.steps):
            if _try_advance_session(session):
                return _render_current_step(session)
            return {"status": "done", "rendered_content": "Great work! Concept complete."}
        return _render_current_step(session)
    
    elif action.action == PedagogicalAction.STAY_ON_STEP:
        return _render_current_step(session, feedback=analysis.get("feedback"))
    
    elif action.action == PedagogicalAction.REPLAN:
        plan = _generate_concept_plan(session, session.state.concept_id)
        session.state.plan = plan
        session.state.current_step_index = 0
        return _render_current_step(session, feedback="Let me adjust our approach.")
    
    elif action.action == PedagogicalAction.CONCLUDE:
        return {"status": "done", "rendered_content": "Session complete!"}
    
    # Content actions - generate response
    response_text = _generate_response_for_action(session, action, observation)
    
    # Update hints/questions counter
    if action.action in [PedagogicalAction.GIVE_HINT, PedagogicalAction.WORKED_EXAMPLE]:
        session.state.hints_given += 1
    if action.action in [PedagogicalAction.SOCRATIC_QUESTION, PedagogicalAction.CONCEPT_CHECK]:
        session.state.questions_asked += 1
    
    return {
        "status": "rendered",
        "rendered_content": response_text,
        "step": _step_to_dict(session.state.current_step),
        "action": action.action.value,
        "thinking": action.thinking,
        "debug": {
            "policy": type(session.tutor_policy).__name__,
            "mastery": session.state.mastery_current,
            "turn": session.state.turn_count,
        },
    }


def _generate_response_for_action(
    session: SessionState,
    action: TutorAction,
    observation: TutorObservation,
) -> str:
    """Generate response text for the action"""
    from mdp.response_generator import ResponseGenerator
    from mdp.rag import RAGTools
    
    # Use ResponseGenerator for actual content
    generator = ResponseGenerator(session.llm, RAGTools())
    
    current_step = session.state.current_step
    if not current_step:
        return "Let's continue learning."
    
    # Map action to pedagogy for ResponseGenerator
    pedagogy_map = {
        PedagogicalAction.EXPLAIN: "explain",
        PedagogicalAction.ELABORATE: "explain",
        PedagogicalAction.SOCRATIC_QUESTION: "question",
        PedagogicalAction.CONCEPT_CHECK: "question",
        PedagogicalAction.GIVE_HINT: "hint",
        PedagogicalAction.WORKED_EXAMPLE: "example",
        PedagogicalAction.USE_ANALOGY: "explain",
        PedagogicalAction.CORRECT_MISCONCEPTION: "explain",
        PedagogicalAction.SUMMARIZE: "explain",
        PedagogicalAction.REFLECT: "reflect",
        PedagogicalAction.CHALLENGE: "question",
        PedagogicalAction.NUDGE: "hint",
    }
    
    step_pedagogy = getattr(current_step, 'pedagogy', 'explain')
    pedagogy = pedagogy_map.get(action.action, step_pedagogy)
    
    # Safely get step attributes
    step_concept = getattr(current_step, 'concept', session.state.concept_id)
    step_content = getattr(current_step, 'content', '') or ""
    
    # Generate using existing system
    result = generator.generate_pedagogical_response(
        step_concept=step_concept,
        step_pedagogy=pedagogy,
        step_content=step_content,
        student_history=[],
        plan_index=session.state.current_step_index,
        plan_length=len(session.state.plan.steps) if session.state.plan else 1,
        feedback_override=observation.student_message,
        student_mastery=session.state.mastery_current,
    )
    
    content = result.get("content", "")
    
    logger.info(f"response_generated action={action.action.value} pedagogy={pedagogy} len={len(content)}")
    
    return content


def _render_current_step(session: SessionState, feedback: str = None) -> Dict[str, Any]:
    """Render current step content"""
    from mdp.response_generator import ResponseGenerator
    from mdp.rag import RAGTools
    
    if not session.state.plan or session.state.current_step_index >= len(session.state.plan.steps):
        return {"status": "done", "rendered_content": "Session complete."}
    
    step = session.state.current_step
    if not step:
        return {"status": "error", "error": "No current step"}
    
    generator = ResponseGenerator(session.llm, RAGTools())
    
    # Safely get step attributes
    step_concept = getattr(step, 'concept', session.state.concept_id)
    step_pedagogy = getattr(step, 'pedagogy', 'explain')
    step_content = getattr(step, 'content', '') or ""
    
    result = generator.generate_pedagogical_response(
        step_concept=step_concept,
        step_pedagogy=step_pedagogy,
        step_content=step_content,
        student_history=[],
        plan_index=session.state.current_step_index,
        plan_length=len(session.state.plan.steps),
        feedback_override=feedback,
        student_mastery=session.state.mastery_current,
    )
    
    logger.info(
        f"step_rendered step={session.state.current_step_index + 1}/{len(session.state.plan.steps)} "
        f"pedagogy={step_pedagogy}"
    )
    
    return {
        "status": "rendered",
        "rendered_content": result.get("content"),
        "step": _step_to_dict(step),
        "debug": result.get("raw_json"),
    }


def _step_to_dict(step) -> Optional[Dict[str, Any]]:
    """Convert PlanStep to dict - safely handles different PlanStep variants"""
    if not step:
        return None
    return {
        "step_id": getattr(step, 'step_id', 0),
        "concept": getattr(step, 'concept', ''),
        "pedagogy": getattr(step, 'pedagogy', 'explain'),
        "content": getattr(step, 'content', '') or "",
    }


def _try_advance_session(session: SessionState) -> bool:
    """Try to advance to next concept in session plan"""
    if not session.state.session_plan:
        return False
    
    next_idx = session.state.session_step_index + 1
    if next_idx >= len(session.state.session_plan.steps):
        return False
    
    session.state.session_step_index = next_idx
    next_step = session.state.session_plan.steps[next_idx]
    session.state.concept_id = next_step.concept_name
    
    # Generate new concept plan
    plan = _generate_concept_plan(session, next_step.concept_name)
    session.state.plan = plan
    session.state.current_step_index = 0
    
    logger.info(f"session_advanced new_concept={next_step.concept_name}")
    
    return True
