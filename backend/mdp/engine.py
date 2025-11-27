from typing import Dict, Optional
import time
import logging
from mdp.schemas import ConceptPolicy, TutorPolicy, StudentProfile, SessionPlan
from mdp.llm_client import LLMClient
from mdp.response_generator import ResponseGenerator
from mdp.rag import RAGTools
from mdp.input_analysis import InputAnalyzer
from mdp.mastery import MasteryModel

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, concept_policy: ConceptPolicy, tutor_policy: TutorPolicy, llm_client: LLMClient):
        self.concept_policy = concept_policy
        self.tutor_policy = tutor_policy
        self.llm_client = llm_client
        self.rag_tools = RAGTools()
        self.response_generator = ResponseGenerator(llm_client, self.rag_tools)
        self.input_analyzer = InputAnalyzer(llm_client)
        self.mastery_model = MasteryModel()
        self.tutor_state = {}
    
    def _load_concept(self, concept: str, student_profile: StudentProfile):
        """Helper to load a new concept plan into the tutor state."""
        state_c = {"concept_id": concept, "student_profile": student_profile.__dict__, "constraints": {"max_steps": 4}}
        plan = self.concept_policy.generate_plan(state_c)
        
        # Update state for new concept, preserving session context if any
        self.tutor_state["plan"] = plan
        self.tutor_state["current_step_index"] = 0
        # Clear concept-level history but preserve session-level history
        if "session_history" not in self.tutor_state:
            self.tutor_state["session_history"] = []
        self.tutor_state["student_response_history"] = []
        self.tutor_state["awaiting_student_input"] = True

    def start_session(self, student_profile: StudentProfile, initial_concept: str):
        self.tutor_state = {
            "student_id": student_profile.student_id
        }
        self._load_concept(initial_concept, student_profile)
        return self.render_current_step()

    def start_with_plan(self, student_profile: StudentProfile, session_plan: SessionPlan):
        self.tutor_state = {
            "student_id": student_profile.student_id,
            "session_plan": session_plan,
            "session_step_index": 0
        }
        if not session_plan.steps:
             return {"status": "done", "error": "Empty session plan"}
        
        # Start first concept
        self._load_concept(session_plan.steps[0].concept_name, student_profile)
        return self.render_current_step()

    def _try_advance_session(self) -> bool:
        s_plan = self.tutor_state.get("session_plan")
        s_idx = self.tutor_state.get("session_step_index")
        
        if s_plan and s_idx is not None:
            # Save current concept history to session history before advancing
            if "student_response_history" in self.tutor_state:
                current_concept_history = {
                    "concept_index": s_idx,
                    "concept_name": s_plan.steps[s_idx].concept_name,
                    "history": self.tutor_state["student_response_history"].copy()
                }
                self.tutor_state["session_history"].append(current_concept_history)
            
            next_idx = s_idx + 1
            if next_idx < len(s_plan.steps):
                self.tutor_state["session_step_index"] = next_idx
                next_step = s_plan.steps[next_idx]
                
                # Reconstruct profile (basic)
                profile = StudentProfile(student_id=self.tutor_state["student_id"])
                
                try:
                    self._load_concept(next_step.concept_name, profile)
                    # Verify the concept plan was generated successfully
                    if self.tutor_state["plan"] and self.tutor_state["plan"].steps:
                        return True
                    else:
                        logger.error(f"Failed to generate plan for concept: {next_step.concept_name}")
                        return False
                except Exception as e:
                    logger.error(f"Error loading concept {next_step.concept_name}: {e}")
                    return False
        return False

    def render_current_step(self, feedback_override: Optional[str] = None, force_reply_mode: bool = False, retrieval_query: Optional[str] = None):
        if self.tutor_state["current_step_index"] >= len(self.tutor_state["plan"].steps):
             if self._try_advance_session():
                  next_concept = self.tutor_state["plan"].steps[0].concept # Concept from the NEW plan
                  return self.render_current_step(feedback_override=f"Great job! Moving on to the next concept: {next_concept}")
             return {"status": "done"}
             
        step = self.tutor_state["plan"].steps[self.tutor_state["current_step_index"]]
        
        # Fetch current mastery for context
        student_id = self.tutor_state.get("student_id")
        mastery_map = self.mastery_model.get_mastery(student_id, [step.concept])
        current_mastery = mastery_map.get(step.concept, 0.0)

        # Use ResponseGenerator to render the response
        rendered = self.response_generator.generate_pedagogical_response(
            step_concept=step.concept,
            step_pedagogy=step.pedagogy,
            step_content=step.content,
            student_history=self.tutor_state.get("student_response_history", []),
            plan_index=self.tutor_state["current_step_index"],
            plan_length=len(self.tutor_state["plan"].steps),
            feedback_override=feedback_override,
            force_reply_mode=force_reply_mode,
            student_mastery=current_mastery,
            retrieval_query=retrieval_query
        )
        
        return {
            "step": step, 
            "rendered_content": rendered.get("content"),
            "debug_json": rendered.get("raw_json"),
            "status": "rendered",
            "error": rendered.get("error")
        }

    def handle_message(self, message: str, student_profile: StudentProfile):
        # Guard: if session done
        if self.tutor_state["current_step_index"] >= len(self.tutor_state["plan"].steps):
            return {"status": "done"}
            
        current_step = self.tutor_state["plan"].steps[self.tutor_state["current_step_index"]]
        
        # 1. Analyze Message
        analysis = self.input_analyzer.analyze(message, current_step)
        
        # 2. Update Mastery
        # Fetch previous mastery first
        mastery_map_before = self.mastery_model.get_mastery(student_profile.student_id, [current_step.concept])
        mastery_before = mastery_map_before.get(current_step.concept, 0.0)

        # We use the step concept. Ideally we'd use concepts detected in analysis, but for MVP:
        c_score = analysis.get("correctness_score", 0.0)
        # Hints tracking not fully implemented yet, assume 0
        new_mastery = self.mastery_model.update_mastery(
            student_profile.student_id, 
            current_step.concept, 
            c_score, 
            hints_used=0
        )
        mastery_delta = new_mastery - mastery_before
        logger.info(f"Mastery update: {current_step.concept} -> {new_mastery:.2f} (delta: {mastery_delta:+.3f})")

        # 3. Tutor Policy Decision
        obs = {
            "analysis": analysis, 
            "message": message,
            "current_mastery": new_mastery # augment obs with mastery
        }
        decision = self.tutor_policy.decide(self.tutor_state, obs)
        
        logger.info("tutor_policy_decision_msg", extra={"state": self.tutor_state, "observation": obs, "decision": decision})

        action = decision.get("action")
        feedback = decision.get("feedback")
        
        # Extract retrieval query from analysis if available
        retrieval_query = analysis.get("retrieval_query")
        
        # Log interaction
        self.tutor_state["student_response_history"].append({
            "step_id": current_step.step_id,
            "message": message, 
            "analysis": analysis,
            "tutor_action": action,
            "ts": time.time(),
            "mastery_before": mastery_before,
            "mastery_after": new_mastery,
            "mastery_delta": mastery_delta
        })

        if action == "replan":
            # call concept policy with updated profile
            state_c = {
                "concept_id": current_step.concept,
                "student_profile": student_profile.__dict__,
                "constraints": {"max_steps": 4}
            }
            new_plan = self.concept_policy.generate_plan(state_c)
            self.tutor_state["plan"] = new_plan
            self.tutor_state["current_step_index"] = 0
            self.tutor_state["student_response_history"] = [] 
            return self.render_current_step(feedback_override="I've updated the lesson plan based on your input.")
            
        elif action == "continue":
            # Advance to next step
            self.tutor_state["current_step_index"] += 1
            # Clear retrieval query for next step unless we want to carry it over (usually not)
            return self.render_current_step(feedback_override=feedback)
            
        elif action == "stay":
            # Re-render current step with feedback
            return self.render_current_step(feedback_override=feedback, retrieval_query=retrieval_query)

        elif action == "reply_to_user":
            # Do NOT advance step, but generate a response focusing on the user's question/input
            # We treat this as a special "stay" where we force the pedagogy to be 'question_answering' or similar
            # For now, we just pass the feedback_override which contains the answer.
            return self.render_current_step(feedback_override=feedback, force_reply_mode=True, retrieval_query=retrieval_query)
            
        elif action == "finish":
            return {"status":"done"}
        
        return {"error": "unknown_action"}

    def handle_button(self, button: str, student_profile: StudentProfile):
        obs = {"button": button}
        decision = self.tutor_policy.decide(self.tutor_state, obs)
        logger.info("tutor_policy_decision", extra={"state": self.tutor_state, "observation": obs, "decision": decision})
        if decision["action"] == "replan":
            # call concept policy with updated profile
            current_step = self.tutor_state["plan"].steps[self.tutor_state["current_step_index"]]
            state_c = {
                "concept_id": current_step.concept,
                "student_profile": student_profile.__dict__,
                "constraints": {"max_steps": 4}
            }
            new_plan = self.concept_policy.generate_plan(state_c)
            self.tutor_state["plan"] = new_plan
            self.tutor_state["current_step_index"] = 0
            self.tutor_state["student_response_history"] = [] 
            return self.render_current_step()
        elif decision["action"] == "continue":
            # log and advance
            self.tutor_state["student_response_history"].append({
                "step_id": self.tutor_state["plan"].steps[self.tutor_state["current_step_index"]].step_id,
                "button": "continue", "ts": time.time()
            })
            self.tutor_state["current_step_index"] += 1
            if self.tutor_state["current_step_index"] >= len(self.tutor_state["plan"].steps):
                return {"status":"done"}
            return self.render_current_step()
        elif decision["action"] == "finish":
            return {"status":"done"}
        
        return {"error": "unknown_action"}
