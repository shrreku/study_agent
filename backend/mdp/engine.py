from typing import Dict
import time
import logging
from mdp.schemas import ConceptPolicy, TutorPolicy, StudentProfile
from mdp.llm_client import LLMClient
from mdp.response_generator import ResponseGenerator
from mdp.rag import RAGTools

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, concept_policy: ConceptPolicy, tutor_policy: TutorPolicy, llm_client: LLMClient):
        self.concept_policy = concept_policy
        self.tutor_policy = tutor_policy
        self.llm_client = llm_client
        self.rag_tools = RAGTools()
        self.response_generator = ResponseGenerator(llm_client, self.rag_tools)
        self.tutor_state = {}
    
    def start_session(self, student_profile: StudentProfile, initial_concept: str):
        state_c = {"concept_id": initial_concept, "student_profile": student_profile.__dict__, "constraints": {"max_steps":4}}
        plan = self.concept_policy.generate_plan(state_c)
        self.tutor_state = {
            "plan": plan,
            "current_step_index": 0,
            "student_response_history": [],
            "awaiting_student_input": True
        }
        return self.render_current_step()

    def render_current_step(self):
        if self.tutor_state["current_step_index"] >= len(self.tutor_state["plan"].steps):
             return {"status": "done"}
             
        step = self.tutor_state["plan"].steps[self.tutor_state["current_step_index"]]
        
        # Use ResponseGenerator to render the response
        rendered = self.response_generator.generate_pedagogical_response(
            step_concept=step.concept,
            step_pedagogy=step.pedagogy,
            step_content=step.content,
            student_history=self.tutor_state.get("student_response_history", []),
            plan_index=self.tutor_state["current_step_index"],
            plan_length=len(self.tutor_state["plan"].steps)
        )
        
        return {
            "step": step, 
            "rendered_content": rendered.get("content"),
            "debug_json": rendered.get("raw_json"),
            "status": "rendered",
            "error": rendered.get("error")
        }

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
