from typing import Dict, Any, List
import time
import uuid
import logging
from mdp.schemas import ConceptPolicy, Plan, PlanStep
from mdp.llm_client import LLMClient
from mdp.rag import RAGTools
import prompts

logger = logging.getLogger(__name__)

class LLMConceptPolicy(ConceptPolicy):
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.rag = RAGTools()

    def generate_plan(self, state_c: Dict[str, Any]) -> Plan:
        concept = state_c["concept_id"]
        student_profile = state_c.get("student_profile", {})
        # Parse student profile dict if needed, or assume it's a dict
        current_mastery = float(student_profile.get("mastery", 0.0))
        target_mastery = 1.0
        mastery_gap = target_mastery - current_mastery
        
        max_steps = state_c.get("constraints", {}).get("max_steps", 4)
        
        # Fetch context using RAG (Vector + Graph)
        context, rag_data = self.rag.get_planning_context(concept)
        
        logger.info("RAG context retrieved", extra={
            "concept": concept, 
            "context_length": len(context),
            "vector_chunks_count": len(rag_data.get("vector_chunks", [])),
            "graph_records_count": len(rag_data.get("graph_records", []))
        })
        
        # Load prompt from prompts module (auto_conversational.yaml)
        # The prompt is nested under 'tutor_rl' in the yaml file
        template = prompts.get("tutor_rl.srl_concept_plan_v1")
        
        # Fallback if template not found (should not happen if files are correct)
        if not template:
            logger.warning("Prompt tutor_rl.srl_concept_plan_v1 not found, using simplified fallback prompt.")
            system_prompt = "You are an expert pedagogical planner."
            user_prompt = f"""
            Create a lesson plan for '{concept}'.
            Context: {context}
            Return JSON with 'steps' list. Each step: {{'concept': '{concept}', 'pedagogy': 'explain', 'content': '...'}}
            """
        else:
            system_prompt = "You are an expert pedagogical planner."
            vars = {
                "concept_id": concept,
                "current_mastery": f"{current_mastery:.2f}",
                "target_mastery": f"{target_mastery:.2f}",
                "mastery_gap": f"{mastery_gap:.2f}",
                "context_obs": context
            }
            user_prompt = prompts.render(template, vars)

        logger.info("generating_plan_prompt", extra={"concept": concept, "context_length": len(context)})
        
        response = self.llm.call_json(system_prompt, user_prompt)
        logger.info("plan_generation_response", extra={"response": response})
        
        steps_data = response.get("steps", [])
        steps = []
        
        if not steps_data:
            # Fallback if LLM fails
            logger.warning("LLM failed to generate plan, using fallback", extra={"concept": concept})
            steps.append(PlanStep(1, concept, "explain", content=f"Introduction to {concept}"))
            steps.append(PlanStep(2, concept, "example", content=f"Basic example of {concept}"))
            steps.append(PlanStep(3, concept, "question", content=f"Practice question for {concept}"))
        else:
            for i, s in enumerate(steps_data):
                # Map prompt step_type to schemas.PlanStep pedagogy
                # Prompt types: introduction, explanation, example, practice, reflection
                # Schema types: explain, example, question, hint, summary
                raw_type = s.get("step_type", "explanation").lower()
                pedagogy = "explain"
                if "example" in raw_type:
                    pedagogy = "example"
                elif "practice" in raw_type or "reflection" in raw_type or "question" in raw_type:
                    pedagogy = "question"
                elif "summary" in raw_type:
                    pedagogy = "summary"
                elif "hint" in raw_type:
                    pedagogy = "hint"
                
                # Content might come from instruction or content
                content = s.get("instruction") or s.get("content") or s.get("subgoal") or ""
                
                steps.append(PlanStep(
                    step_id=i+1,
                    concept=concept, # Prompt focuses on this concept
                    pedagogy=pedagogy,
                    content=content
                ))
        
        # Create plan and persist RAG data in meta
        meta = {
            "created_at": time.time(),
            "rag_data": rag_data  # Persist chunks/graph data
        }
        plan = Plan(plan_id=str(uuid.uuid4()), steps=steps, meta=meta)

        step_summaries = [
            {
                "step_id": step.step_id,
                "pedagogy": step.pedagogy,
                "content": (step.content[:160] if step.content else None),
            }
            for step in steps
        ]
        logger.info("generated_plan_steps concept=%s steps=%s", concept, step_summaries)

        return plan
