from typing import Dict, Any, Optional
import logging
import prompts
from mdp.llm_client import LLMClient
from mdp.rag import RAGTools

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, llm_client: LLMClient, rag_tools: RAGTools):
        self.llm = llm_client
        self.rag = rag_tools

    def generate_pedagogical_response(self, 
                                      step_concept: str, 
                                      step_pedagogy: str, 
                                      step_content: str, 
                                      student_history: list,
                                      plan_index: int = 0,
                                      plan_length: int = 1) -> Dict[str, Any]:
        """
        Generate a response using specific prompts from baseline.yaml.
        """
        # Map step_pedagogy to baseline prompt keys
        # Keys available in baseline.yaml: tutor.explain, tutor.ask, tutor.hint, tutor.reflect
        key_map = {
            "explain": "tutor.explain",
            "question": "tutor.ask",
            "hint": "tutor.hint",
            "reflect": "tutor.reflect",
            "example": "tutor.explain", # Reuse explain for examples for MVP simplicity
            "summary": "tutor.explain"  # Reuse explain for summaries
        }
        
        prompt_key = key_map.get(step_pedagogy.lower(), "tutor.explain")
        template = prompts.get(prompt_key)
        
        if not template:
             return {"content": f"Error: Prompt {prompt_key} not found.", "error": "missing_prompt"}

        # Prepare Context
        # Combine step content, plan progress, and RAG
        combined_context = f"Lesson Step {plan_index+1} of {plan_length}: {step_content}"
        
        if len(combined_context) < 100:
             rag_context = self.rag.search_context(step_concept)
             if rag_context:
                 combined_context += f"\n\nReference Material:\n{rag_context}"

        # Format student history
        history_str = "None"
        if student_history:
             history_str = "\n".join([f"- {h.get('button', 'action')} at step {h.get('step_id')}" for h in student_history[-3:]])

        # Prepare variables for baseline prompts
        # They expect: {{concept}}, {{student_message}}, {{level}}, {{context}}
        # tutor.reflect also uses {{recent_history}}
        vars = {
            "concept": step_concept,
            "student_message": "continue", # Default since MDP drives the flow
            "level": "intermediate", # TODO: fetch from profile
            "context": combined_context,
            "recent_history": history_str
        }
        
        # Render
        user_prompt = prompts.render(template, vars)
        system_prompt = "You are an adaptive AI tutor."
        
        active_set = prompts.active_set()
        logger.info(f"generating_response_prompt key={prompt_key} set={active_set} len={len(user_prompt)}")

        # Call LLM
        try:
            response_json = self.llm.call_json(system_prompt, user_prompt)
            
            # Extract response
            # baseline prompts return {"response": ...} or {"question": ...}
            text_response = response_json.get("response") or response_json.get("question") or str(response_json)
            
            return {
                "content": text_response,
                "raw_json": response_json
            }
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {"content": step_content, "error": str(e)}
