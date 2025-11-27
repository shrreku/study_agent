from typing import Dict, Any
import logging
import prompts
from mdp.llm_client import LLMClient

logger = logging.getLogger(__name__)

class InputAnalyzer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def analyze(self, message: str, current_step: Any) -> Dict[str, Any]:
        template = prompts.get("tutor_rl.input_analysis")
        if not template:
            # Fallback
            template = """
            Analyze user input.
            Context: {{concept}}, {{pedagogy}}, {{content}}
            Message: "{{message}}"
            Return JSON: {"intent": "...", "correctness": "...", "feedback": "...", "recommended_action": "advance|reply|replan|stay", "retrieval_query": "search query for RAG (optional)"}
            """
        
        vars = {
            "concept": current_step.concept,
            "pedagogy": current_step.pedagogy,
            "content": current_step.content or "",
            "message": message
        }
        
        # Render prompt
        user_prompt = prompts.render(template, vars)
        
        # We need to make sure the template in baseline.yaml also supports recommended_action.
        # Since prompts.render is simple string replacement, we rely on the prompt text.
        # But wait, the template in input_analysis.py line 16 is a FALLBACK.
        # The actual prompt is likely in prompts/baseline.yaml (tutor_rl.input_analysis).
        # I need to update baseline.yaml as well.

        system_prompt = "You are a pedagogical text analyzer."
        
        try:
            logger.info(f"Analyzing input: {message[:50]}...")
            response = self.llm.call_json(system_prompt, user_prompt)
            
            # Normalize correctness to a float score
            c_str = str(response.get("correctness", "")).lower()
            score = 0.0
            if "incorrect" in c_str:
                score = 0.0
            elif "partially" in c_str or "partial" in c_str:
                score = 0.5
            elif "correct" in c_str:
                score = 1.0
            
            response["correctness_score"] = score
            
            # Default retrieval query if missing
            if not response.get("retrieval_query"):
                # If reply/question, defaults to message or concept
                # But we leave it empty to let downstream decide
                response["retrieval_query"] = None
            
            logger.info(f"Analysis result: {response}")
            return response
        except Exception as e:
            logger.error(f"Input analysis failed: {e}")
            return {
                "intent": "acknowledge", # Conservative fallback
                "correctness": None,
                "feedback": "",
                "sentiment": "neutral"
            }
