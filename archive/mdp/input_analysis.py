from typing import Dict, Any
import logging
from prompts import get as prompt_get, render as prompt_render
from llm import call_llm_json

logger = logging.getLogger(__name__)

class InputAnalyzer:
    def __init__(self):
        pass

    def analyze(self, message: str, current_step: Any) -> Dict[str, Any]:
        template = prompt_get("tutor_rl.input_analysis")
        if not template:
            # Fallback
            template = """
            Analyze user input.
            Context: {{concept}}, {{pedagogy}}, {{content}}
            Message: "{{message}}"
            Return JSON: {"intent": "...", "correctness": "...", "feedback": "..."}
            """
        
        # Extract step details safely
        concept = getattr(current_step, "concept", "") if current_step else ""
        pedagogogy = getattr(current_step, "pedagogogy", "") if current_step else ""
        # Try to get content/instruction/subgoal
        content = getattr(current_step, "content", "") or getattr(current_step, "instruction", "") or getattr(current_step, "subgoal", "") or ""

        vars = {
            "concept": concept,
            "pedagogy": pedagogogy,
            "content": content,
            "message": message
        }
        
        # Render prompt
        prompt = prompt_render(template, vars)
        
        try:
            # Default schema structure
            default_response = {
                "intent": "acknowledge",
                "correctness": None,
                "feedback": "",
                "sentiment": "neutral"
            }
            
            logger.info(f"Analyzing input: {message[:50]}...")
            response = call_llm_json(prompt, default_response)
            logger.info(f"Analysis result: {response}")
            return response
        except Exception as e:
            logger.error(f"Input analysis failed: {e}")
            return {
                "intent": "acknowledge", 
                "correctness": None,
                "feedback": "",
                "sentiment": "neutral"
            }
