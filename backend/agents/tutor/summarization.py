from __future__ import annotations

from typing import List
import logging

from llm import call_llm_json
from prompts import get as prompt_get, render as prompt_render

logger = logging.getLogger(__name__)

class HistorySummarizer:
    """Summarizes conversation history to maintain context in long sessions."""
    
    def __init__(self, model_hint: str = "mini"):
        self.template_name = "tutor.summarize_history"
        self.model_hint = model_hint

    def summarize(self, history_lines: List[str], current_summary: str = "") -> str:
        """
        Generate a summary of the provided history lines, 
        integrating with existing summary if present.
        """
        if not history_lines:
            return current_summary

        history_text = "\n".join(history_lines)
        if current_summary:
            context_text = f"Previous Summary:\n{current_summary}\n\nNew Lines:\n{history_text}"
        else:
            context_text = history_text
            
        try:
            template = prompt_get(self.template_name)
            prompt = prompt_render(template, {"history": context_text})
            
            # We use a cheaper model for summarization usually
            result = call_llm_json(prompt, model_hint=self.model_hint)
            return str(result.get("summary", current_summary))
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return current_summary

