import os
import requests
import json
import logging
from typing import Dict, Any, Optional, List

from llm.common import call_json_chat

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model_override: Optional[str] = None, base_url_override: Optional[str] = None):
        self.base_url = base_url_override or os.getenv("OPENAI_API_BASE")
        self.api_key = os.getenv("OPENAI_API_KEY")
        # Allow model override, or use env var, or default
        self.model = model_override or os.getenv("LLM_MODEL_MINI") or "gpt-3.5-turbo"

        if self.base_url and not self.base_url.endswith("/v1"):
            self.base_url = self.base_url.rstrip("/") + "/v1"

    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7, json_mode: bool = False) -> Optional[Dict[str, Any]]:
        if not self.base_url or not self.api_key:
             logger.error("LLMClient: Missing base_url or api_key")
             return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"LLMClient Error: {e}")
            return None

    def call_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call the shared JSON-focused chat helper for planning.

        This delegates to llm.common.call_json_chat so that we benefit from
        consistent JSON extraction/repair and provider configuration.
        """
        try:
            return call_json_chat(
                user_prompt,
                default={},
                system_prompt=system_prompt or "Return ONLY minified JSON. No markdown.",
                model_hint=self.model,
                allow_text_fallback=False,
            )
        except Exception as e:
            logger.error(f"LLMClient.call_json_error: {e}")
            return {}
