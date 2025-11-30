"""
LLM Configuration System for MDP Components

Allows configuring different LLM models for different components:
- Planner: Generates lesson plans
- Policy: Selects pedagogical actions  
- ResponseGenerator: Generates tutor responses
- InputAnalyzer: Analyzes student messages
- StudentSimulator: Simulates student responses

Usage:
    config = LLMConfig(
        planner_model="openai/gpt-4o",
        policy_model="openai/gpt-4o-mini",
        response_model="openai/gpt-4o-mini",
        analyzer_model="google/gemini-2.5-flash-lite",
        student_model="google/gemini-2.0-flash-lite-001",
    )
    
    clients = LLMClientManager(config)
    planner_llm = clients.get_planner()
    analyzer_llm = clients.get_analyzer()
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM models used by different MDP components."""
    
    # Planner LLM - generates lesson plans (can use more capable model)
    planner_model: Optional[str] = None
    
    # Policy LLM - selects pedagogical actions
    policy_model: Optional[str] = None
    
    # Response Generator LLM - generates tutor responses
    response_model: Optional[str] = None
    
    # Input Analyzer LLM - analyzes student messages
    analyzer_model: Optional[str] = None
    
    # Student Simulator LLM - simulates student responses
    student_model: Optional[str] = None
    
    # Base URL override (if different from env)
    base_url: Optional[str] = None
    
    # Default model if specific model not set
    default_model: Optional[str] = None
    
    def __post_init__(self):
        """Set defaults from environment if not specified."""
        if self.default_model is None:
            self.default_model = os.getenv("LLM_MODEL_MINI") or "gpt-4o-mini"
        
        # Use default for any unset models
        if self.planner_model is None:
            self.planner_model = os.getenv("LLM_MODEL_PLANNER") or self.default_model
        if self.policy_model is None:
            self.policy_model = os.getenv("LLM_MODEL_POLICY") or self.default_model
        if self.response_model is None:
            self.response_model = os.getenv("LLM_MODEL_RESPONSE") or self.default_model
        if self.analyzer_model is None:
            self.analyzer_model = os.getenv("LLM_MODEL_ANALYZER") or self.default_model
        if self.student_model is None:
            self.student_model = os.getenv("LLM_MODEL_STUDENT") or "google/gemini-2.0-flash-lite-001"
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create config entirely from environment variables."""
        return cls()
    
    @classmethod
    def all_same(cls, model: str) -> "LLMConfig":
        """Create config with same model for all components."""
        return cls(
            planner_model=model,
            policy_model=model,
            response_model=model,
            analyzer_model=model,
            student_model=model,
            default_model=model,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "planner": self.planner_model,
            "policy": self.policy_model,
            "response": self.response_model,
            "analyzer": self.analyzer_model,
            "student": self.student_model,
        }


class LLMClientManager:
    """
    Manages LLM clients for different MDP components.
    
    Each component can use a different model while sharing the same base URL/API key.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        from mdp.llm_client import LLMClient
        
        self.config = config or LLMConfig.from_env()
        self._clients: Dict[str, LLMClient] = {}
        
        logger.info(f"LLMClientManager initialized with config: {self.config.to_dict()}")
    
    def _get_or_create(self, model: str) -> "LLMClient":
        """Get existing client for model or create new one."""
        from mdp.llm_client import LLMClient
        
        if model not in self._clients:
            self._clients[model] = LLMClient(
                model_override=model,
                base_url_override=self.config.base_url,
            )
        return self._clients[model]
    
    def get_planner(self) -> "LLMClient":
        """Get LLM client for plan generation."""
        return self._get_or_create(self.config.planner_model)
    
    def get_policy(self) -> "LLMClient":
        """Get LLM client for policy action selection."""
        return self._get_or_create(self.config.policy_model)
    
    def get_response(self) -> "LLMClient":
        """Get LLM client for response generation."""
        return self._get_or_create(self.config.response_model)
    
    def get_analyzer(self) -> "LLMClient":
        """Get LLM client for input analysis."""
        return self._get_or_create(self.config.analyzer_model)
    
    def get_student(self) -> "LLMClient":
        """Get LLM client for student simulation."""
        return self._get_or_create(self.config.student_model)
    
    def get_default(self) -> "LLMClient":
        """Get LLM client with default model."""
        return self._get_or_create(self.config.default_model)


# Convenience function for quick setup
def create_llm_clients(
    planner: Optional[str] = None,
    policy: Optional[str] = None,
    response: Optional[str] = None,
    analyzer: Optional[str] = None,
    student: Optional[str] = None,
) -> LLMClientManager:
    """
    Create LLM client manager with specified models.
    
    Example:
        clients = create_llm_clients(
            planner="gpt-4o",
            analyzer="google/gemini-2.5-flash-lite",
            student="google/gemini-2.0-flash-lite-001",
        )
    """
    config = LLMConfig(
        planner_model=planner,
        policy_model=policy,
        response_model=response,
        analyzer_model=analyzer,
        student_model=student,
    )
    return LLMClientManager(config)
