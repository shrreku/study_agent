"""
TutorConfig: Unified configuration for tutor behavior.

Consolidates 5+ environment variables into 1 simple TUTOR_MODE setting.
Provides backward compatibility with legacy env vars (with deprecation warnings).

Modes:
- "simple": Basic heuristics only (no LLM policy, no planning)
- "intelligent": LLM policy + SRL planning (default)
- "step_by_step": Step-by-step planning with explicit steps
- "debug": Verbose logging and detailed decision traces

Each mode maps to feature flags controlling:
- Enable LLM policy decisions
- Enable SRL planning
- Enable multi-step execution
- Response grounding mode (llm_integrated vs explicit_citation)
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class TutorConfig:
    """Unified tutor configuration."""
    
    mode: str  # "simple", "intelligent", "step_by_step", "debug"
    
    # Feature flags
    enable_llm_policy: bool = True
    enable_srl_planning: bool = True
    enable_multi_step_execution: bool = False
    enable_cold_start: bool = True
    enable_debug_logging: bool = False
    
    # Response configuration
    response_grounding_mode: str = "llm_integrated"  # or "explicit_citation"
    citation_limit: int = 3
    
    # State machine
    enable_state_machine: bool = True
    # Runtime selection / feature flags
    # When true, step-by-step sessions use the new MDP-oriented runtime_v2
    # orchestrator instead of the legacy step-by-step orchestrator.
    enable_step_mdp_runtime: bool = False
    
    # Mastery tracking
    mastery_update_threshold: float = 0.05
    mastery_quality_threshold: float = 0.6
    
    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_modes = {"simple", "intelligent", "step_by_step", "debug"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of: {valid_modes}")
        
        valid_grounding = {"llm_integrated", "explicit_citation"}
        if self.response_grounding_mode not in valid_grounding:
            raise ValueError(
                f"Invalid grounding mode '{self.response_grounding_mode}'. "
                f"Must be one of: {valid_grounding}"
            )
    
    @classmethod
    def from_env(cls) -> TutorConfig:
        """Load configuration from environment variables.
        
        Priority:
        1. TUTOR_MODE (new single var)
        2. Legacy env vars (with deprecation warnings)
        3. Defaults
        """
        mode = os.getenv("TUTOR_MODE", "").strip().lower()
        
        # Use legacy env vars if TUTOR_MODE not set
        if not mode:
            mode = _infer_mode_from_legacy_env()
        
        # Validate mode
        if mode not in {"simple", "intelligent", "step_by_step", "debug"}:
            mode = "intelligent"  # Default
        
        # Create config based on mode
        if mode == "simple":
            config = cls._create_simple_mode()
        elif mode == "intelligent":
            config = cls._create_intelligent_mode()
        elif mode == "step_by_step":
            config = cls._create_step_by_step_mode()
        elif mode == "debug":
            config = cls._create_debug_mode()
        else:
            config = cls._create_intelligent_mode()

        # Runtime v2 feature flag: enable MDP-oriented step-by-step runtime.
        raw_step_mdp = os.getenv("TUTOR_STEP_MDP_RUNTIME_ENABLED", "false").strip().lower()
        config.enable_step_mdp_runtime = raw_step_mdp in {"1", "true", "yes"}

        return config
    
    @classmethod
    def _create_simple_mode(cls) -> TutorConfig:
        """Simple mode: Basic heuristics only."""
        return cls(
            mode="simple",
            enable_llm_policy=False,
            enable_srl_planning=False,
            enable_multi_step_execution=False,
            enable_cold_start=True,
            enable_debug_logging=False,
            response_grounding_mode="explicit_citation",
            enable_state_machine=True,
        )
    
    @classmethod
    def _create_intelligent_mode(cls) -> TutorConfig:
        """Intelligent mode: LLM policy + SRL planning (default)."""
        return cls(
            mode="intelligent",
            enable_llm_policy=True,
            enable_srl_planning=True,
            enable_multi_step_execution=False,
            enable_cold_start=True,
            enable_debug_logging=False,
            response_grounding_mode="llm_integrated",
            enable_state_machine=True,
        )
    
    @classmethod
    def _create_step_by_step_mode(cls) -> TutorConfig:
        """Step-by-step mode: Planning with explicit step execution."""
        return cls(
            mode="step_by_step",
            enable_llm_policy=False,
            enable_srl_planning=True,
            enable_multi_step_execution=True,
            enable_cold_start=True,
            enable_debug_logging=False,
            response_grounding_mode="llm_integrated",
            enable_state_machine=True,
        )
    
    @classmethod
    def _create_debug_mode(cls) -> TutorConfig:
        """Debug mode: Same as intelligent with verbose logging."""
        return cls(
            mode="debug",
            enable_llm_policy=True,
            enable_srl_planning=True,
            enable_multi_step_execution=False,
            enable_cold_start=True,
            enable_debug_logging=True,
            response_grounding_mode="llm_integrated",
            enable_state_machine=True,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "mode": self.mode,
            "enable_llm_policy": self.enable_llm_policy,
            "enable_srl_planning": self.enable_srl_planning,
            "enable_multi_step_execution": self.enable_multi_step_execution,
            "enable_cold_start": self.enable_cold_start,
            "enable_debug_logging": self.enable_debug_logging,
            "response_grounding_mode": self.response_grounding_mode,
            "citation_limit": self.citation_limit,
            "enable_state_machine": self.enable_state_machine,
            "enable_step_mdp_runtime": self.enable_step_mdp_runtime,
            "mastery_update_threshold": self.mastery_update_threshold,
            "mastery_quality_threshold": self.mastery_quality_threshold,
        }
    
    def log_config(self) -> None:
        """Log current configuration."""
        logger.info(
            "tutor_config_initialized mode=%s grounding=%s",
            self.mode,
            self.response_grounding_mode,
            extra=self.to_dict(),
        )


def _infer_mode_from_legacy_env() -> str:
    """Infer TUTOR_MODE from legacy environment variables.
    
    This provides backward compatibility. If old env vars are set,
    we infer what mode they represent and log deprecation warnings.
    
    Returns: inferred mode string
    """
    llm_policy_enabled = os.getenv("TUTOR_LLM_POLICY_ENABLED", "").lower() == "true"
    srl_mode = os.getenv("TUTOR_SRL_MODE", "").lower() == "true"
    srl_planning_enabled = os.getenv("TUTOR_SRL_PLANNING_ENABLED", "").lower() == "true"
    multi_step = os.getenv("TUTOR_SRL_MULTI_STEP_EXECUTE", "").lower() == "true"
    
    # If any legacy vars are set, warn about deprecation
    legacy_vars = [
        "TUTOR_LLM_POLICY_ENABLED",
        "TUTOR_SRL_MODE",
        "TUTOR_SRL_PLANNING_ENABLED",
        "TUTOR_SRL_MULTI_STEP_EXECUTE",
        "TUTOR_COLD_START_ENABLED",
        "TUTOR_RESPONSE_GROUNDING_MODE",
    ]
    
    if any(os.getenv(var) for var in legacy_vars):
        warnings.warn(
            "Legacy tutor environment variables detected. "
            "Please migrate to TUTOR_MODE environment variable. "
            f"Set TUTOR_MODE to one of: simple, intelligent, step_by_step, debug. "
            f"Detected vars: {', '.join(v for v in legacy_vars if os.getenv(v))}"
        )
        logger.warning(
            "legacy_env_vars_detected vars=%s",
            [v for v in legacy_vars if os.getenv(v)],
        )
    
    # Infer mode from legacy vars
    if multi_step and srl_planning_enabled:
        logger.info("inferred_mode from legacy env mode=step_by_step")
        return "step_by_step"
    
    if llm_policy_enabled and srl_mode and srl_planning_enabled:
        logger.info("inferred_mode from legacy env mode=intelligent")
        return "intelligent"
    
    if llm_policy_enabled or srl_mode or srl_planning_enabled:
        logger.info("inferred_mode from legacy env mode=intelligent")
        return "intelligent"
    
    # Default
    logger.info("inferred_mode from legacy env mode=intelligent (default)")
    return "intelligent"


# Singleton instance for easy access
_config_instance: Optional[TutorConfig] = None


def get_tutor_config() -> TutorConfig:
    """Get or create the global tutor configuration."""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = TutorConfig.from_env()
        _config_instance.log_config()
    
    return _config_instance


def reset_tutor_config() -> None:
    """Reset the global configuration (mainly for testing)."""
    global _config_instance
    _config_instance = None

