"""Tests for TutorConfig - configuration simplification."""

import os
import pytest
from backend.agents.tutor.config import (
    TutorConfig,
    get_tutor_config,
    reset_tutor_config,
    _infer_mode_from_legacy_env,
)


class TestTutorConfigModes:
    """Test each configuration mode."""

    def test_simple_mode(self):
        """Test simple mode configuration."""
        config = TutorConfig._create_simple_mode()

        assert config.mode == "simple"
        assert config.enable_llm_policy is False
        assert config.enable_srl_planning is False
        assert config.enable_multi_step_execution is False
        assert config.enable_debug_logging is False
        assert config.response_grounding_mode == "explicit_citation"
        assert config.enable_state_machine is True

    def test_intelligent_mode(self):
        """Test intelligent mode configuration (default)."""
        config = TutorConfig._create_intelligent_mode()

        assert config.mode == "intelligent"
        assert config.enable_llm_policy is True
        assert config.enable_srl_planning is True
        assert config.enable_multi_step_execution is False
        assert config.enable_debug_logging is False
        assert config.response_grounding_mode == "llm_integrated"
        assert config.enable_state_machine is True

    def test_step_by_step_mode(self):
        """Test step-by-step mode configuration."""
        config = TutorConfig._create_step_by_step_mode()

        assert config.mode == "step_by_step"
        assert config.enable_llm_policy is False
        assert config.enable_srl_planning is True
        assert config.enable_multi_step_execution is True
        assert config.enable_debug_logging is False
        assert config.response_grounding_mode == "llm_integrated"
        assert config.enable_state_machine is True

    def test_debug_mode(self):
        """Test debug mode configuration."""
        config = TutorConfig._create_debug_mode()

        assert config.mode == "debug"
        assert config.enable_llm_policy is True
        assert config.enable_srl_planning is True
        assert config.enable_multi_step_execution is False
        assert config.enable_debug_logging is True
        assert config.response_grounding_mode == "llm_integrated"
        assert config.enable_state_machine is True

    # ===== FROM_ENV TESTS =====

    def test_from_env_uses_tutor_mode(self, monkeypatch):
        """Test that TUTOR_MODE environment variable is used."""
        monkeypatch.setenv("TUTOR_MODE", "simple")

        config = TutorConfig.from_env()

        assert config.mode == "simple"
        assert config.enable_llm_policy is False

    def test_from_env_intelligent_default(self, monkeypatch):
        """Test that intelligent is the default mode."""
        # Clear any TUTOR_MODE
        monkeypatch.delenv("TUTOR_MODE", raising=False)
        # Clear legacy vars
        monkeypatch.delenv("TUTOR_LLM_POLICY_ENABLED", raising=False)
        monkeypatch.delenv("TUTOR_SRL_MODE", raising=False)
        monkeypatch.delenv("TUTOR_SRL_PLANNING_ENABLED", raising=False)

        config = TutorConfig.from_env()

        assert config.mode == "intelligent"

    def test_from_env_case_insensitive(self, monkeypatch):
        """Test that TUTOR_MODE is case-insensitive."""
        monkeypatch.setenv("TUTOR_MODE", "INTELLIGENT")

        config = TutorConfig.from_env()

        assert config.mode == "intelligent"

    def test_from_env_whitespace_trimmed(self, monkeypatch):
        """Test that whitespace in TUTOR_MODE is trimmed."""
        monkeypatch.setenv("TUTOR_MODE", "  step_by_step  ")

        config = TutorConfig.from_env()

        assert config.mode == "step_by_step"

    def test_from_env_invalid_mode_defaults_to_intelligent(self, monkeypatch):
        """Test that invalid mode defaults to intelligent."""
        monkeypatch.setenv("TUTOR_MODE", "invalid_mode")

        config = TutorConfig.from_env()

        assert config.mode == "intelligent"

    # ===== VALIDATION TESTS =====

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            TutorConfig(
                mode="invalid",
                enable_llm_policy=False,
                enable_srl_planning=False,
            )

    def test_invalid_grounding_mode_raises_error(self):
        """Test that invalid grounding mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid grounding mode"):
            TutorConfig(
                mode="simple",
                enable_llm_policy=False,
                enable_srl_planning=False,
                response_grounding_mode="invalid_grounding",
            )

    # ===== TO_DICT TESTS =====

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict() includes all configuration fields."""
        config = TutorConfig._create_intelligent_mode()

        config_dict = config.to_dict()

        assert config_dict["mode"] == "intelligent"
        assert "enable_llm_policy" in config_dict
        assert "enable_srl_planning" in config_dict
        assert "response_grounding_mode" in config_dict
        assert "enable_state_machine" in config_dict
        assert "mastery_update_threshold" in config_dict

    def test_to_dict_values_match_config(self):
        """Test that to_dict() values match config attributes."""
        config = TutorConfig._create_step_by_step_mode()

        config_dict = config.to_dict()

        assert config_dict["mode"] == config.mode
        assert config_dict["enable_llm_policy"] == config.enable_llm_policy
        assert config_dict["enable_srl_planning"] == config.enable_srl_planning
        assert config_dict["enable_multi_step_execution"] == config.enable_multi_step_execution
        assert config_dict["enable_debug_logging"] == config.enable_debug_logging
        assert config_dict["response_grounding_mode"] == config.response_grounding_mode

    # ===== LEGACY ENV VAR TESTS =====

    def test_infer_mode_intelligent_from_legacy(self, monkeypatch):
        """Test inferring intelligent mode from legacy env vars."""
        monkeypatch.setenv("TUTOR_LLM_POLICY_ENABLED", "true")
        monkeypatch.setenv("TUTOR_SRL_MODE", "true")
        monkeypatch.setenv("TUTOR_SRL_PLANNING_ENABLED", "true")

        mode = _infer_mode_from_legacy_env()

        assert mode == "intelligent"

    def test_infer_mode_step_by_step_from_legacy(self, monkeypatch):
        """Test inferring step-by-step mode from legacy env vars."""
        monkeypatch.setenv("TUTOR_SRL_PLANNING_ENABLED", "true")
        monkeypatch.setenv("TUTOR_SRL_MULTI_STEP_EXECUTE", "true")

        mode = _infer_mode_from_legacy_env()

        assert mode == "step_by_step"

    def test_infer_mode_defaults_to_intelligent(self, monkeypatch):
        """Test that inferred mode defaults to intelligent when no legacy vars."""
        # Clear any legacy vars
        for var in [
            "TUTOR_LLM_POLICY_ENABLED",
            "TUTOR_SRL_MODE",
            "TUTOR_SRL_PLANNING_ENABLED",
            "TUTOR_SRL_MULTI_STEP_EXECUTE",
            "TUTOR_COLD_START_ENABLED",
        ]:
            monkeypatch.delenv(var, raising=False)

        mode = _infer_mode_from_legacy_env()

        assert mode == "intelligent"

    def test_legacy_vars_trigger_from_env_fallback(self, monkeypatch):
        """Test that legacy vars are used when TUTOR_MODE is not set."""
        monkeypatch.delenv("TUTOR_MODE", raising=False)
        monkeypatch.setenv("TUTOR_SRL_PLANNING_ENABLED", "true")
        monkeypatch.setenv("TUTOR_SRL_MULTI_STEP_EXECUTE", "true")

        config = TutorConfig.from_env()

        # Should infer step_by_step from legacy vars
        assert config.mode == "step_by_step"

    def test_tutor_mode_takes_precedence_over_legacy(self, monkeypatch):
        """Test that TUTOR_MODE takes precedence over legacy vars."""
        monkeypatch.setenv("TUTOR_MODE", "simple")
        monkeypatch.setenv("TUTOR_LLM_POLICY_ENABLED", "true")
        monkeypatch.setenv("TUTOR_SRL_MODE", "true")

        config = TutorConfig.from_env()

        # Should use TUTOR_MODE, not infer from legacy
        assert config.mode == "simple"
        assert config.enable_llm_policy is False

    # ===== SINGLETON TESTS =====

    def test_get_tutor_config_returns_singleton(self, monkeypatch):
        """Test that get_tutor_config returns a singleton."""
        monkeypatch.setenv("TUTOR_MODE", "debug")
        reset_tutor_config()

        config1 = get_tutor_config()
        config2 = get_tutor_config()

        assert config1 is config2
        assert config1.mode == "debug"

    def test_get_tutor_config_loads_from_env(self, monkeypatch):
        """Test that get_tutor_config loads configuration from environment."""
        monkeypatch.setenv("TUTOR_MODE", "step_by_step")
        reset_tutor_config()

        config = get_tutor_config()

        assert config.mode == "step_by_step"
        assert config.enable_srl_planning is True

    def test_reset_tutor_config(self, monkeypatch):
        """Test that reset_tutor_config clears the singleton."""
        monkeypatch.setenv("TUTOR_MODE", "simple")
        reset_tutor_config()

        config1 = get_tutor_config()
        assert config1.mode == "simple"

        # Change env and reset
        monkeypatch.setenv("TUTOR_MODE", "intelligent")
        reset_tutor_config()

        config2 = get_tutor_config()
        assert config2.mode == "intelligent"
        assert config1 is not config2

    # ===== FEATURE FLAG TESTS =====

    def test_simple_mode_has_no_ai_features(self):
        """Test that simple mode disables all AI features."""
        config = TutorConfig._create_simple_mode()

        assert config.enable_llm_policy is False
        assert config.enable_srl_planning is False
        assert config.enable_multi_step_execution is False
        assert config.enable_debug_logging is False

    def test_intelligent_mode_has_all_ai_features(self):
        """Test that intelligent mode enables AI features."""
        config = TutorConfig._create_intelligent_mode()

        assert config.enable_llm_policy is True
        assert config.enable_srl_planning is True

    def test_step_by_step_planning_enabled(self):
        """Test that step-by-step enables planning but not policy."""
        config = TutorConfig._create_step_by_step_mode()

        assert config.enable_llm_policy is False
        assert config.enable_srl_planning is True
        assert config.enable_multi_step_execution is True

    def test_debug_mode_enables_logging(self):
        """Test that debug mode enables debug logging."""
        config = TutorConfig._create_debug_mode()

        assert config.enable_debug_logging is True

    # ===== GROUNDING MODE TESTS =====

    def test_simple_mode_explicit_citation_grounding(self):
        """Test that simple mode uses explicit citation grounding."""
        config = TutorConfig._create_simple_mode()

        assert config.response_grounding_mode == "explicit_citation"

    def test_intelligent_mode_llm_integrated_grounding(self):
        """Test that intelligent mode uses LLM-integrated grounding."""
        config = TutorConfig._create_intelligent_mode()

        assert config.response_grounding_mode == "llm_integrated"

    def test_step_by_step_llm_integrated_grounding(self):
        """Test that step-by-step uses LLM-integrated grounding."""
        config = TutorConfig._create_step_by_step_mode()

        assert config.response_grounding_mode == "llm_integrated"

    # ===== COLD START TESTS =====

    def test_all_modes_enable_cold_start(self):
        """Test that all modes enable cold start by default."""
        for mode_name in ["simple", "intelligent", "step_by_step", "debug"]:
            if mode_name == "simple":
                config = TutorConfig._create_simple_mode()
            elif mode_name == "intelligent":
                config = TutorConfig._create_intelligent_mode()
            elif mode_name == "step_by_step":
                config = TutorConfig._create_step_by_step_mode()
            else:
                config = TutorConfig._create_debug_mode()

            assert config.enable_cold_start is True

    # ===== STATE MACHINE TESTS =====

    def test_all_modes_enable_state_machine(self):
        """Test that all modes enable state machine."""
        for mode_name in ["simple", "intelligent", "step_by_step", "debug"]:
            if mode_name == "simple":
                config = TutorConfig._create_simple_mode()
            elif mode_name == "intelligent":
                config = TutorConfig._create_intelligent_mode()
            elif mode_name == "step_by_step":
                config = TutorConfig._create_step_by_step_mode()
            else:
                config = TutorConfig._create_debug_mode()

            assert config.enable_state_machine is True

    # ===== MASTERY TRACKING TESTS =====

    def test_mastery_thresholds_default(self):
        """Test that mastery thresholds have sensible defaults."""
        config = TutorConfig._create_intelligent_mode()

        assert config.mastery_update_threshold == 0.05
        assert config.mastery_quality_threshold == 0.6

    def test_citation_limit_default(self):
        """Test that citation limit has default."""
        config = TutorConfig._create_intelligent_mode()

        assert config.citation_limit == 3

    # ===== EDGE CASES =====

    def test_empty_tutor_mode_defaults_to_intelligent(self, monkeypatch):
        """Test that empty TUTOR_MODE defaults to intelligent."""
        monkeypatch.setenv("TUTOR_MODE", "")

        config = TutorConfig.from_env()

        assert config.mode == "intelligent"

    def test_none_tutor_mode_defaults_to_intelligent(self, monkeypatch):
        """Test that missing TUTOR_MODE defaults to intelligent."""
        monkeypatch.delenv("TUTOR_MODE", raising=False)

        config = TutorConfig.from_env()

        assert config.mode == "intelligent"

    def test_extra_params_can_be_stored(self):
        """Test that extra parameters can be stored."""
        config = TutorConfig(
            mode="intelligent",
            enable_llm_policy=True,
            enable_srl_planning=True,
            extra_params={"custom_key": "custom_value"},
        )

        assert config.extra_params["custom_key"] == "custom_value"
