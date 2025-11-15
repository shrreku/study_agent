from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_list(key: str, default: List[str], *, separator: str = ",") -> List[str]:
    raw = os.getenv(key)
    if not raw:
        return default
    parts = [item.strip() for item in raw.split(separator)]
    return [item for item in parts if item]


@dataclass(frozen=True)
class RewardWeights:
    stepwise_rubric: float = 0.0
    rubric: float = 0.4
    intent: float = 0.2
    gating: float = 0.2
    grounding: float = 0.15
    style: float = 0.05

    @classmethod
    def from_env(cls) -> "RewardWeights":
        return cls(
            stepwise_rubric=_env_float("TUTOR_RL_WEIGHT_STEPWISE", cls.stepwise_rubric),
            rubric=_env_float("TUTOR_RL_WEIGHT_RUBRIC", cls.rubric),
            intent=_env_float("TUTOR_RL_WEIGHT_INTENT", cls.intent),
            gating=_env_float("TUTOR_RL_WEIGHT_GATING", cls.gating),
            grounding=_env_float("TUTOR_RL_WEIGHT_GROUNDING", cls.grounding),
            style=_env_float("TUTOR_RL_WEIGHT_STYLE", cls.style),
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "stepwise_rubric": self.stepwise_rubric,
            "rubric": self.rubric,
            "intent": self.intent,
            "gating": self.gating,
            "grounding": self.grounding,
            "style": self.style,
        }

    def normalized(self) -> Dict[str, float]:
        weights = self.as_dict()
        total = sum(weights.values()) or 1.0
        return {name: value / total for name, value in weights.items()}


@dataclass(frozen=True)
class ValidatorThresholds:
    stepwise_rubric: float = 0.6
    rubric: float = 0.6
    intent: float = 0.6
    gating: float = 0.7
    grounding: float = 0.65
    style: float = 0.5

    @classmethod
    def from_env(cls) -> "ValidatorThresholds":
        return cls(
            stepwise_rubric=_env_float("TUTOR_RL_THRESHOLD_STEPWISE", cls.stepwise_rubric),
            rubric=_env_float("TUTOR_RL_THRESHOLD_RUBRIC", cls.rubric),
            intent=_env_float("TUTOR_RL_THRESHOLD_INTENT", cls.intent),
            gating=_env_float("TUTOR_RL_THRESHOLD_GATING", cls.gating),
            grounding=_env_float("TUTOR_RL_THRESHOLD_GROUNDING", cls.grounding),
            style=_env_float("TUTOR_RL_THRESHOLD_STYLE", cls.style),
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "stepwise_rubric": self.stepwise_rubric,
            "rubric": self.rubric,
            "intent": self.intent,
            "gating": self.gating,
            "grounding": self.grounding,
            "style": self.style,
        }


@dataclass(frozen=True)
class ValidatorConfig:
    banned_phrases: List[str] = field(
        default_factory=lambda: [
            "as an ai language model",
            "i am an ai",
        ]
    )
    suggestion_markers: List[str] = field(
        default_factory=lambda: [
            "try",
            "consider",
            "can you",
            "let's",
            "what about",
        ]
    )
    example_markers: List[str] = field(
        default_factory=lambda: [
            "for example",
            "for instance",
            "such as",
            "e.g.",
        ]
    )
    reasoning_markers: List[str] = field(
        default_factory=lambda: [
            "because",
            "therefore",
            "so that",
            "as a result",
        ]
    )
    min_words: int = 30
    max_words: int = 220
    advanced_term_penalty: float = 0.4
    direct_answer_markers: List[str] = field(
        default_factory=lambda: [
            "is",
            "are",
            "means",
            "refers",
            "defines",
        ]
    )
    thresholds: ValidatorThresholds = field(default_factory=ValidatorThresholds.from_env)

    @classmethod
    def from_env(cls) -> "ValidatorConfig":
        defaults = cls()
        return cls(
            banned_phrases=_env_list(
                "TUTOR_RL_BANNED_PHRASES",
                defaults.banned_phrases,
            ),
            suggestion_markers=_env_list(
                "TUTOR_RL_SUGGESTION_MARKERS",
                defaults.suggestion_markers,
            ),
            example_markers=_env_list(
                "TUTOR_RL_EXAMPLE_MARKERS",
                defaults.example_markers,
            ),
            reasoning_markers=_env_list(
                "TUTOR_RL_REASONING_MARKERS",
                defaults.reasoning_markers,
            ),
            min_words=_env_int("TUTOR_RL_MIN_WORDS", defaults.min_words),
            max_words=_env_int("TUTOR_RL_MAX_WORDS", defaults.max_words),
            advanced_term_penalty=_env_float(
                "TUTOR_RL_ADVANCED_TERM_PENALTY",
                defaults.advanced_term_penalty,
            ),
            direct_answer_markers=_env_list(
                "TUTOR_RL_DIRECT_MARKERS",
                defaults.direct_answer_markers,
            ),
            thresholds=ValidatorThresholds.from_env(),
        )


__all__ = [
    "RewardWeights",
    "ValidatorConfig",
    "ValidatorThresholds",
]

