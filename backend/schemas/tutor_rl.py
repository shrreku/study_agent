from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class RewardComponent(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)
    flags: List[str] = Field(default_factory=list)


class RewardPayload(BaseModel):
    components: Dict[str, RewardComponent]
    total: float = Field(ge=0.0, le=1.0)
    weights: Dict[str, float] = Field(default_factory=dict)
    normalized_weights: Dict[str, float] = Field(default_factory=dict)
    flags: List[str] = Field(default_factory=list)

    @field_validator("weights", "normalized_weights", mode="after")
    @classmethod
    def _non_negative(cls, value: Dict[str, float]) -> Dict[str, float]:
        for key, val in (value or {}).items():
            if val < 0:
                raise ValueError(f"weight {key} must be non-negative")
        return value


class CriticPayload(BaseModel):
    clarity: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    support: float = Field(ge=0.0, le=1.0)
    hallucination_flag: bool
    confidence: float = Field(ge=0.0, le=1.0)
    notes: Optional[str] = Field(default="")
    prompt_set: Optional[str] = None


class SFTMeta(BaseModel):
    prompt_set: Optional[str] = None
    source_chunk_ids: Optional[List[str]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class SFTRecord(BaseModel):
    observation: Dict[str, Any]
    action: Dict[str, Any]
    response: str
    reward: RewardPayload
    critic: Optional[CriticPayload] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class PreferenceCandidate(BaseModel):
    action: Dict[str, Any]
    response: str
    reward: Optional[RewardPayload] = None
    critic: Optional[CriticPayload] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class PreferenceDecision(BaseModel):
    chosen: int = Field(ge=0)
    scores: List[float]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: Optional[str] = None
    prompt_set: Optional[str] = None

    @field_validator("scores")
    @classmethod
    def _score_bounds(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("scores must contain at least one element")
        for idx, score in enumerate(value):
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"score[{idx}] must be between 0 and 1")
        return value


class PreferenceRecord(BaseModel):
    observation: Dict[str, Any]
    candidates: List[PreferenceCandidate]
    preference: PreferenceDecision
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidates")
    @classmethod
    def _at_least_two(cls, value: List[PreferenceCandidate]) -> List[PreferenceCandidate]:
        if len(value) < 2:
            raise ValueError("preference samples require at least two candidates")
        return value


__all__ = [
    "RewardComponent",
    "RewardPayload",
    "CriticPayload",
    "SFTRecord",
    "PreferenceCandidate",
    "PreferenceDecision",
    "PreferenceRecord",
]

