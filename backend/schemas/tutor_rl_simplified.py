"""Simplified RL schemas with essential fields only, removing duplication."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SimplifiedObservation(BaseModel):
    """Compact observation with only essential context."""
    message: str
    user_id: str
    target_concepts: List[str]
    intent: str
    affect: str
    focus_concept: Optional[str]
    concept_level: str
    chunk_ids: List[str]
    pedagogy_roles: List[str]


class SimplifiedReward(BaseModel):
    """Compact reward with component scores and total."""
    rubric: float = Field(ge=0.0, le=1.0)
    intent: float = Field(ge=0.0, le=1.0)
    gating: float = Field(ge=0.0, le=1.0)
    grounding: float = Field(ge=0.0, le=1.0)
    style: float = Field(ge=0.0, le=1.0)
    total: float = Field(ge=0.0, le=1.0)
    flags: List[str] = Field(default_factory=list)


class SimplifiedCritic(BaseModel):
    """Compact critic scores."""
    clarity: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    support: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    hallucination: bool
    notes: str = ""


class SimplifiedSFTRecord(BaseModel):
    """Minimal SFT record for training."""
    observation: SimplifiedObservation
    action_type: str
    response: str
    reward: SimplifiedReward
    critic: SimplifiedCritic
    confidence: float = Field(ge=0.0, le=1.0)


class SimplifiedCandidate(BaseModel):
    """Minimal candidate for preference learning."""
    action_type: str
    response: str
    reward_total: float = Field(ge=0.0, le=1.0)
    critic_confidence: float = Field(ge=0.0, le=1.0)


class SimplifiedPreferenceRecord(BaseModel):
    """Minimal preference record."""
    observation: SimplifiedObservation
    candidates: List[SimplifiedCandidate]
    chosen_index: int = Field(ge=0)
    scores: List[float]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


__all__ = [
    "SimplifiedObservation",
    "SimplifiedReward",
    "SimplifiedCritic",
    "SimplifiedSFTRecord",
    "SimplifiedCandidate",
    "SimplifiedPreferenceRecord",
]

