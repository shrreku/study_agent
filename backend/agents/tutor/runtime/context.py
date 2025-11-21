"""Data structures for tutor turn execution context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..state import TutorSessionPolicy
from ..policy_decision import TutorPolicyDecision
from ..planning import TutorPlan


@dataclass
class TurnContext:
    """Input context for a single tutor turn."""
    session_id: str
    user_id: str
    turn_index: int
    message: str
    target_concepts: List[str]
    resource_id: Optional[str]
    dry_run: bool
    emit_state_requested: bool
    payload: Dict[str, Any]


@dataclass
class ClassificationContext:
    """Classification results for a turn."""
    intent: str
    affect: str
    concept: Optional[str]
    confidence: Optional[float]


@dataclass
class ConceptContext:
    """Concept and mastery state for a turn."""
    focus_concept: Optional[str]
    concept_level: str
    learning_path: List[str]
    learning_targets: List[str]
    mastery_map: Dict[str, Dict[str, Any]]
    prereq_check: Optional[Any]


@dataclass
class RetrievalContext:
    """Retrieval results for a turn."""
    chunks: List[Dict[str, Any]]
    query: str
    pedagogy_roles: List[str]
    chunk_ids: List[str]


@dataclass
class DecisionContext:
    """Context for policy and action decision-making."""
    classification: ClassificationContext
    concepts: ConceptContext
    policy_state: TutorSessionPolicy
    policy_decision: Optional[TutorPolicyDecision]
    srl_mode: bool
    plan: Optional[TutorPlan]
    retrieval: RetrievalContext
