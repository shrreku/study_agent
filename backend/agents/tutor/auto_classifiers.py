from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TurnSignals:
    """Turn-level student signals for auto/intelligent mode.

    These are intended to be produced by an LLM/json classifier but can also be
    approximated with lightweight heuristics in development.
    """

    student_confirmation: str = "not_confirmed"  # "confirmed" | "uncertain" | "not_confirmed"
    wants_orientation: bool = False
    wants_study_plan: bool = False
    wants_closure: bool = False
    reflection_provided: bool = False
    meta_intent: Optional[str] = None  # "ask_question" | "answer" | "meta_talk" | "off_topic" | ...


@dataclass
class PhaseSuggestion:
    """Suggested high-level learning phase/state for this turn."""

    suggested_state: str  # "orientation" | "teaching" | "assessment" | "review" | "closure"
    confidence: float = 0.5
    reason: str = ""


@dataclass
class ActionSuggestion:
    """Suggested next pedagogic action for this turn."""

    current_action_interpretation: Optional[str] = None
    next_action: str = "explain"  # "explain" | "ask" | "reflect" | "hint" | "review" | "orient"
    pedagogy_focus: Optional[List[str]] = None
    should_use_planning: bool = False
    should_use_multi_step: bool = False
    desired_state_after: Optional[str] = None


@dataclass
class RetrievalSuggestion:
    """Suggested retrieval/reuse strategy for this turn."""

    strategy: str = "fresh_retrieval"  # "reuse_last_chunks" | "fresh_retrieval" | "refine_query" | "no_retrieval"
    retrieval_query: Optional[str] = None
    max_chunks: int = 8
    prefer_sections: Optional[List[str]] = None
