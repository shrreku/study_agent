from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .state_machine import TutorState
from .planning import TutorPlan
from .auto_classifiers import (
    TurnSignals,
    PhaseSuggestion,
    ActionSuggestion,
    RetrievalSuggestion,
)


@dataclass
class TutorContext:
    """
    Unified context object for the Tutor Agent.
    
    Acts as the single source of truth for:
    - User input and session metadata
    - Student state (Mastery, Learning Path)
    - Perception (Classification)
    - Retrieved Knowledge
    - Conversation History
    """
    
    # Metadata
    session_id: str
    user_id: str
    turn_index: int
    
    # User Input
    message: str
    
    # Perception / Classification
    intent: str
    affect: str
    inferred_concept: Optional[str]
    
    # Session State
    current_state: TutorState
    focus_concept: Optional[str]
    concept_level: str
    
    # Knowledge State
    mastery_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    learning_path: List[str] = field(default_factory=list)
    
    # Retrieval
    retrieval_chunks: List[Dict[str, Any]] = field(default_factory=list)
    
    # History
    recent_turns: List[Dict[str, Any]] = field(default_factory=list)
    session_summary: str = ""
    recent_mcq_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Plan
    current_plan: Optional[TutorPlan] = None
    plan_step_index: int = 0
    
    # Flags
    cold_start_eligible: bool = False
    
    # Auto-conversational classifier outputs (optional)
    turn_signals: Optional[TurnSignals] = None
    phase_suggestion: Optional[PhaseSuggestion] = None
    action_suggestion: Optional[ActionSuggestion] = None
    retrieval_suggestion: Optional[RetrievalSuggestion] = None
    
    def to_policy_observation(self) -> Dict[str, Any]:
        """Serialize context for Policy LLM."""
        return {
            "current_message": self.message,
            "classifier_intent": self.intent,
            "classifier_affect": self.affect,
            "classifier_concept": self.inferred_concept,
            "classifier_confidence": 0.9,  # Simplified for now
            "focus_concept": self.focus_concept,
            "concept_level": self.concept_level,
            "mastery_map": self.mastery_map,
            "recent_dialogue": self.recent_turns[-3:],  # Short window for policy
            "learning_path": self.learning_path,
            "session_summary": self.session_summary,
            
            # Legacy/Placeholder fields for template compatibility
            "policy_phase": self.current_state.value,
            "last_action": "unknown",  # Ideally this should be tracked in state
            "consecutive_explains": 0,
            "target_concepts": [],
            "current_mastery": 0.0,
            "weak_concepts": [],
        }

    def to_planning_observation(self) -> Dict[str, Any]:
        """Serialize context for SRL Planner."""
        # Format history string
        history_str = f"Session Summary: {self.session_summary}\n" if self.session_summary else ""
        if self.recent_turns:
            lines = []
            for turn in self.recent_turns[-6:]:  # Longer window for planner
                role = turn.get("role", "user")
                text = (turn.get("text") or "").strip()
                if text:
                    lines.append(f"{role}: {text}")
            history_str += "\n".join(lines)

        return {
            "student_message": self.message,
            "intent": self.intent,
            "affect": self.affect,
            "focus_concept": self.focus_concept,
            "student_level": self.concept_level,
            "recent_history": history_str,
            "learning_path": self.learning_path,
            "chunks": self.retrieval_chunks[:5],
            "mastery_snapshot": self.mastery_map,
            "previous_action": "unknown", # TODO: plumb this through
            "recent_mcq_outcomes": self.recent_mcq_outcomes[-5:],
        }

