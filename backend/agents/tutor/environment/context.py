"""
Simplified context models for the 3-layer MDP environment.

These are lightweight alternatives to the complex TutorContext,
designed specifically for the environment architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionContext:
    """Minimal context for session-level decisions.
    
    Contains only what's needed to manage concept sequencing
    and session-level goals.
    """
    
    session_id: str
    user_id: str
    target_concepts: List[str]
    current_concept_index: int = 0
    mastery_map: Dict[str, float] = field(default_factory=dict)
    session_strategy: str = "sequential"
    
    def get_current_concept(self) -> Optional[str]:
        """Get the concept currently being studied.
        
        Returns:
            Current concept ID or None if session complete
        """
        if 0 <= self.current_concept_index < len(self.target_concepts):
            return self.target_concepts[self.current_concept_index]
        return None
    
    def is_session_complete(self) -> bool:
        """Check if all concepts have been covered.
        
        Returns:
            True if session should end
        """
        return self.current_concept_index >= len(self.target_concepts)
    
    def advance_concept(self) -> bool:
        """Move to the next concept.
        
        Returns:
            True if advanced successfully, False if session complete
        """
        self.current_concept_index += 1
        return not self.is_session_complete()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "target_concepts": list(self.target_concepts),
            "current_concept_index": self.current_concept_index,
            "mastery_map": dict(self.mastery_map),
            "session_strategy": self.session_strategy,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionContext:
        """Deserialize from dictionary."""
        return cls(
            session_id=str(data.get("session_id", "")),
            user_id=str(data.get("user_id", "")),
            target_concepts=list(data.get("target_concepts", [])),
            current_concept_index=int(data.get("current_concept_index", 0)),
            mastery_map=dict(data.get("mastery_map", {})),
            session_strategy=str(data.get("session_strategy", "sequential")),
        )


@dataclass
class ConceptContext:
    """Minimal context for concept-level decisions.
    
    Focuses on plan execution and mastery tracking for one concept.
    """
    
    concept_id: str
    current_mastery: float = 0.0
    target_mastery: float = 0.8
    current_step_index: int = 0
    phase: str = "learning"  # learning, assessment, complete
    
    def is_mastery_reached(self) -> bool:
        """Check if target mastery has been achieved.
        
        Returns:
            True if mastery goal met
        """
        return self.current_mastery >= self.target_mastery
    
    def get_mastery_gap(self) -> float:
        """Calculate gap to target mastery.
        
        Returns:
            Distance to target (positive means more learning needed)
        """
        return self.target_mastery - self.current_mastery
    
    def advance_step(self) -> None:
        """Move to the next plan step."""
        self.current_step_index += 1
    
    def reset_plan(self) -> None:
        """Reset to beginning of plan."""
        self.current_step_index = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "concept_id": self.concept_id,
            "current_mastery": self.current_mastery,
            "target_mastery": self.target_mastery,
            "current_step_index": self.current_step_index,
            "phase": self.phase,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConceptContext:
        """Deserialize from dictionary."""
        return cls(
            concept_id=str(data.get("concept_id", "")),
            current_mastery=float(data.get("current_mastery", 0.0)),
            target_mastery=float(data.get("target_mastery", 0.8)),
            current_step_index=int(data.get("current_step_index", 0)),
            phase=str(data.get("phase", "learning")),
        )


@dataclass
class TutorContext:
    """Minimal context for tutor-level pedagogical actions.
    
    Contains just what's needed to execute one teaching step.
    """
    
    step_instruction: str
    step_type: str
    button_label: str = "Continue"
    awaiting_response: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_instruction": self.step_instruction,
            "step_type": self.step_type,
            "button_label": self.button_label,
            "awaiting_response": self.awaiting_response,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TutorContext:
        """Deserialize from dictionary."""
        return cls(
            step_instruction=str(data.get("step_instruction", "")),
            step_type=str(data.get("step_type", "explain")),
            button_label=str(data.get("button_label", "Continue")),
            awaiting_response=bool(data.get("awaiting_response", False)),
        )
