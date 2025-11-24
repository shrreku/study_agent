from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TutorContext:
    """
    Context object holding the current state of the tutor session and user interaction.
    """
    session_id: str
    user_id: str
    policy_state: Any  # TutorSessionPolicy
    mastery_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    focus_concept: Optional[str] = None
    inferred_concept: Optional[str] = None
    concept_level: str = "unknown"
    turn_index: int = 0
    
    # Additional fields for observation
    student_message: str = ""
    recent_history: str = ""
    
    # Session-level context
    target_concepts: list = field(default_factory=list)
    
    # Button control state (from frontend payload)
    confirmed_action: Optional[str] = None
    step_control_type: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    
    def to_planning_observation(self) -> Dict[str, Any]:
        """Convert context to a dictionary suitable for planning prompts."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "focus_concept": self.focus_concept,
            "student_level": self.concept_level,
            "student_message": self.student_message,
            "recent_history": self.recent_history,
        }
    
    def get_control_signal(self) -> Optional[str]:
        """Extract the primary control signal from button payloads.
        
        Returns:
            'continue', 'replan_concept', or None
        """
        # Check step_control first (highest priority)
        if self.step_control_type:
            if self.step_control_type == "replan_concept":
                return "replan_concept"
            elif self.step_control_type in {"continue", "next"}:
                return "continue"
        
        # Check confirmed_action
        if self.confirmed_action:
            if self.confirmed_action in {"continue", "next", "yes"}:
                return "continue"
        
        return None
