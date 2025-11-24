from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TutorSessionPolicy:
    """
    Persistent state for the tutor session policy.
    """
    session_plan: Dict[str, Any] = field(default_factory=dict)
    session_plan_index: int = 0
    session_strategy: str = "learning_path"
    
    concept_episode_id: Optional[str] = None
    concept_episode_mastery_start: Optional[float] = None
    srl_plan_step_index: int = 0
    
    # NEW: Persist concept plan
    concept_plan: Dict[str, Any] = field(default_factory=dict)
    
    quiz_phase: str = ""
    quiz_question_index: int = 0
    quiz_max_questions: int = 0
    concept_episode_quiz_correct: int = 0
    concept_episode_quiz_wrong: int = 0
    concept_episode_step_count: int = 0
    concept_episode_last_control_type: Optional[str] = None
