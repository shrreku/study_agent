from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TutorSessionPolicy:
    learning_path: List[str] = field(default_factory=list)
    focus_concept: Optional[str] = None
    focus_level: Optional[str] = None
    cold_start: bool = False
    cold_start_completed: List[str] = field(default_factory=list)
    consecutive_explains: int = 0
    last_action: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TutorSessionPolicy":
        if not data:
            return cls()
        learning_path = data.get("learning_path") or []
        if not isinstance(learning_path, list):
            learning_path = []
        completed = data.get("cold_start_completed") or []
        if not isinstance(completed, list):
            completed = []
        return cls(
            learning_path=list(learning_path),
            focus_concept=data.get("focus_concept"),
            focus_level=data.get("focus_level"),
            cold_start=bool(data.get("cold_start")),
            cold_start_completed=list(completed),
            consecutive_explains=int(data.get("consecutive_explains") or 0),
            last_action=data.get("last_action"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_path": self.learning_path,
            "focus_concept": self.focus_concept,
            "focus_level": self.focus_level,
            "cold_start": self.cold_start,
            "cold_start_completed": self.cold_start_completed,
            "consecutive_explains": self.consecutive_explains,
            "last_action": self.last_action,
        }

    def mark_cold_start(self, concept: Optional[str]) -> None:
        if concept and concept not in self.cold_start_completed:
            self.cold_start_completed.append(concept)
        self.cold_start = True

    def update_action(self, action: str) -> None:
        """Track consecutive explains to force assessment after 2+ in a row."""
        if action == "explain":
            if self.last_action == "explain":
                self.consecutive_explains += 1
            else:
                self.consecutive_explains = 1
        else:
            self.consecutive_explains = 0
        self.last_action = action


@dataclass
class TutorTurnParams:
    message: str
    user_id: str
    session_id: Optional[str] = None
    resource_id: Optional[str] = None
    target_concepts: List[str] = field(default_factory=list)
    session_policy: TutorSessionPolicy = field(default_factory=TutorSessionPolicy)
