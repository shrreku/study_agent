from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time

# ------- Data Schemas -------
@dataclass
class StudentProfile:
    student_id: str
    mastery: float = 0.0
    confidence: float = 0.5
    recent_errors: List[str] = field(default_factory=list)

@dataclass
class PlanStep:
    step_id: int
    concept: str
    pedagogy: str  # 'explain', 'example', 'question', 'hint', 'summary'
    content: Optional[str] = None
    content_ref: Optional[str] = None

@dataclass
class Plan:
    plan_id: str
    steps: List[PlanStep]
    meta: Dict[str, Any] = field(default_factory=dict)

# ------- Interfaces -------
class ConceptPolicy:
    """Abstract interface for concept-level policy"""
    def generate_plan(self, state_c: Dict[str, Any]) -> Plan:
        raise NotImplementedError

class TutorPolicy:
    """Abstract interface for tutor-level policy"""
    def decide(self, state_t: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
