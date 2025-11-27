from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

# ------- Data Schemas -------
@dataclass
class StudentProfile:
    student_id: str
    mastery: float = 0.0
    confidence: float = 0.5
    recent_errors: List[str] = field(default_factory=list)


# ===== ENHANCED STATE FOR RL TRAINING =====

@dataclass
class Exchange:
    """Single conversation turn for history tracking"""
    role: str  # "student" | "tutor"
    content: str
    analysis: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    mastery_delta: float = 0.0


@dataclass
class TutorStateRL:
    """
    Rich state representation for RL training.
    Includes cognitive context, mastery trajectory, and conversation history.
    """
    # ===== COGNITIVE CONTEXT =====
    concept_id: str = ""
    concept_prerequisites: List[str] = field(default_factory=list)
    concept_depth: int = 0  # Distance from root in knowledge graph
    
    # ===== MASTERY STATE =====
    student_mastery: Dict[str, float] = field(default_factory=dict)
    mastery_trajectory: List[float] = field(default_factory=list)  # Recent deltas
    target_mastery: float = 0.8
    
    # ===== CONVERSATION STATE =====
    turn_count: int = 0
    last_n_exchanges: List[Exchange] = field(default_factory=list)
    student_errors: List[str] = field(default_factory=list)  # Tracked misconceptions
    hints_given: int = 0
    
    # ===== PLAN STATE =====
    plan: Optional["Plan"] = None
    plan_step_index: int = 0
    
    # ===== RAG CONTEXT =====
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    context_relevance_score: float = 0.0
    
    @property
    def steps_remaining(self) -> int:
        if not self.plan:
            return 0
        return len(self.plan.steps) - self.plan_step_index
    
    @property
    def current_mastery(self) -> float:
        return self.student_mastery.get(self.concept_id, 0.0)
    
    def add_exchange(self, role: str, content: str, analysis: Optional[Dict] = None, mastery_delta: float = 0.0):
        """Add exchange and maintain sliding window"""
        exchange = Exchange(
            role=role, 
            content=content, 
            analysis=analysis,
            mastery_delta=mastery_delta
        )
        self.last_n_exchanges.append(exchange)
        # Keep last 10 exchanges
        if len(self.last_n_exchanges) > 10:
            self.last_n_exchanges = self.last_n_exchanges[-10:]
        
        # Track mastery trajectory
        if mastery_delta != 0:
            self.mastery_trajectory.append(mastery_delta)
            if len(self.mastery_trajectory) > 5:
                self.mastery_trajectory = self.mastery_trajectory[-5:]
    
    def to_prompt_context(self) -> str:
        """Serialize state for LLM input"""
        lines = [
            f"<|state|>",
            f"Concept: {self.concept_id}",
            f"Prerequisites: {', '.join(self.concept_prerequisites[:3]) if self.concept_prerequisites else 'none'}",
            f"Mastery: {self.current_mastery:.2f} (target: {self.target_mastery})",
            f"Mastery trend: {self._mastery_trend()}",
            f"Turn: {self.turn_count}",
            f"Hints given: {self.hints_given}",
            f"Recent errors: {', '.join(self.student_errors[-3:]) if self.student_errors else 'none'}",
            f"Plan step: {self.plan_step_index + 1}/{len(self.plan.steps) if self.plan else 0}",
            f"<|/state|>"
        ]
        return "\n".join(lines)
    
    def _mastery_trend(self) -> str:
        if not self.mastery_trajectory:
            return "no data"
        avg = sum(self.mastery_trajectory) / len(self.mastery_trajectory)
        if avg > 0.05:
            return "improving"
        elif avg < -0.05:
            return "struggling"
        return "stable"


@dataclass
class TutorAction:
    """
    Hierarchical action with explicit reasoning trace.
    Format: <think>reasoning</think> [Strategy: X] Response
    """
    # Level 1: Internal reasoning (hidden from student)
    thinking: str = ""
    
    # Level 2: Pedagogical strategy selection
    strategy: str = "explain"  # PedagogicalStrategy value
    
    # Level 3: Actual response
    response_text: str = ""
    
    # Metadata
    retrieval_query: Optional[str] = None
    difficulty_level: str = "intermediate"
    confidence: float = 0.5
    
    def to_training_format(self) -> str:
        """Format for SFT training target"""
        parts = []
        if self.thinking:
            parts.append(f"<think>\n{self.thinking}\n</think>")
        parts.append(f"[Strategy: {self.strategy}]")
        parts.append(self.response_text)
        return "\n".join(parts)
    
    @classmethod
    def from_raw_output(cls, raw: str) -> "TutorAction":
        """Parse from model output"""
        import re
        
        thinking = ""
        strategy = "explain"
        response = raw
        
        # Extract thinking
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            response = raw[think_match.end():].strip()
        
        # Extract strategy
        strat_match = re.search(r"\[Strategy:\s*(\w+)\]", response, re.IGNORECASE)
        if strat_match:
            strategy = strat_match.group(1).lower()
            response = response[strat_match.end():].strip()
        
        return cls(thinking=thinking, strategy=strategy, response_text=response)

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

@dataclass
class SessionStep:
    step_id: int
    concept_id: str
    concept_name: str
    reason: str
    status: str = "pending"  # pending, completed, skipped
    estimated_duration: int = 15  # minutes

@dataclass
class SessionPlan:
    session_id: str
    steps: List[SessionStep]
    resource_ids: List[str]
    created_at: float
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
