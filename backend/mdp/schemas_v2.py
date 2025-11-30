"""
MDP Schemas for 3-Layer Hierarchical Tutoring System

Architecture:
    Level 1: Session Planner - Decides which concepts to cover in a session
    Level 2: Concept Planner - Creates pedagogical steps for a concept  
    Level 3: Tutor MDP - Decides pedagogical actions and generates responses (TRAINABLE)

This module defines the state, action, reward, and transition structures
for the Tutor MDP (Level 3) which will be trained using PPO.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS: Pedagogical Action Space
# =============================================================================

class PedagogicalAction(Enum):
    """
    Discrete action space for the Tutor MDP.
    These are the pedagogical strategies the tutor can choose.
    
    The policy learns to select the optimal action given the state.
    """
    # Scaffolding Actions (preferred for learning - avoid answer leakage)
    SOCRATIC_QUESTION = "socratic_question"    # Guide via questioning
    GIVE_HINT = "give_hint"                    # Partial reveal without full answer
    WORKED_EXAMPLE = "worked_example"          # Step-by-step demo of similar problem
    USE_ANALOGY = "use_analogy"                # Connect to familiar concept
    NUDGE = "nudge"                            # Gentle redirection when close
    
    # Direct Instruction Actions (when scaffolding is insufficient)
    EXPLAIN = "explain"                        # Clear, direct explanation
    CORRECT_MISCONCEPTION = "correct"          # Address specific error
    ELABORATE = "elaborate"                    # Add depth/detail
    SUMMARIZE = "summarize"                    # Consolidate learning
    
    # Assessment Actions
    CONCEPT_CHECK = "concept_check"            # Quick verification question
    CHALLENGE = "challenge"                    # Harder problem to test mastery
    REFLECT = "reflect"                        # Metacognitive prompt
    
    # Flow Control Actions
    ADVANCE_STEP = "advance"                   # Move to next plan step
    STAY_ON_STEP = "stay"                      # Remain on current step
    REPLAN = "replan"                          # Request new plan
    CONCLUDE = "conclude"                      # End session/concept
    
    @classmethod
    def scaffolding_actions(cls) -> List["PedagogicalAction"]:
        """Actions that guide without giving answers directly"""
        return [
            cls.SOCRATIC_QUESTION, cls.GIVE_HINT, cls.WORKED_EXAMPLE, 
            cls.USE_ANALOGY, cls.NUDGE, cls.CONCEPT_CHECK, cls.REFLECT
        ]
    
    @classmethod
    def instruction_actions(cls) -> List["PedagogicalAction"]:
        """Direct instruction actions"""
        return [cls.EXPLAIN, cls.CORRECT_MISCONCEPTION, cls.ELABORATE, cls.SUMMARIZE]
    
    @classmethod
    def flow_actions(cls) -> List["PedagogicalAction"]:
        """Flow control actions that don't generate content"""
        return [cls.ADVANCE_STEP, cls.STAY_ON_STEP, cls.REPLAN, cls.CONCLUDE]
    
    @classmethod
    def content_actions(cls) -> List["PedagogicalAction"]:
        """Actions that require generating response content"""
        return [a for a in cls if a not in cls.flow_actions()]
    
    @classmethod
    def from_string(cls, s: str) -> "PedagogicalAction":
        """Parse action from string with fuzzy matching"""
        if not s:
            return cls.EXPLAIN
        s_lower = s.lower().strip().replace("_", "").replace("-", "")
        
        # Direct matches
        for action in cls:
            if action.value.replace("_", "") == s_lower:
                return action
        
        # Fuzzy matches
        mapping = {
            "question": cls.SOCRATIC_QUESTION,
            "socratic": cls.SOCRATIC_QUESTION,
            "ask": cls.SOCRATIC_QUESTION,
            "hint": cls.GIVE_HINT,
            "example": cls.WORKED_EXAMPLE,
            "analogy": cls.USE_ANALOGY,
            "nudge": cls.NUDGE,
            "explain": cls.EXPLAIN,
            "correct": cls.CORRECT_MISCONCEPTION,
            "misconception": cls.CORRECT_MISCONCEPTION,
            "elaborate": cls.ELABORATE,
            "summary": cls.SUMMARIZE,
            "summarize": cls.SUMMARIZE,
            "check": cls.CONCEPT_CHECK,
            "verify": cls.CONCEPT_CHECK,
            "challenge": cls.CHALLENGE,
            "reflect": cls.REFLECT,
            "advance": cls.ADVANCE_STEP,
            "continue": cls.ADVANCE_STEP,
            "next": cls.ADVANCE_STEP,
            "stay": cls.STAY_ON_STEP,
            "retry": cls.STAY_ON_STEP,
            "replan": cls.REPLAN,
            "conclude": cls.CONCLUDE,
            "finish": cls.CONCLUDE,
            "end": cls.CONCLUDE,
        }
        
        for key, action in mapping.items():
            if key in s_lower:
                return action
        
        return cls.EXPLAIN  # Default fallback


class StudentIntent(Enum):
    """Classification of student message intent"""
    ANSWER = "answer"              # Student attempting to answer
    QUESTION = "question"          # Student asking a question
    ACKNOWLEDGE = "acknowledge"    # Student acknowledging (ok, got it)
    CONFUSION = "confusion"        # Student expressing confusion
    REQUEST_HELP = "request_help"  # Explicit help request
    OFF_TOPIC = "off_topic"        # Unrelated message
    GIVE_UP = "give_up"            # Student giving up
    CONTINUE = "continue"          # Student explicitly wants to continue/advance


class RecommendedAction(Enum):
    """Recommended flow action from input analysis"""
    ADVANCE = "advance"    # Move to next step (correct answer or explicit continue)
    REPLY = "reply"        # Reply to student without advancing (question, partial)
    STAY = "stay"          # Stay on current step (retry needed)
    REPLAN = "replan"      # Student is lost, need to replan


class CorrectnessLevel(Enum):
    """Classification of answer correctness"""
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    NOT_APPLICABLE = "na"


# =============================================================================
# DATA CLASSES: Student & Plan Structures
# =============================================================================

@dataclass
class StudentProfile:
    """Student information and preferences"""
    student_id: str
    mastery: float = 0.0
    confidence: float = 0.5
    recent_errors: List[str] = field(default_factory=list)
    learning_style: str = "balanced"  # visual, verbal, balanced


@dataclass
class PlanStep:
    """Single step in a concept plan (Level 2 output)"""
    step_id: int
    concept: str
    pedagogy: str  # The default pedagogy for this step
    content: Optional[str] = None  # Internal instruction/hint
    subgoal: Optional[str] = None  # What student should learn
    content_ref: Optional[str] = None  # Reference to source material


@dataclass
class Plan:
    """Concept-level plan (Level 2 output)"""
    plan_id: str
    steps: List[PlanStep]
    concept: str = ""
    target_mastery: float = 0.8
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionStep:
    """Step in session plan (Level 1 output)"""
    step_id: int
    concept_id: str
    concept_name: str
    reason: str
    status: str = "pending"
    estimated_duration: int = 15


@dataclass
class SessionPlan:
    """Session-level plan (Level 1 output)"""
    session_id: str
    steps: List[SessionStep]
    resource_ids: List[str]
    created_at: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# TUTOR MDP: State Definition
# =============================================================================

@dataclass
class StudentAnalysis:
    """Analysis of student's last message"""
    intent: StudentIntent = StudentIntent.ACKNOWLEDGE
    correctness: CorrectnessLevel = CorrectnessLevel.NOT_APPLICABLE
    correctness_score: float = 0.0  # Numeric: 0.0 (wrong) to 1.0 (correct)
    sentiment: str = "neutral"
    key_errors: List[str] = field(default_factory=list)
    retrieval_query: Optional[str] = None  # Legacy: single query string
    retrieval_queries: List[str] = field(default_factory=list)  # New: array of 2-3 word queries
    feedback_hint: str = ""  # Internal hint for response generation
    recommended_action: RecommendedAction = RecommendedAction.STAY  # Flow recommendation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "correctness": self.correctness.value,
            "correctness_score": self.correctness_score,
            "sentiment": self.sentiment,
            "key_errors": self.key_errors,
            "retrieval_query": self.retrieval_query,
            "retrieval_queries": self.retrieval_queries,
            "feedback_hint": self.feedback_hint,
            "recommended_action": self.recommended_action.value,
        }


@dataclass
class ConversationTurn:
    """Single turn in conversation history"""
    role: str  # "student" | "tutor"
    content: str
    timestamp: float = field(default_factory=time.time)
    action: Optional[str] = None  # For tutor turns
    analysis: Optional[Dict[str, Any]] = None  # For student turns
    mastery_delta: float = 0.0


@dataclass 
class TutorState:
    """
    Complete state for the Tutor MDP (Level 3).
    
    This is the internal state maintained by the orchestrator.
    The policy receives a subset of this as TutorObservation.
    """
    # === Session Context ===
    session_id: str = ""
    student_id: str = ""
    
    # === Concept Context ===
    concept_id: str = ""
    concept_prerequisites: List[str] = field(default_factory=list)
    
    # === Plan Context (from Level 2) ===
    plan: Optional[Plan] = None
    current_step_index: int = 0
    
    # === Mastery Tracking ===
    mastery_current: float = 0.0
    mastery_target: float = 0.8
    mastery_trajectory: List[float] = field(default_factory=list)  # Last N deltas
    
    # === Conversation Context ===
    turn_count: int = 0
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    
    # === Interaction Tracking ===
    hints_given: int = 0
    questions_asked: int = 0
    errors_tracked: List[str] = field(default_factory=list)
    consecutive_incorrect: int = 0
    
    # === RAG Context ===
    last_retrieved_chunks: List[str] = field(default_factory=list)
    last_retrieval_query: str = ""
    
    # === Session Plan Context (from Level 1) ===
    session_plan: Optional[SessionPlan] = None
    session_step_index: int = 0
    
    @property
    def current_step(self) -> Optional[PlanStep]:
        """Get current plan step"""
        if self.plan and 0 <= self.current_step_index < len(self.plan.steps):
            return self.plan.steps[self.current_step_index]
        return None
    
    @property
    def steps_remaining(self) -> int:
        """Steps remaining in current plan"""
        if not self.plan:
            return 0
        return max(0, len(self.plan.steps) - self.current_step_index)
    
    @property
    def mastery_gap(self) -> float:
        """Gap to target mastery"""
        return max(0, self.mastery_target - self.mastery_current)
    
    @property
    def mastery_trend(self) -> str:
        """Trend in recent mastery changes"""
        if not self.mastery_trajectory:
            return "unknown"
        avg = sum(self.mastery_trajectory[-5:]) / len(self.mastery_trajectory[-5:])
        if avg > 0.03:
            return "improving"
        elif avg < -0.03:
            return "struggling"
        return "stable"
    
    def add_turn(self, role: str, content: str, action: str = None, 
                 analysis: Dict = None, mastery_delta: float = 0.0):
        """Add a conversation turn and maintain history window"""
        turn = ConversationTurn(
            role=role,
            content=content,
            action=action,
            analysis=analysis,
            mastery_delta=mastery_delta
        )
        self.conversation_history.append(turn)
        
        # Keep last 20 turns
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        # Track mastery trajectory
        if mastery_delta != 0:
            self.mastery_trajectory.append(mastery_delta)
            if len(self.mastery_trajectory) > 10:
                self.mastery_trajectory = self.mastery_trajectory[-10:]
        
        self.turn_count += 1
    
    def get_recent_history(self, n: int = 5) -> List[ConversationTurn]:
        """Get last N turns"""
        return self.conversation_history[-n:] if self.conversation_history else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging"""
        return {
            "session_id": self.session_id,
            "student_id": self.student_id,
            "concept_id": self.concept_id,
            "current_step_index": self.current_step_index,
            "steps_remaining": self.steps_remaining,
            "mastery_current": self.mastery_current,
            "mastery_target": self.mastery_target,
            "mastery_gap": self.mastery_gap,
            "mastery_trend": self.mastery_trend,
            "turn_count": self.turn_count,
            "hints_given": self.hints_given,
            "questions_asked": self.questions_asked,
            "consecutive_incorrect": self.consecutive_incorrect,
            "errors_tracked": self.errors_tracked[-3:],
        }


# =============================================================================
# TUTOR MDP: Observation (Policy Input)
# =============================================================================

@dataclass
class TutorObservation:
    """
    Observation provided to the Tutor Policy for action selection.
    
    This is a filtered/processed view of TutorState that the policy
    uses to decide which PedagogicalAction to take.
    """
    # === Concept Context ===
    concept_id: str = ""
    concept_prerequisites: List[str] = field(default_factory=list)
    
    # === Current Step Context ===
    step_pedagogy: str = ""  # Default pedagogy for current step
    step_content: str = ""   # Internal instruction for step
    step_subgoal: str = ""   # Learning objective
    step_index: int = 0
    steps_total: int = 0
    
    # === Student Message Analysis ===
    student_message: str = ""
    student_intent: StudentIntent = StudentIntent.ACKNOWLEDGE
    student_correctness: CorrectnessLevel = CorrectnessLevel.NOT_APPLICABLE
    correctness_score: float = 0.0
    
    # === Mastery Context ===
    mastery_current: float = 0.0
    mastery_target: float = 0.8
    mastery_trend: str = "unknown"
    
    # === Interaction Context ===
    turn_count: int = 0
    hints_given: int = 0
    questions_asked: int = 0
    consecutive_incorrect: int = 0
    recent_errors: List[str] = field(default_factory=list)
    
    # === Conversation Context (last few turns) ===
    recent_history: List[Dict[str, str]] = field(default_factory=list)
    
    # === RAG Context ===
    retrieved_context: str = ""
    
    # === Flow Recommendation (from InputAnalyzer) ===
    recommended_action: str = "stay"  # advance|reply|stay|replan
    feedback_hint: str = ""  # Internal hint for response generation

    def to_prompt_string(self) -> str:
        """
        Format observation for LLM policy input.
        This is what the model sees when deciding an action.
        """
        history_str = ""
        for turn in self.recent_history[-4:]:
            role = "Student" if turn.get("role") == "student" else "Tutor"
            history_str += f"  {role}: {turn.get('content', '')[:150]}\n"
        
        return f"""<|observation|>
CONCEPT: {self.concept_id}
PREREQUISITES: {', '.join(self.concept_prerequisites[:3]) if self.concept_prerequisites else 'none'}

CURRENT_STEP: {self.step_index + 1}/{self.steps_total}
STEP_PEDAGOGY: {self.step_pedagogy}
STEP_GOAL: {self.step_subgoal or self.step_content or 'not specified'}

STUDENT_MESSAGE: "{self.student_message}"
STUDENT_INTENT: {self.student_intent.value}
CORRECTNESS: {self.student_correctness.value} (score: {self.correctness_score:.2f})

MASTERY: {self.mastery_current:.2f} / {self.mastery_target:.2f} (trend: {self.mastery_trend})
TURN: {self.turn_count}
HINTS_GIVEN: {self.hints_given}
CONSECUTIVE_WRONG: {self.consecutive_incorrect}
RECENT_ERRORS: {', '.join(self.recent_errors[-2:]) if self.recent_errors else 'none'}

RECOMMENDED_FLOW: {self.recommended_action}
FEEDBACK_HINT: {self.feedback_hint[:100] if self.feedback_hint else 'none'}

RECENT_CONVERSATION:
{history_str if history_str else '  (session start)'}
<|/observation|>"""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/training"""
        return {
            "concept_id": self.concept_id,
            "step_index": self.step_index,
            "steps_total": self.steps_total,
            "step_pedagogy": self.step_pedagogy,
            "student_message": self.student_message,
            "student_intent": self.student_intent.value,
            "correctness": self.student_correctness.value,
            "correctness_score": self.correctness_score,
            "mastery_current": self.mastery_current,
            "mastery_target": self.mastery_target,
            "mastery_trend": self.mastery_trend,
            "turn_count": self.turn_count,
            "hints_given": self.hints_given,
            "consecutive_incorrect": self.consecutive_incorrect,
            "recommended_action": self.recommended_action,
            "feedback_hint": self.feedback_hint[:100] if self.feedback_hint else "",
        }

    @classmethod
    def from_state(cls, state: TutorState, analysis: StudentAnalysis, 
                   student_message: str, retrieved_context: str = "") -> "TutorObservation":
        """Create observation from state and analysis"""
        current_step = state.current_step
        
        # Build recent history
        recent = []
        for turn in state.get_recent_history(5):
            recent.append({
                "role": turn.role,
                "content": turn.content[:200],
            })
        
        return cls(
            concept_id=state.concept_id,
            concept_prerequisites=state.concept_prerequisites,
            step_pedagogy=current_step.pedagogy if current_step else "explain",
            step_content=current_step.content or "" if current_step else "",
            step_subgoal=getattr(current_step, 'subgoal', '') or "" if current_step else "",
            step_index=state.current_step_index,
            steps_total=len(state.plan.steps) if state.plan else 0,
            student_message=student_message,
            student_intent=analysis.intent,
            student_correctness=analysis.correctness,
            correctness_score=analysis.correctness_score,
            mastery_current=state.mastery_current,
            mastery_target=state.mastery_target,
            mastery_trend=state.mastery_trend,
            turn_count=state.turn_count,
            hints_given=state.hints_given,
            questions_asked=state.questions_asked,
            consecutive_incorrect=state.consecutive_incorrect,
            recent_errors=state.errors_tracked[-3:],
            recent_history=recent,
            retrieved_context=retrieved_context,
            recommended_action=analysis.recommended_action.value if hasattr(analysis.recommended_action, 'value') else str(analysis.recommended_action),
            feedback_hint=analysis.feedback_hint,
        )


# =============================================================================
# TUTOR MDP: Action (Policy Output)
# =============================================================================

@dataclass
class TutorAction:
    """
    Action output from the Tutor Policy.
    
    The policy outputs:
    1. A discrete pedagogical action
    2. Optional thinking trace (for interpretability)
    3. The generated response text
    """
    # === Core Action ===
    action: PedagogicalAction = PedagogicalAction.EXPLAIN
    
    # === Thinking Trace (for training/interpretability) ===
    thinking: str = ""  # Internal reasoning (hidden from student)
    
    # === Generated Response ===
    response_text: str = ""  # What the student sees
    
    # === Metadata ===
    confidence: float = 0.5
    retrieval_query: Optional[str] = None
    
    # === Logging ===
    raw_output: str = ""  # Full model output before parsing
    
    def to_training_format(self) -> str:
        """Format for SFT training target"""
        parts = []
        if self.thinking:
            parts.append(f"<think>\n{self.thinking}\n</think>")
        parts.append(f"[Action: {self.action.value}]")
        parts.append(self.response_text)
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging"""
        return {
            "action": self.action.value,
            "thinking": self.thinking[:200] if self.thinking else "",
            "response_text": self.response_text[:300],
            "confidence": self.confidence,
            "retrieval_query": self.retrieval_query,
        }
    
    @classmethod
    def from_raw_output(cls, raw: str) -> "TutorAction":
        """Parse action from model output"""
        import re
        
        thinking = ""
        action = PedagogicalAction.EXPLAIN
        response = raw
        
        # Extract thinking
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking = think_match.group(1).strip()
            response = raw[think_match.end():].strip()
        
        # Extract action
        action_match = re.search(
            r"\[(?:Action|Strategy):\s*(\w+)\]", 
            response, 
            re.IGNORECASE
        )
        if action_match:
            action = PedagogicalAction.from_string(action_match.group(1))
            response = response[action_match.end():].strip()
        
        return cls(
            action=action,
            thinking=thinking,
            response_text=response,
            raw_output=raw,
        )


# =============================================================================
# TUTOR MDP: Reward
# =============================================================================

@dataclass
class TutorReward:
    """
    Composite reward for a single tutor turn.
    
    Components:
    - mastery_delta: Change in student mastery (verifiable)
    - pedagogical_quality: Quality of teaching approach (LLM judge)
    - answer_leakage: Penalty for giving away answers (verifiable)
    - grounding: How well response is grounded in RAG (verifiable)
    - efficiency: Reward for reaching goals efficiently (verifiable)
    """
    # === Core Components ===
    mastery_delta: float = 0.0
    pedagogical_quality: float = 0.5
    no_answer_leakage: bool = True
    grounding_score: float = 0.5
    efficiency_score: float = 0.5
    
    # === Sub-components (for analysis) ===
    scaffolding_used: bool = False
    action_appropriate: bool = True
    response_concise: bool = True
    
    # === Weights ===
    weight_mastery: float = 0.35
    weight_pedagogical: float = 0.30
    weight_grounding: float = 0.15
    weight_efficiency: float = 0.20
    
    # === Penalties ===
    leakage_penalty: float = -0.5
    
    # === Metadata ===
    flags: List[str] = field(default_factory=list)
    
    def compute_total(self) -> float:
        """Compute weighted total reward"""
        # Hard penalty for answer leakage
        if not self.no_answer_leakage:
            self.flags.append("ANSWER_LEAKAGE")
            return max(-1.0, self.mastery_delta * 0.3 + self.leakage_penalty)
        
        # Weighted sum
        total = (
            self.mastery_delta * self.weight_mastery +
            self.pedagogical_quality * self.weight_pedagogical +
            self.grounding_score * self.weight_grounding +
            self.efficiency_score * self.weight_efficiency
        )
        
        # Bonus for scaffolding
        if self.scaffolding_used:
            total += 0.05
        
        return max(-1.0, min(1.0, total))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging"""
        return {
            "total": self.compute_total(),
            "mastery_delta": self.mastery_delta,
            "pedagogical_quality": self.pedagogical_quality,
            "no_answer_leakage": self.no_answer_leakage,
            "grounding_score": self.grounding_score,
            "efficiency_score": self.efficiency_score,
            "scaffolding_used": self.scaffolding_used,
            "flags": self.flags,
        }


# =============================================================================
# TUTOR MDP: Transition (for Training Data)
# =============================================================================

@dataclass
class TutorTransition:
    """
    Single transition in the Tutor MDP: (s, a, r, s')
    
    Used for:
    - Logging during interaction
    - Training data for PPO/SFT
    """
    transition_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    
    # === State (s) ===
    observation: Optional[TutorObservation] = None
    
    # === Action (a) ===
    action: Optional[TutorAction] = None
    
    # === Reward (r) ===
    reward: Optional[TutorReward] = None
    
    # === Next State Context ===
    mastery_before: float = 0.0
    mastery_after: float = 0.0
    step_before: int = 0
    step_after: int = 0
    
    # === Episode Info ===
    session_id: str = ""
    concept_id: str = ""
    turn_number: int = 0
    is_terminal: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/training"""
        return {
            "transition_id": self.transition_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "concept_id": self.concept_id,
            "turn_number": self.turn_number,
            "observation": self.observation.to_dict() if self.observation else None,
            "action": self.action.to_dict() if self.action else None,
            "reward": self.reward.to_dict() if self.reward else None,
            "mastery_before": self.mastery_before,
            "mastery_after": self.mastery_after,
            "mastery_delta": self.mastery_after - self.mastery_before,
            "step_before": self.step_before,
            "step_after": self.step_after,
            "is_terminal": self.is_terminal,
        }
    
    def to_ppo_format(self) -> Dict[str, Any]:
        """Format for PPO training"""
        return {
            "query": self.observation.to_prompt_string() if self.observation else "",
            "response": self.action.to_training_format() if self.action else "",
            "reward": self.reward.compute_total() if self.reward else 0.0,
        }


# =============================================================================
# INTERFACES: Policy Abstractions
# =============================================================================

class ConceptPolicy:
    """Abstract interface for Level 2: Concept Planner"""
    def generate_plan(self, state_c: Dict[str, Any]) -> Plan:
        raise NotImplementedError


class TutorPolicy:
    """
    Abstract interface for Level 3: Tutor Policy
    
    This is what we train with PPO.
    """
    def select_action(self, observation: TutorObservation) -> TutorAction:
        """Select pedagogical action given observation"""
        raise NotImplementedError
    
    def generate_response(self, observation: TutorObservation, 
                          action: PedagogicalAction,
                          retrieved_context: str = "") -> str:
        """Generate response text for selected action"""
        raise NotImplementedError


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def log_transition(transition: TutorTransition, logger: logging.Logger = logger):
    """Log a transition with structured data"""
    logger.info(
        "tutor_transition",
        extra={
            "transition_id": transition.transition_id,
            "session_id": transition.session_id,
            "turn": transition.turn_number,
            "action": transition.action.action.value if transition.action else None,
            "reward": transition.reward.compute_total() if transition.reward else None,
            "mastery_delta": transition.mastery_after - transition.mastery_before,
            "is_terminal": transition.is_terminal,
        }
    )


def create_observation_from_legacy(
    tutor_state: Dict[str, Any],
    analysis: Dict[str, Any],
    message: str,
    retrieved_context: str = ""
) -> TutorObservation:
    """
    Create TutorObservation from legacy state format.
    For backward compatibility with existing orchestrator.
    """
    plan = tutor_state.get("plan")
    current_idx = tutor_state.get("current_step_index", 0)
    
    step_pedagogy = ""
    step_content = ""
    steps_total = 0
    
    if plan and hasattr(plan, "steps") and plan.steps:
        steps_total = len(plan.steps)
        if current_idx < steps_total:
            step = plan.steps[current_idx]
            step_pedagogy = step.pedagogy
            step_content = step.content or ""
    
    # Parse analysis
    intent = StudentIntent.ACKNOWLEDGE
    correctness = CorrectnessLevel.NOT_APPLICABLE
    
    intent_str = analysis.get("intent", "").lower()
    if "answer" in intent_str:
        intent = StudentIntent.ANSWER
    elif "question" in intent_str:
        intent = StudentIntent.QUESTION
    elif "confusion" in intent_str:
        intent = StudentIntent.CONFUSION
    
    corr_str = analysis.get("correctness", "").lower()
    if "correct" in corr_str and "incorrect" not in corr_str:
        correctness = CorrectnessLevel.CORRECT
    elif "partial" in corr_str:
        correctness = CorrectnessLevel.PARTIAL
    elif "incorrect" in corr_str:
        correctness = CorrectnessLevel.INCORRECT
    
    return TutorObservation(
        concept_id=plan.steps[current_idx].concept if plan and plan.steps else "",
        step_pedagogy=step_pedagogy,
        step_content=step_content,
        step_index=current_idx,
        steps_total=steps_total,
        student_message=message,
        student_intent=intent,
        student_correctness=correctness,
        correctness_score=analysis.get("correctness_score", 0.0),
        mastery_current=analysis.get("current_mastery", 0.0),
        turn_count=len(tutor_state.get("student_response_history", [])),
        retrieved_context=retrieved_context,
    )
