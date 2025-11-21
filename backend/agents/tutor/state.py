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
    phase: str = "teaching"
    pending_question_type: Optional[str] = None
    pending_concept: Optional[str] = None
    last_explained_concept: Optional[str] = None
    # Lightweight controls for how much conversational history to surface
    # to the policy and other tools. history_window is a small integer
    # (number of recent turns), history_focus can hint at which roles to
    # prioritize (e.g. "mixed", "student", "tutor", "minimal").
    history_window: int = 3
    history_focus: Optional[str] = None
    # Optional SRL plan state for step-by-step mode. We persist the last
    # generated plan and a cursor so that subsequent "continue" turns can
    # follow the same plan instead of re-planning.
    srl_plan: Optional[Dict[str, Any]] = None
    srl_plan_step_index: int = 0
    # Optional session-level plan for step-by-step mode. This captures
    # the ordered list of concepts for the current session and a cursor
    # over that list.
    session_plan: Optional[Dict[str, Any]] = None
    session_plan_index: int = 0
    session_strategy: Optional[str] = None
    # V2 Architecture: Explicit state machine data
    # Tracks tutor session state: orientation, teaching, assessment, review, closure
    state_machine: Optional[Dict[str, Any]] = None
    # Running summary of the session for context management
    session_summary: str = ""
    quiz_phase: str = ""
    quiz_question_index: int = 0
    quiz_max_questions: int = 0
    last_mcq: Optional[Dict[str, Any]] = None
    recent_mcq_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    concept_episode_id: Optional[str] = None
    concept_episode_concept: Optional[str] = None
    concept_episode_start_turn: int = 0
    concept_episode_mastery_start: Optional[float] = None
    concept_episode_step_count: int = 0
    concept_episode_quiz_correct: int = 0
    concept_episode_quiz_wrong: int = 0
    concept_episode_last_control_type: Optional[str] = None

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
        try:
            history_window = int(data.get("history_window") or 3)
        except Exception:
            history_window = 3
        history_focus = data.get("history_focus")
        srl_plan = data.get("srl_plan") if isinstance(data.get("srl_plan"), dict) else None
        try:
            srl_plan_step_index = int(data.get("srl_plan_step_index") or 0)
        except Exception:
            srl_plan_step_index = 0
        session_plan = data.get("session_plan") if isinstance(data.get("session_plan"), dict) else None
        try:
            session_plan_index = int(data.get("session_plan_index") or 0)
        except Exception:
            session_plan_index = 0
        session_strategy = data.get("session_strategy")
        state_machine = data.get("state_machine") if isinstance(data.get("state_machine"), dict) else None
        session_summary = str(data.get("session_summary") or "")
        quiz_phase = str(data.get("quiz_phase") or "")
        try:
            quiz_question_index = int(data.get("quiz_question_index") or 0)
        except Exception:
            quiz_question_index = 0
        try:
            quiz_max_questions = int(data.get("quiz_max_questions") or 0)
        except Exception:
            quiz_max_questions = 0
        last_mcq = data.get("last_mcq") if isinstance(data.get("last_mcq"), dict) else None
        recent_mcq_outcomes = data.get("recent_mcq_outcomes") or []
        if not isinstance(recent_mcq_outcomes, list):
            recent_mcq_outcomes = []
        concept_episode_id = data.get("concept_episode_id")
        concept_episode_concept = data.get("concept_episode_concept")
        try:
            concept_episode_start_turn = int(data.get("concept_episode_start_turn") or 0)
        except Exception:
            concept_episode_start_turn = 0
        raw_episode_mastery_start = data.get("concept_episode_mastery_start")
        if raw_episode_mastery_start in (None, ""):
            concept_episode_mastery_start = None
        else:
            try:
                concept_episode_mastery_start = float(raw_episode_mastery_start)
            except Exception:
                concept_episode_mastery_start = None
        try:
            concept_episode_step_count = int(data.get("concept_episode_step_count") or 0)
        except Exception:
            concept_episode_step_count = 0
        try:
            concept_episode_quiz_correct = int(data.get("concept_episode_quiz_correct") or 0)
        except Exception:
            concept_episode_quiz_correct = 0
        try:
            concept_episode_quiz_wrong = int(data.get("concept_episode_quiz_wrong") or 0)
        except Exception:
            concept_episode_quiz_wrong = 0
        concept_episode_last_control_type = data.get("concept_episode_last_control_type")
        return cls(
            learning_path=list(learning_path),
            focus_concept=data.get("focus_concept"),
            focus_level=data.get("focus_level"),
            cold_start=bool(data.get("cold_start")),
            cold_start_completed=list(completed),
            consecutive_explains=int(data.get("consecutive_explains") or 0),
            last_action=data.get("last_action"),
            phase=str(data.get("phase") or "teaching"),
            pending_question_type=data.get("pending_question_type"),
            pending_concept=data.get("pending_concept"),
            last_explained_concept=data.get("last_explained_concept"),
            history_window=history_window,
            history_focus=history_focus,
            srl_plan=srl_plan,
            srl_plan_step_index=srl_plan_step_index,
            session_plan=session_plan,
            session_plan_index=session_plan_index,
            session_strategy=session_strategy,
            state_machine=state_machine,
            session_summary=session_summary,
            quiz_phase=quiz_phase,
            quiz_question_index=quiz_question_index,
            quiz_max_questions=quiz_max_questions,
            last_mcq=last_mcq,
            recent_mcq_outcomes=list(recent_mcq_outcomes),
            concept_episode_id=concept_episode_id,
            concept_episode_concept=concept_episode_concept,
            concept_episode_start_turn=concept_episode_start_turn,
            concept_episode_mastery_start=concept_episode_mastery_start,
            concept_episode_step_count=concept_episode_step_count,
            concept_episode_quiz_correct=concept_episode_quiz_correct,
            concept_episode_quiz_wrong=concept_episode_quiz_wrong,
            concept_episode_last_control_type=concept_episode_last_control_type,
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
            "phase": self.phase,
            "pending_question_type": self.pending_question_type,
            "pending_concept": self.pending_concept,
            "last_explained_concept": self.last_explained_concept,
            "history_window": self.history_window,
            "history_focus": self.history_focus,
            "srl_plan": self.srl_plan,
            "srl_plan_step_index": self.srl_plan_step_index,
            "session_plan": self.session_plan,
            "session_plan_index": self.session_plan_index,
            "session_strategy": self.session_strategy,
            "state_machine": self.state_machine,
            "session_summary": self.session_summary,
            "quiz_phase": self.quiz_phase,
            "quiz_question_index": self.quiz_question_index,
            "quiz_max_questions": self.quiz_max_questions,
            "last_mcq": self.last_mcq,
            "recent_mcq_outcomes": self.recent_mcq_outcomes,
            "concept_episode_id": self.concept_episode_id,
            "concept_episode_concept": self.concept_episode_concept,
            "concept_episode_start_turn": self.concept_episode_start_turn,
            "concept_episode_mastery_start": self.concept_episode_mastery_start,
            "concept_episode_step_count": self.concept_episode_step_count,
            "concept_episode_quiz_correct": self.concept_episode_quiz_correct,
            "concept_episode_quiz_wrong": self.concept_episode_quiz_wrong,
            "concept_episode_last_control_type": self.concept_episode_last_control_type,
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
