from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionPlanEntry:
    concept_id: str
    target_mastery: Optional[float] = None
    difficulty: Optional[str] = None
    prerequisites: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "target_mastery": self.target_mastery,
            "difficulty": self.difficulty,
            "prerequisites": list(self.prerequisites or []),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionPlanEntry":
        concept_id = str(data.get("concept_id") or "").strip()
        return cls(
            concept_id=concept_id,
            target_mastery=_safe_float(data.get("target_mastery")),
            difficulty=(str(data.get("difficulty")) or None) if data.get("difficulty") is not None else None,
            prerequisites=_safe_str_list(data.get("prerequisites")),
            notes=str(data.get("notes")) if data.get("notes") is not None else None,
        )


@dataclass
class SessionPlan:
    strategy: str
    entries: List[SessionPlanEntry] = field(default_factory=list)
    plan_id: str = ""
    created_at_step: int = 0
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "plan_id": self.plan_id,
            "created_at_step": self.created_at_step,
            "source": self.source,
            # Rich representation of entries for future use / logging.
            "entries": [e.to_dict() for e in self.entries],
            # Backwards-compatible simple representation expected by existing code.
            "concept_plan": [e.concept_id for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionPlan":
        strategy = str(data.get("strategy") or "learning_path").strip() or "learning_path"
        plan_id = str(data.get("plan_id") or "").strip()
        created_at_step = _safe_int(data.get("created_at_step"), default=0)
        source = str(data.get("source") or "").strip()

        entries: List[SessionPlanEntry] = []
        raw_entries = data.get("entries")
        if isinstance(raw_entries, list) and raw_entries:
            for item in raw_entries:
                if isinstance(item, dict):
                    try:
                        entry = SessionPlanEntry.from_dict(item)
                        if entry.concept_id:
                            entries.append(entry)
                    except Exception:
                        continue

        # Fallback to legacy "concept_plan": ["concept_id", ...] shape.
        if not entries:
            raw_cp = data.get("concept_plan")
            if isinstance(raw_cp, list):
                for cid in raw_cp:
                    if isinstance(cid, str) and cid.strip():
                        entries.append(SessionPlanEntry(concept_id=cid.strip()))

        return cls(
            strategy=strategy,
            entries=entries,
            plan_id=plan_id,
            created_at_step=created_at_step,
            source=source,
        )


@dataclass
class ConceptPlanStep:
    step_id: str
    step_type: str
    subgoal: Optional[str] = None
    instruction: str = ""
    tool_call: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    evaluation_type: Optional[str] = None
    expected_duration_steps: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "subgoal": self.subgoal,
            "instruction": self.instruction,
            "tool_call": self.tool_call,
            "params": dict(self.params or {}),
            "evaluation_type": self.evaluation_type,
            "expected_duration_steps": self.expected_duration_steps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptPlanStep":
        step_id = str(data.get("step_id") or "").strip()
        step_type = str(data.get("step_type") or "").strip()
        subgoal = str(data.get("subgoal")) if data.get("subgoal") is not None else None
        instruction = str(data.get("instruction") or "")
        tool_call = str(data.get("tool_call")) if data.get("tool_call") is not None else None
        params_raw = data.get("params")
        params: Dict[str, Any]
        if isinstance(params_raw, dict):
            params = dict(params_raw)
        else:
            params = {}
        evaluation_type = str(data.get("evaluation_type")) if data.get("evaluation_type") is not None else None
        expected_duration_steps = _safe_int(data.get("expected_duration_steps"), default=1)

        return cls(
            step_id=step_id,
            step_type=step_type,
            subgoal=subgoal,
            instruction=instruction,
            tool_call=tool_call,
            params=params,
            evaluation_type=evaluation_type,
            expected_duration_steps=expected_duration_steps,
        )


@dataclass
class ConceptPlan:
    plan_id: str
    concept_id: str
    steps: List[ConceptPlanStep] = field(default_factory=list)
    created_at_step: int = 0
    source: str = ""
    initial_mastery: Optional[float] = None
    target_mastery: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "concept_id": self.concept_id,
            "created_at_step": self.created_at_step,
            "source": self.source,
            "initial_mastery": self.initial_mastery,
            "target_mastery": self.target_mastery,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptPlan":
        plan_id = str(data.get("plan_id") or "").strip()
        concept_id = str(data.get("concept_id") or "").strip()
        created_at_step = _safe_int(data.get("created_at_step"), default=0)
        source = str(data.get("source") or "").strip()
        initial_mastery = _safe_float(data.get("initial_mastery"))
        target_mastery = _safe_float(data.get("target_mastery"))

        steps: List[ConceptPlanStep] = []
        raw_steps = data.get("steps")
        if isinstance(raw_steps, list):
            for item in raw_steps:
                if isinstance(item, dict):
                    try:
                        step = ConceptPlanStep.from_dict(item)
                        if step.step_type:
                            steps.append(step)
                    except Exception:
                        continue

        return cls(
            plan_id=plan_id,
            concept_id=concept_id,
            steps=steps,
            created_at_step=created_at_step,
            source=source,
            initial_mastery=initial_mastery,
            target_mastery=target_mastery,
        )


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _safe_str_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        result: List[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    result.append(text)
        return result
    # Single string case
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []
