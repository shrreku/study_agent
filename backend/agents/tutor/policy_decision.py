from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_ALLOWED_ACTIONS = {"explain", "ask", "reflect", "orientation", "review", "hint"}
_ALLOWED_MODES = {"orientation", "assessment", "cold_start", "prereq_review", "default"}


@dataclass
class TutorPolicyDecision:
    next_action: str = "explain"
    mode: str = "default"
    focus_concept: Optional[str] = None
    retrieval_query: Optional[str] = None
    advance_learning_path: bool = False
    use_srl_planning: bool = False
    use_multi_step: bool = False
    pedagogy_focus: List[str] = field(default_factory=list)
    history_window: Optional[int] = None
    history_focus: Optional[str] = None
    should_update_mastery: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TutorPolicyDecision":
        if not isinstance(data, dict):
            return cls()
        action_raw = str(data.get("next_action") or "").strip().lower()
        action = action_raw if action_raw in _ALLOWED_ACTIONS else "explain"
        mode_raw = str(data.get("mode") or "").strip().lower()
        mode = mode_raw if mode_raw in _ALLOWED_MODES else "default"
        focus = str(data.get("focus_concept") or "").strip() or None
        retrieval_query_raw = str(data.get("retrieval_query") or "").strip() or None
        advance = bool(data.get("advance_learning_path", False))
        use_plan = bool(data.get("use_srl_planning", False))
        use_multi = bool(data.get("use_multi_step", False))
        pf_raw = data.get("pedagogy_focus") or []
        pf_list: List[str] = []
        if isinstance(pf_raw, list):
            for item in pf_raw:
                try:
                    text = str(item).strip()
                except Exception:
                    continue
                if text:
                    pf_list.append(text)
        history_window_raw = data.get("history_window")
        history_window: Optional[int]
        if history_window_raw is None:
            history_window = None
        else:
            try:
                history_window = int(history_window_raw)
            except Exception:
                history_window = None
        history_focus_val = data.get("history_focus")
        if history_focus_val is None:
            history_focus: Optional[str] = None
        else:
            try:
                hf = str(history_focus_val).strip().lower()
            except Exception:
                hf = ""
            history_focus = hf or None
        should_update_raw = data.get("should_update_mastery")
        if should_update_raw is None:
            should_update: Optional[bool] = None
        else:
            should_update = bool(should_update_raw)
        return cls(
            next_action=action,
            mode=mode,
            focus_concept=focus,
            retrieval_query=retrieval_query_raw,
            advance_learning_path=advance,
            use_srl_planning=use_plan,
            use_multi_step=use_multi,
            pedagogy_focus=pf_list,
            history_window=history_window,
            history_focus=history_focus,
            should_update_mastery=should_update,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_action": self.next_action,
            "mode": self.mode,
            "focus_concept": self.focus_concept or "",
            "retrieval_query": self.retrieval_query or "",
            "advance_learning_path": bool(self.advance_learning_path),
            "use_srl_planning": bool(self.use_srl_planning),
            "use_multi_step": bool(self.use_multi_step),
            "pedagogy_focus": list(self.pedagogy_focus or []),
            "history_window": self.history_window,
            "history_focus": (self.history_focus or ""),
            "should_update_mastery": self.should_update_mastery,
        }
