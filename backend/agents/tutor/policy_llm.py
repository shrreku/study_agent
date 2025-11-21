from __future__ import annotations

from typing import Any, Dict, List, Optional

from prompts import get as prompt_get, render as prompt_render
from llm import call_llm_json

from .constants import logger
from .state import TutorSessionPolicy
from .policy_decision import TutorPolicyDecision


def build_policy_observation(
    *,
    message: str,
    classification: Dict[str, Any],
    focus_concept: Optional[str],
    concept_level: str,
    learning_targets: List[str],
    learning_path: List[str],
    mastery_map: Dict[str, Dict[str, Any]],
    policy_state: TutorSessionPolicy,
    recent_turns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    intent = str(classification.get("intent") or "unknown")
    affect = str(classification.get("affect") or "neutral")
    concept = str(classification.get("concept") or "")
    confidence = classification.get("confidence")
    try:
        classifier_conf = float(confidence) if confidence is not None else None
    except Exception:
        classifier_conf = None

    mastery_snapshot = mastery_map.get(focus_concept) if focus_concept else None

    weak_concepts: List[str] = []
    for name, data in mastery_map.items():
        try:
            score = float((data or {}).get("mastery", 0.0) or 0.0)
        except Exception:
            continue
        if score < 0.4:
            weak_concepts.append(name)
        if len(weak_concepts) >= 5:
            break

    history_entries: List[Dict[str, Any]] = []
    for item in recent_turns[-6:]:
        role = item.get("role")
        if role not in {"user", "tutor"}:
            role = "user"
        text = str(item.get("text") or "").strip()
        action_type = item.get("action_type")
        concept_item = item.get("concept")
        history_entries.append(
            {
                "role": role,
                "text": text,
                "action_type": action_type,
                "concept": concept_item,
            }
        )

    obs: Dict[str, Any] = {
        "recent_dialogue": history_entries,
        "current_message": message,
        "classifier_intent": intent,
        "classifier_affect": affect,
        "classifier_concept": concept,
        "classifier_confidence": classifier_conf,
        "focus_concept": focus_concept or "",
        "concept_level": concept_level,
        "policy_phase": policy_state.phase,
        "last_action": policy_state.last_action or "",
        "consecutive_explains": policy_state.consecutive_explains,
        "learning_path": list(learning_path or []),
        "target_concepts": list(learning_targets or []),
        "current_mastery": (mastery_snapshot or {}).get("mastery"),
        "weak_concepts": weak_concepts,
    }
    return obs


class TutorPolicyLLM:
    def __init__(self) -> None:
        self._template_name = "tutor.policy"

    def decide(self, observation: Dict[str, Any]) -> Optional[TutorPolicyDecision]:
        try:
            template = prompt_get(self._template_name)
        except Exception:
            logger.exception("tutor_policy_prompt_missing")
            return None
        prompt = prompt_render(template, observation)
        default = TutorPolicyDecision().to_dict()
        try:
            raw = call_llm_json(prompt, default=default)
        except Exception:
            logger.exception("tutor_policy_llm_call_failed")
            return None
        try:
            decision = TutorPolicyDecision.from_dict(raw)
        except Exception:
            logger.exception("tutor_policy_llm_parse_failed")
            return None
        return decision
