"""LLM policy decision-making with heuristic fallback."""

from typing import Any, Dict, List, Optional

from ..constants import logger
from ..state import TutorSessionPolicy
from ..policy_llm import TutorPolicyLLM, build_policy_observation
from ..policy_decision import TutorPolicyDecision
from ..persistence import get_recent_turns
from ..config import get_tutor_config
from .context import ClassificationContext, ConceptContext
from .utils import normalize_single_concept_label


def decide_with_policy(
    *,
    cur: Any,
    session_id: str,
    message: str,
    classification: ClassificationContext,
    concepts: ConceptContext,
    policy_state: TutorSessionPolicy,
    agent_action_mode: str = "auto",
) -> tuple[Optional[TutorPolicyDecision], Optional[bool]]:
    """Make a policy decision using LLM if enabled, otherwise return None.
    
    Returns (policy_decision, should_update_mastery_override).
    """
    policy_decision = None
    policy_should_update: Optional[bool] = None

    mode_label = (agent_action_mode or "auto").strip().lower()
    if mode_label == "step_by_step":
        logger.info("tutor_tool_policy_llm_disabled_by_mode")
        return policy_decision, policy_should_update

    # Use unified TutorConfig instead of a separate env toggle.
    try:
        config = get_tutor_config()
        llm_policy_enabled = bool(getattr(config, "enable_llm_policy", False))
    except Exception:
        llm_policy_enabled = False

    if not llm_policy_enabled:
        logger.info("tutor_tool_policy_llm_disabled")
        return policy_decision, policy_should_update
    
    logger.info("tutor_tool_policy_llm_enabled")

    try:
        # Build a compact, history-aware view of recent dialogue for the policy.
        try:
            max_window = int(getattr(policy_state, "history_window", 3) or 3)
        except Exception:
            max_window = 3
        if max_window <= 0:
            max_window = 3
        history_focus = (getattr(policy_state, "history_focus", None) or "mixed").lower()
        if history_focus == "minimal":
            max_window = 1
        if max_window > 4:
            max_window = 4

        try:
            db_turns = get_recent_turns(cur, session_id, limit=max_window)
        except Exception:
            db_turns = []

        recent_turns: List[Dict[str, Any]] = []
        # db_turns is newest-first; iterate oldest-first to preserve order.
        for row in reversed(db_turns):
            concept_item = row.get("concept")
            action_type = row.get("action_type")
            user_text = str(row.get("user_text") or "").strip()
            tutor_text = str(row.get("response_text") or "").strip()
            if user_text and history_focus in {"mixed", "student", "minimal"}:
                recent_turns.append(
                    {
                        "role": "user",
                        "text": user_text[:280],
                        "action_type": None,
                        "concept": concept_item,
                    }
                )
            if tutor_text and history_focus in {"mixed", "tutor"}:
                recent_turns.append(
                    {
                        "role": "tutor",
                        "text": tutor_text[:280],
                        "action_type": action_type,
                        "concept": concept_item,
                    }
                )

        # Always include the current user message as the final entry.
        recent_turns.append(
            {
                "role": "user",
                "text": str(message or "")[:280],
                "action_type": None,
                "concept": classification.concept,
            }
        )
        if len(recent_turns) > 6:
            recent_turns = recent_turns[-6:]

        policy_obs = build_policy_observation(
            message=message,
            classification={
                "intent": classification.intent,
                "affect": classification.affect,
                "concept": classification.concept,
                "confidence": classification.confidence,
            },
            focus_concept=concepts.focus_concept,
            concept_level=concepts.concept_level,
            learning_targets=concepts.learning_targets,
            learning_path=concepts.learning_path,
            mastery_map=concepts.mastery_map,
            policy_state=policy_state,
            recent_turns=recent_turns,
        )
        policy_llm = TutorPolicyLLM()
        decision = policy_llm.decide(policy_obs)
        if decision is not None:
            policy_decision = decision
            policy_should_update = decision.should_update_mastery
            logger.info(
                "tutor_tool_policy_llm_decision_made",
                extra={
                    "next_action": decision.next_action,
                    "mode": decision.mode,
                    "use_srl_planning": decision.use_srl_planning,
                    "should_update_mastery": decision.should_update_mastery,
                    "history_window": decision.history_window,
                    "history_focus": decision.history_focus,
                },
            )
            # Allow the policy to steer how much history to surface on future turns.
            try:
                if decision.history_window is not None:
                    try:
                        hw = int(decision.history_window)
                    except Exception:
                        hw = getattr(policy_state, "history_window", 3) or 3
                    if hw <= 0:
                        hw = 1
                    if hw > 4:
                        hw = 4
                    policy_state.history_window = hw
                if decision.history_focus:
                    try:
                        hf = str(decision.history_focus).strip().lower()
                    except Exception:
                        hf = ""
                    if hf in {"mixed", "student", "tutor", "minimal"}:
                        policy_state.history_focus = hf
            except Exception:
                pass
        else:
            logger.info("tutor_tool_policy_llm_decision_failed")
    except Exception:
        logger.exception("tutor_tool_policy_llm_failed")

    return policy_decision, policy_should_update
