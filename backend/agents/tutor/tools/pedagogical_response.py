from __future__ import annotations

from typing import Any, Dict, Optional
import logging

from llm import call_llm_json
from prompts import get as prompt_get, render as prompt_render

from ..mdp.pedagogical_tutor import PedagogicalTutorAction, PedagogicalTutorState
from ..mdp.plans import ConceptPlan


logger = logging.getLogger(__name__)


class DefaultPedagogicalResponseGeneratorTool:
    """Minimal pedagogical response generator for tutor MDP.

    Uses a single YAML-backed prompt (tutor.pedagogical_mdp_v1) and the
    generic llm.call_llm_json helper. Focuses only on:
    - concept name
    - pedagogical action name
    - optional trigger (for logging/debug only)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self._config = config or {}

    def __call__(
        self,
        *,
        user_id: str,
        session_id: str,
        concept_id: str,
        pedagogical_action: PedagogicalTutorAction,
        ped_state: PedagogicalTutorState,
        concept_plan: Optional[ConceptPlan],
        context_obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a single tutor message for the given pedagogical action.

        Required inputs:
        - concept_id: current concept name/ID
        - pedagogical_action: MDP action enum
        - context_obs["student_message"]: latest student utterance (may be empty)

        Optional inputs:
        - context_obs["trigger"]: e.g. "continue_plan", "new_plan" (for logging)
        """

        action = pedagogical_action
        student_message = context_obs.get("student_message") or ""
        trigger = context_obs.get("trigger") or "unspecified"
        feedback_override = context_obs.get("feedback_override")

        # Map core state into the MDP prompt payload. Tutor state is still
        # minimal, so we only rely on plan index/length when present.
        plan_index = int(getattr(ped_state, "plan_index", 0) or 0)
        plan_length = int(getattr(ped_state, "plan_length", 0) or 0)
        plan_subgoal: Optional[str] = None
        if concept_plan is not None:
            try:
                steps = list(getattr(concept_plan, "steps", []) or [])
                if 0 <= plan_index < len(steps):
                    step = steps[plan_index]
                    plan_subgoal = getattr(step, "subgoal", None) or getattr(step, "instruction", None)
            except Exception:
                plan_subgoal = None

        student_level = context_obs.get("student_level") or "unknown"
        recent_history = context_obs.get("recent_history") or ""

        # Map action to specific baseline prompt key
        key_map = {
            "explain": "tutor.explain",
            "ask_question": "tutor.ask",
            "hint": "tutor.hint",
            "reflection_prompt": "tutor.reflect",
            "worked_example": "tutor.explain",
            "guided_practice": "tutor.ask",
            "summary": "tutor.explain",
            "define_term": "tutor.explain",
            "wait_for_confirmation": "tutor.explain"
        }
        
        raw_action = str(action.value).lower()
        prompt_key = key_map.get(raw_action, "tutor.explain")
        
        # Fallback if not found, though baseline should have them
        template = prompt_get(prompt_key)
        if not template:
            prompt_key = "tutor.pedagogical_mdp_v1"
            template = prompt_get(prompt_key)

        # Construct payload
        if prompt_key == "tutor.pedagogical_mdp_v1":
            # Legacy Payload
            prompt_payload = {
                "pedagogical_action": action.value,
                "concept_id": concept_id,
                "plan_index": plan_index,
                "plan_length": plan_length,
                "plan_subgoal": plan_subgoal or "",
                "student_level": student_level,
                "student_message": student_message,
                "recent_history": recent_history,
            }
            if feedback_override:
                prompt_payload["student_message"] += f" [System Feedback: {feedback_override}]"
        else:
            # Baseline Prompt Payload
            # Expects: concept, student_message, level, context, recent_history
            
            context_text = f"Step {plan_index+1} of {plan_length}."
            if plan_subgoal:
                context_text += f"\nContent: {plan_subgoal}"
            
            if feedback_override:
                context_text += f"\n\n[IMPORTANT]: The student responded to the previous step. Analysis/Feedback: {feedback_override}. Start your response by acknowledging this feedback/correction."
                
            # For specific actions, we might want to prefix the context
            if raw_action == "worked_example":
                context_text = "Provide a worked example.\n" + context_text
            elif raw_action == "summary":
                context_text = "Provide a summary.\n" + context_text

            prompt_payload = {
                "concept": concept_id,
                "student_message": student_message or "continue",
                "level": student_level,
                "context": context_text,
                "recent_history": recent_history
            }

        prompt = prompt_render(template, prompt_payload)

        default_payload: Dict[str, Any] = {
            "response": "Let's continue with this concept.",
            "ui_mode": "free_text",
            "mcq_payload": None,
        }

        try:
            logger.info(
                "ped_response_mdp_call action=%s concept_id=%s plan_index=%s plan_length=%s trigger=%s prompt_key=%s",
                action.value,
                concept_id,
                plan_index,
                plan_length,
                trigger,
                prompt_key,
            )
            result = call_llm_json(prompt, default_payload)
        except Exception:
            logger.exception(
                "ped_response_mdp_error action=%s concept_id=%s trigger=%s",
                action.value,
                concept_id,
                trigger,
            )
            result = default_payload

        # Minimal handling: we only care about a text response for now and
        # keep ui_mode fixed to free_text. MCQ support can be added later.
        text = result.get("response") or default_payload["response"]

        messages = [
            {
                "role": "assistant",
                "content": text,
            }
        ]

        debug: Dict[str, Any] = {
            "pedagogical_action": action.value,
            "concept_id": concept_id,
            "session_id": session_id,
            "plan_index": plan_index,
            "plan_length": plan_length,
            "trigger": trigger,
            "prompt_key": prompt_key,
        }

        if concept_plan is not None:
            debug["concept_plan_id"] = getattr(concept_plan, "plan_id", None)

        return {
            "messages": messages,
            "ui_mode": "free_text",
            "mcq_payload": None,
            "debug": debug,
        }
