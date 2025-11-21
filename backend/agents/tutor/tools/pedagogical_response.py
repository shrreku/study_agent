from __future__ import annotations

from typing import Any, Dict, Optional

from llm import call_llm_json
from prompts import get as prompt_get, render as prompt_render

from ..mdp.pedagogical_tutor import PedagogicalTutorAction, PedagogicalTutorState
from ..mdp.plans import ConceptPlan


class DefaultPedagogicalResponseGeneratorTool:
    """Default implementation of the PedagogicalResponseGeneratorTool.

    This is intentionally lightweight and prompt-agnostic. It maps
    PedagogicalTutorAction values into simple, well-formed tutor messages and
    UI payloads suitable for development and testing.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self._config = config

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
        action = pedagogical_action
        ui_mode = "free_text"
        mcq_payload: Optional[Dict[str, Any]] = None

        user_message = context_obs.get("student_message") or ""

        # Extract plan step metadata for the prompt.
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

        prompt_payload: Dict[str, Any] = {
            "pedagogical_action": action.value,
            "concept_id": concept_id,
            "plan_index": plan_index,
            "plan_length": plan_length,
            "plan_subgoal": plan_subgoal or "",
            "student_level": student_level,
            "student_message": user_message,
            "recent_history": recent_history,
        }

        llm_result: Optional[Dict[str, Any]] = None
        try:
            template = prompt_get("tutor.pedagogical_mdp_v1")
            prompt = prompt_render(template, prompt_payload)
            result = call_llm_json(prompt, model_hint=None)
            if isinstance(result, dict):
                llm_result = result
        except Exception:
            llm_result = None

        text: str
        use_fallback = False
        if llm_result is not None:
            raw_text = llm_result.get("response")
            raw_mode = llm_result.get("ui_mode")
            # Basic validation of required fields
            if not isinstance(raw_text, str) or not raw_text.strip() or not isinstance(raw_mode, str):
                use_fallback = True
            else:
                text = raw_text.strip()
                ui_mode_candidate = raw_mode.strip()
                if ui_mode_candidate not in {"free_text", "mcq"}:
                    use_fallback = True
                else:
                    ui_mode = ui_mode_candidate
                    raw_mcq = llm_result.get("mcq_payload")
                    if ui_mode == "mcq":
                        # For MCQ mode we require a dict payload; otherwise fallback.
                        if isinstance(raw_mcq, dict):
                            mcq_payload = raw_mcq
                        else:
                            use_fallback = True
        else:
            use_fallback = True

        if use_fallback:
            # Fallback to a simple, deterministic behaviour if the LLM call fails
            # or returns an invalid / malformed payload.
            if action is PedagogicalTutorAction.QUIZ_MCQ:
                ui_mode = "mcq"
                mcq_payload = self._build_simple_mcq(concept_id=concept_id, context_obs=context_obs)
                text = "Let's check your understanding with a quick question."
            elif action is PedagogicalTutorAction.EXPLAIN:
                text = "Let me explain this idea step by step."
            elif action is PedagogicalTutorAction.DEFINE_TERM:
                text = "I'll start by defining the key term we are working with."
            elif action is PedagogicalTutorAction.WORKED_EXAMPLE:
                text = "I'll walk through a worked example so you can see how it applies."
            elif action is PedagogicalTutorAction.GUIDED_PRACTICE:
                text = "Now it's your turn to try a similar problem. I'll guide you through it."
            elif action is PedagogicalTutorAction.ASK_QUESTION:
                text = "Here's a question for you: what part of this concept feels unclear so far?"
            elif action is PedagogicalTutorAction.REFLECTION_PROMPT:
                text = "Take a moment to summarize in your own words what you've learned so far."
            elif action is PedagogicalTutorAction.SUMMARY:
                text = "Let's quickly summarize the key points we've covered before we move on."
            else:
                text = "Let's continue working on this concept together."

        if user_message:
            # Light acknowledgment of the student's latest message.
            text = text + "\n\nYou said: " + str(user_message)

        messages = [
            {
                "role": "assistant",
                "content": text,
            }
        ]

        debug: Dict[str, Any] = {
            "pedagogical_action": action.value,
            "phase": ped_state.phase,
            "plan_index": ped_state.plan_index,
            "plan_length": ped_state.plan_length,
            "concept_id": concept_id,
            "session_id": session_id,
        }

        if concept_plan is not None:
            debug["concept_plan_id"] = getattr(concept_plan, "plan_id", None)

        return {
            "messages": messages,
            "ui_mode": ui_mode,
            "mcq_payload": mcq_payload,
            "debug": debug,
        }

    def _build_simple_mcq(
        self,
        *,
        concept_id: str,
        context_obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Construct a minimal MCQ payload for development and testing.

        This can be replaced or extended by more sophisticated generators
        without changing the main tool interface.
        """

        question_text = context_obs.get("mcq_question") or (
            f"Which statement about '{concept_id}' is correct?"
        )

        options = [
            {"id": "A", "text": "I feel confident with this concept."},
            {"id": "B", "text": "I am somewhat unsure."},
            {"id": "C", "text": "I do not understand it yet."},
        ]

        correct_option_id = context_obs.get("mcq_correct_option_id") or "A"

        return {
            "question": question_text,
            "options": options,
            "correct_option_id": correct_option_id,
        }
