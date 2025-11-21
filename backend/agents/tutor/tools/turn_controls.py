from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ParsedTurnControls:
    agent_action_mode: str
    step_control_type: Optional[str]
    step_control_params: Dict[str, Any]
    override_type: Optional[str]
    override_params: Dict[str, Any]
    confirmed_action: Optional[str]
    mcq_answer: Optional[Dict[str, Any]]
    is_control_turn: bool
    canonical_control_label: Optional[str]
    message_for_classification: str


def parse_turn_controls(
    payload: Dict[str, Any],
    *,
    message: str,
    default_mode: str = "auto",
) -> ParsedTurnControls:
    payload = payload or {}

    raw_mode = payload.get("agent_action_mode")
    base_mode = raw_mode if isinstance(raw_mode, str) and raw_mode.strip() else default_mode
    agent_action_mode = (base_mode or "auto").strip().lower()

    raw_confirmed = payload.get("confirmed_action")
    try:
        confirmed_action = str(raw_confirmed or "").strip().lower() or None
    except Exception:
        confirmed_action = None

    raw_mcq_answer = payload.get("mcq_answer")
    mcq_answer = raw_mcq_answer if isinstance(raw_mcq_answer, dict) else None

    raw_override = payload.get("action_override") or {}
    override_type: Optional[str] = None
    override_params: Dict[str, Any] = {}
    if isinstance(raw_override, dict):
        raw_type = raw_override.get("type")
        if isinstance(raw_type, str):
            override_type = raw_type.strip() or None
        params = raw_override.get("params")
        if isinstance(params, dict):
            allowed_keys = {"concept", "level", "difficulty", "question_type"}
            override_params = {
                key: value
                for key, value in params.items()
                if key in allowed_keys and value not in (None, "")
            }

    raw_step = payload.get("step_control") or {}
    step_control_type: Optional[str] = None
    step_control_params: Dict[str, Any] = {}
    if isinstance(raw_step, dict):
        raw_step_type = raw_step.get("type")
        if isinstance(raw_step_type, str):
            step_control_type = raw_step_type.strip().lower() or None
        step_params = raw_step.get("params")
        if isinstance(step_params, dict):
            step_control_params = step_params

    if agent_action_mode == "step_by_step" and step_control_type:
        if step_control_type == "continue":
            if not confirmed_action:
                confirmed_action = "continue"
        elif step_control_type == "skip_to_quiz":
            override_type = "step_skip_to_assessment"
        elif step_control_type in {"skip_to_next_concept", "skip_to_next"}:
            override_type = "step_next_concept"
        elif step_control_type == "next_step_override":
            try:
                raw_action = str(step_control_params.get("action") or "").strip().lower()
            except Exception:
                raw_action = ""
            if raw_action in {"ask", "explain", "hint", "review", "reflect"}:
                override_type = raw_action
                diff = step_control_params.get("difficulty")
                if isinstance(diff, str) and diff:
                    if not override_params:
                        override_params = {}
                    override_params.setdefault("difficulty", diff)

    message_for_classification: str = message if isinstance(message, str) else ""
    if agent_action_mode == "step_by_step" and step_control_type:
        message_for_classification = ""

    is_control_turn = bool(
        agent_action_mode == "step_by_step"
        and (step_control_type or mcq_answer is not None)
        and not (isinstance(message_for_classification, str) and message_for_classification.strip())
    )

    control_label: Optional[str] = None
    if mcq_answer is not None:
        control_label = "mcq_answer"
    elif step_control_type:
        control_label = step_control_type
    elif confirmed_action:
        control_label = confirmed_action

    return ParsedTurnControls(
        agent_action_mode=agent_action_mode,
        step_control_type=step_control_type,
        step_control_params=step_control_params,
        override_type=override_type,
        override_params=override_params,
        confirmed_action=confirmed_action,
        mcq_answer=mcq_answer,
        is_control_turn=is_control_turn,
        canonical_control_label=control_label,
        message_for_classification=message_for_classification,
    )


def to_canonical_control_label(parsed: ParsedTurnControls) -> Optional[str]:
    return parsed.canonical_control_label
