from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging
import os
import random


logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    scenario_type: str
    intent: str
    affect: str
    message: str


@dataclass
class ConceptCheckAnswer:
    answer: str
    answer_correctness: str
    misconception_type: Optional[str]


_SCENARIO_INTENT_AFFECT: Dict[str, Dict[str, str]] = {
    "explain": {"intent": "question", "affect": "confused"},
    "worked_example": {"intent": "question", "affect": "curious"},
    "concept_check": {"intent": "answer", "affect": "unsure"},
    "reflection": {"intent": "reflection", "affect": "thoughtful"},
    "hint": {"intent": "question", "affect": "stuck"},
}


def _resolve_obs_llm_model(domain_config: Dict[str, Any]) -> Optional[str]:
    """Resolve which LLM model to use for observation-generation calls.

    Priority:
    1) domain_config["llm_model"] when set and non-empty
    2) OBS_LLM_MODEL environment variable
    3) fall back to call_llm_json's own defaults
    """

    value = domain_config.get("llm_model")
    if isinstance(value, str):
        v = value.strip()
        if v:
            return v

    env_val = os.getenv("OBS_LLM_MODEL", "").strip()
    if env_val:
        return env_val

    return None


def _call_llm_json_for_observations(
    domain_config: Dict[str, Any],
    prompt: str,
    default_payload: Dict[str, Any],
    log_event: str,
) -> Optional[Dict[str, Any]]:
    """Shared helper to call call_llm_json with optional model override.

    Returns parsed JSON dict on success, or None on any failure.
    """

    try:
        try:
            from backend.llm import call_llm_json  # type: ignore
        except Exception:
            from llm import call_llm_json  # type: ignore
    except Exception:
        return None

    model_name = _resolve_obs_llm_model(domain_config)
    model_ctx = None
    if model_name:
        try:
            try:
                from backend.llm.common import model_override_context  # type: ignore
            except Exception:
                from llm.common import model_override_context  # type: ignore
        except Exception:
            model_ctx = None
        else:
            model_ctx = model_override_context

    try:
        if model_name and model_ctx is not None:
            with model_ctx(model_name):
                return call_llm_json(prompt, default_payload)
        return call_llm_json(prompt, default_payload)
    except Exception:
        logger.exception(log_event)
        return None


def _choose_scenario_type(domain_config: Dict[str, Any], rng: random.Random) -> str:
    dist_cfg = domain_config.get("scenario_distribution") or {}
    items = list(dist_cfg.items())
    if not items:
        # Reasonable default mix
        items = [
            ("explain", 0.4),
            ("worked_example", 0.2),
            ("concept_check", 0.2),
            ("reflection", 0.1),
            ("hint", 0.1),
        ]
    total = sum(max(0.0, float(p)) for _, p in items) or 1.0
    r = rng.random() * total
    acc = 0.0
    for name, prob in items:
        acc += max(0.0, float(prob))
        if r <= acc:
            return name
    return items[-1][0]


def _build_message(scenario_type: str, concept: str, mastery_bucket: str) -> str:
    c = concept
    if scenario_type == "explain":
        return f"I don't fully understand {c}. Could you explain it in simple terms?"
    if scenario_type == "worked_example":
        return f"Can you walk me through a worked example that uses {c}?"
    if scenario_type == "concept_check":
        if mastery_bucket == "high":
            return f"I think I understand {c}, but can you check if my understanding is correct?"
        return f"Could you give me a quick question to check my understanding of {c}?"
    if scenario_type == "reflection":
        return f"How does {c} connect to the bigger picture of heat transfer?"
    if scenario_type == "hint":
        return f"I'm stuck on a problem about {c}. Can you give me a hint without giving away the full solution?"
    # Fallback
    return f"Can you help me with {c}?"


def _config_flag(domain_config: Dict[str, Any], name: str) -> bool:
    value = domain_config.get(name)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _generate_student_message_llm(
    domain_config: Dict[str, Any],
    concept: str,
    scenario_type: str,
    mastery_bucket: str,
    fallback_message: str,
) -> Optional[str]:
    """Use LLM to generate a realistic student message.

    Returns a validated message string or None on any failure.
    """

    prompt = (
        "You are simulating a student interacting with an AI tutor in an online course. "
        "Generate a single short utterance from the student's point of view. "
        "Return ONLY JSON with the shape {\"message\": \"...\"}. "
        "No extra keys, no commentary.\n\n"
        f"- Focus concept: {concept}\n"
        f"- Scenario type: {scenario_type}\n"
        f"- Mastery bucket: {mastery_bucket} (low/medium/high)\n"
        "- Tone: natural, concise, and aligned with the scenario type."
    )

    default_payload = {"message": fallback_message}

    result = _call_llm_json_for_observations(
        domain_config,
        prompt,
        default_payload,
        "obs_llm_student_message_failed",
    )
    if result is None:
        return None

    try:
        raw = result.get("message") if isinstance(result, dict) else None
    except Exception:
        raw = None
    if not raw:
        return None

    msg = str(raw).strip()
    if not msg:
        return None

    # Length bounds to avoid degenerate outputs
    if len(msg) < 8 or len(msg) > 512:
        return None

    # For most scenarios, ensure the concept is at least mentioned once
    requires_concept = scenario_type in {"explain", "worked_example", "reflection", "hint"}
    if requires_concept:
        if concept.lower() not in msg.lower():
            return None

    return msg


def build_scenario(domain_config: Dict[str, Any], concept: str, mastery_bucket: str, rng: random.Random) -> Scenario:
    scenario_type = _choose_scenario_type(domain_config, rng)
    intent_affect = _SCENARIO_INTENT_AFFECT.get(
        scenario_type,
        {"intent": "question", "affect": "neutral"},
    )
    base_message = _build_message(scenario_type, concept, mastery_bucket)

    message = base_message
    if _config_flag(domain_config, "use_llm_messages"):
        llm_msg = _generate_student_message_llm(domain_config, concept, scenario_type, mastery_bucket, base_message)
        if llm_msg:
            message = llm_msg

    return Scenario(
        scenario_type=scenario_type,
        intent=intent_affect["intent"],
        affect=intent_affect["affect"],
        message=message,
    )


def generate_concept_check_answer(
    domain_config: Dict[str, Any],
    concept: str,
    mastery_bucket: str,
) -> Optional[ConceptCheckAnswer]:
    """Optionally use LLM to generate a student's concept-check answer.

    Returns ConceptCheckAnswer or None if disabled or generation fails.
    """
    if not _config_flag(domain_config, "use_llm_answers"):
        return None

    prompt = (
        "You are simulating a student's short written answer in a concept-check question. "
        "Return ONLY JSON with keys: {\"answer\": string, \"answer_correctness\": string, \"misconception_type\": string}. "
        "answer_correctness must be one of: correct, near_miss, incorrect. "
        "misconception_type can be an empty string if not applicable.\n\n"
        f"- Focus concept: {concept}\n"
        f"- Mastery bucket: {mastery_bucket} (low/medium/high)\n"
        "- The answer should be 1-3 sentences, in the student's own words."
    )

    default_payload = {
        "answer": "",
        "answer_correctness": "unknown",
        "misconception_type": "",
    }

    result = _call_llm_json_for_observations(
        domain_config,
        prompt,
        default_payload,
        "obs_llm_concept_check_answer_failed",
    )
    if result is None:
        return None

    if not isinstance(result, dict):
        return None

    answer_raw = result.get("answer")
    if not answer_raw:
        return None

    answer = str(answer_raw).strip()
    if not answer or len(answer) < 5 or len(answer) > 512:
        return None

    correctness = str(result.get("answer_correctness") or "unknown").strip().lower()
    allowed = {"correct", "near_miss", "incorrect"}
    if correctness not in allowed:
        correctness = "unknown"

    misconception_raw = result.get("misconception_type")
    misconception = str(misconception_raw).strip() if misconception_raw is not None else ""
    if not misconception:
        misconception = None

    return ConceptCheckAnswer(
        answer=answer,
        answer_correctness=correctness,
        misconception_type=misconception,
    )
