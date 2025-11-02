from typing import Dict, Any

from .study_plan import study_plan_agent
from .daily_quiz import daily_quiz_agent
from .doubt import doubt_agent
from .analysis import analysis_agent
from .tutor.agent import tutor_agent


def orchestrator_dispatch(agent_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to specialist agents after simple payload validation.

    Each agent has a lightweight JSON schema (required keys) declared in
    AGENT_SCHEMAS. Validation is permissive: it only enforces presence of
    required top-level keys when specified. This keeps the API developer-
    friendly while providing machine-readable schemas for clients.
    """
    # Agent schema registry: minimal required keys per agent
    AGENT_SCHEMAS = {
        "study-plan": {"required": []},
        "daily-quiz": {"required": ["concepts"]},
        "doubt": {"required": ["question"]},
        "analysis": {"required": ["user_id"]},
        "tutor": {"required": ["message", "user_id"]},
        "mock": {"required": []},
    }

    def _validate(agent: str, payload_dict: Dict[str, Any]) -> None:
        schema = AGENT_SCHEMAS.get(agent)
        if not schema:
            return
        reqs = schema.get("required", [])
        missing = [k for k in reqs if k not in payload_dict or payload_dict.get(k) in (None, "")]
        if missing:
            raise ValueError(f"invalid payload, missing keys: {missing}")

    if agent_name not in {"mock", "study-plan", "daily-quiz", "doubt", "analysis", "tutor"}:
        raise ValueError(f"unknown agent: {agent_name}")

    # run validation (will raise ValueError on missing required keys)
    _validate(agent_name, payload or {})

    if agent_name == "mock":
        return {"ok": True, "agent": "mock", "payload": payload}
    if agent_name == "study-plan":
        return study_plan_agent(payload)
    if agent_name == "daily-quiz":
        return daily_quiz_agent(payload)
    if agent_name == "doubt":
        return doubt_agent(payload)
    if agent_name == "analysis":
        return analysis_agent(payload)
    if agent_name == "tutor":
        return tutor_agent(payload)
    # unreachable due to check above
    raise ValueError(f"unknown agent: {agent_name}")


