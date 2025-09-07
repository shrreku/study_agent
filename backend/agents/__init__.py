from typing import Dict, Any

from .study_plan import study_plan_agent
from .daily_quiz import daily_quiz_agent
from .doubt import doubt_agent


def orchestrator_dispatch(agent_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if agent_name == "mock":
        return {"ok": True, "agent": "mock", "payload": payload}
    if agent_name == "study-plan":
        return study_plan_agent(payload)
    if agent_name == "daily-quiz":
        return daily_quiz_agent(payload)
    if agent_name == "doubt":
        return doubt_agent(payload)
    raise ValueError(f"unknown agent: {agent_name}")


