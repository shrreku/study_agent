from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, conint

# Import from scripts directory
# In Docker: /app/api/rl_tools.py -> /app -> /app/scripts
# Locally: backend/api/rl_tools.py -> backend -> root -> scripts
_API_DIR = Path(__file__).resolve().parent  # .../api
_BACKEND_OR_APP_DIR = _API_DIR.parent  # .../backend or /app
_SCRIPTS_DIR = _BACKEND_OR_APP_DIR / "scripts"

# Check if the scripts directory contains our rollout script
_ROLLOUT_SCRIPT = _SCRIPTS_DIR / "tutor_rollout_bandit.py"
if not _ROLLOUT_SCRIPT.exists():
    # Try going up one more level (local development with backend/ directory)
    _ROOT_DIR = _BACKEND_OR_APP_DIR.parent
    _SCRIPTS_DIR = _ROOT_DIR / "scripts"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from tutor_rollout_bandit import DEFAULT_ACTIONS, RolloutConfig, run_rollout  # type: ignore  # noqa: E402

# Import simplifier for optional simplified output
try:
    from api.rl_simplifier import (  # type: ignore
        simplify_sft_record,
        simplify_preference_record,
        extract_agent_steps,
    )
    HAS_SIMPLIFIER = True
except ImportError:
    HAS_SIMPLIFIER = False


router = APIRouter(prefix="/api/rl", tags=["rl-bandit"])


class RolloutRequest(BaseModel):
    observations: List[Dict[str, Any]]
    actions: Optional[List[str]] = None
    candidates: conint(ge=1, le=8) = Field(default=3)
    prompt_set: Optional[str] = None
    mock: bool = True
    seed: Optional[int] = None
    simplified: bool = Field(default=False, description="Return simplified format with essential fields only")
    detailed_steps: bool = Field(default=False, description="Include step-by-step breakdown for UI display")
    model_per_candidate: Optional[List[Dict[str, str]]] = None  # [{"action": "explain", "model": "gpt-4o"}, ...]
    critic_model: Optional[str] = None  # Model to use for critic scoring


class RolloutResponse(BaseModel):
    sft: List[Dict[str, Any]]
    prefs: List[Dict[str, Any]]
    steps: Optional[List[Dict[str, Any]]] = None  # Detailed steps for UI when requested


@contextmanager
def _temporary_env(mock: bool, prompt_set: Optional[str]):
    original_mock = os.environ.get("USE_LLM_MOCK")
    original_prompt = os.environ.get("PROMPT_SET")
    try:
        if mock:
            os.environ["USE_LLM_MOCK"] = "1"
        else:
            if original_mock is not None:
                os.environ.pop("USE_LLM_MOCK", None)
        if prompt_set:
            os.environ["PROMPT_SET"] = prompt_set
        else:
            if original_prompt is not None:
                os.environ.pop("PROMPT_SET", None)
        yield
    finally:
        if original_mock is None:
            os.environ.pop("USE_LLM_MOCK", None)
        else:
            os.environ["USE_LLM_MOCK"] = original_mock
        if original_prompt is None:
            os.environ.pop("PROMPT_SET", None)
        else:
            os.environ["PROMPT_SET"] = original_prompt


@router.post("/rollout", response_model=RolloutResponse)
def api_rollout(request: RolloutRequest) -> RolloutResponse:
    if not request.observations:
        raise HTTPException(status_code=400, detail="observations list must not be empty")

    actions: Sequence[str] = tuple(request.actions or DEFAULT_ACTIONS)
    if not actions:
        raise HTTPException(status_code=400, detail="At least one action must be provided")

    config = RolloutConfig(
        actions=actions,
        candidates=request.candidates,
        prompt_set=request.prompt_set,
        mock_mode=request.mock,
        seed=request.seed,
        model_per_candidate=request.model_per_candidate,
        critic_model=request.critic_model,
    )

    with _temporary_env(request.mock, request.prompt_set):
        results = run_rollout(request.observations, config=config)

    # Apply simplification if requested
    sft_data = results["sft"]
    prefs_data = results["prefs"]
    steps_data = None
    
    if request.simplified and HAS_SIMPLIFIER:
        sft_data = [simplify_sft_record(rec) for rec in sft_data]
        prefs_data = [simplify_preference_record(rec) for rec in prefs_data]
    
    # Extract detailed steps if requested
    if request.detailed_steps and HAS_SIMPLIFIER and sft_data:
        steps_data = []
        for sft_rec in results["sft"]:  # Use original for full data
            obs = sft_rec.get("observation", {})
            action_info = {
                "response": sft_rec.get("response", ""),
                "confidence": sft_rec.get("meta", {}).get("confidence", 0.0),
                "source_chunk_ids": sft_rec.get("action", {}).get("source_chunk_ids", []),
            }
            steps_data.append({
                "observation_id": obs.get("session", {}).get("session_id", ""),
                "steps": extract_agent_steps(obs, action_info),
            })

    return RolloutResponse(sft=sft_data, prefs=prefs_data, steps=steps_data)

