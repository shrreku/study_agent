from __future__ import annotations

from typing import Any, Dict, Iterable, List
import os

from config.tutor_rl import RewardWeights, ValidatorConfig
from .grounding import grounding_check
from .intent import intent_alignment
from .prereq import prereq_gate
from .rubric import rubric_check
from .stepwise_rubric import stepwise_rubric_check
from .style import style_check
from .types import ValidatorComponentResult, ValidatorContext


COMPONENT_ORDER: List[str] = ["stepwise_rubric", "rubric", "intent", "gating", "grounding", "style"]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _build_context(
    *, observation: Dict[str, Any], response_text: str, response_metadata: Dict[str, Any]
) -> ValidatorContext:
    return ValidatorContext(
        observation=observation,
        response_text=response_text,
        response_metadata=response_metadata,
    )


def score_response(
    observation: Dict[str, Any],
    response_text: str,
    response_metadata: Dict[str, Any] | None = None,
    *,
    weights: RewardWeights | None = None,
    config: ValidatorConfig | None = None,
) -> Dict[str, Any]:
    config = config or ValidatorConfig.from_env()
    weights = weights or RewardWeights.from_env()
    response_metadata = response_metadata or {}

    context = _build_context(
        observation=observation,
        response_text=response_text,
        response_metadata=response_metadata,
    )

    use_stepwise = os.getenv("TUTOR_STEPWISE_RUBRIC_ENABLED", "false").strip().lower() == "true"

    # Build component list in order; stepwise optional
    components: List[ValidatorComponentResult] = []
    if use_stepwise:
        try:
            components.append(stepwise_rubric_check(context, config))
        except Exception:
            # Fail open: ignore stepwise errors to avoid blocking
            pass
    components.extend(
        [
            rubric_check(context, config),
            intent_alignment(context, config),
            prereq_gate(context, config),
            grounding_check(context, config),
            style_check(context, config),
        ]
    )

    normalized_weights = weights.normalized()
    components_payload: Dict[str, Dict[str, Any]] = {}
    aggregated_flags: List[str] = []
    total = 0.0

    for component in components:
        components_payload[component.name] = component.to_dict()
        aggregated_flags.extend(component.flags)
        weight = normalized_weights.get(component.name, 0.0)
        total += component.score * weight

        threshold = config.thresholds.as_dict().get(component.name)
        if threshold is not None and component.score < threshold:
            aggregated_flags.append(f"{component.name}_below_threshold")

    total = _clamp(total)

    # Optionally export stepwise rubric step scores without affecting reward
    export_steps = os.getenv("TUTOR_RL_EXPORT_STEP_SCORES", "false").strip().lower() == "true"
    if export_steps and not use_stepwise:
        try:
            sw_component = stepwise_rubric_check(context, config)
            components_payload[sw_component.name] = sw_component.to_dict()
        except Exception:
            pass

    return {
        "components": components_payload,
        "total": round(total, 4),
        "weights": weights.as_dict(),
        "normalized_weights": normalized_weights,
        "flags": aggregated_flags,
    }


__all__ = ["score_response", "COMPONENT_ORDER"]

