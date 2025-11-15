"""Helper functions to simplify RL dataset format and extract agent steps."""
from __future__ import annotations

from typing import Any, Dict, List


def simplify_observation(full_obs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract essential observation fields."""
    user = full_obs.get("user") or {}
    classifier = full_obs.get("classifier") or {}
    tutor = full_obs.get("tutor") or {}
    retrieval = full_obs.get("retrieval") or {}
    
    return {
        "message": user.get("message", ""),
        "user_id": user.get("user_id", ""),
        "target_concepts": user.get("target_concepts", []),
        "intent": classifier.get("intent", "unknown"),
        "affect": classifier.get("affect", "neutral"),
        "focus_concept": tutor.get("focus_concept"),
        "concept_level": tutor.get("concept_level", "beginner"),
        "chunk_ids": retrieval.get("chunk_ids", []),
        "pedagogy_roles": retrieval.get("pedagogy_roles", []),
    }


def simplify_reward(full_reward: Dict[str, Any]) -> Dict[str, Any]:
    """Extract essential reward components."""
    if not full_reward:
        return {
            "rubric": 0.0,
            "intent": 0.0,
            "gating": 0.0,
            "grounding": 0.0,
            "style": 0.0,
            "total": 0.0,
            "flags": [],
        }
    
    components = full_reward.get("components", {})
    return {
        "rubric": float(components.get("rubric", {}).get("score") or 0.0),
        "intent": float(components.get("intent", {}).get("score") or 0.0),
        "gating": float(components.get("gating", {}).get("score") or 0.0),
        "grounding": float(components.get("grounding", {}).get("score") or 0.0),
        "style": float(components.get("style", {}).get("score") or 0.0),
        "total": float(full_reward.get("total") or 0.0),
        "flags": full_reward.get("flags", []),
    }


def simplify_critic(full_critic: Dict[str, Any]) -> Dict[str, Any]:
    """Extract essential critic scores."""
    return {
        "clarity": full_critic.get("clarity", 0.0),
        "accuracy": full_critic.get("accuracy", 0.0),
        "support": full_critic.get("support", 0.0),
        "confidence": full_critic.get("confidence", 0.0),
        "hallucination": full_critic.get("hallucination_flag", False),
        "notes": full_critic.get("notes", ""),
    }


def simplify_sft_record(full_record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert full SFT record to simplified format."""
    action = full_record.get("action") or {}
    meta = full_record.get("meta") or {}
    
    return {
        "observation": simplify_observation(full_record.get("observation", {})),
        "action_type": action.get("type", "explain"),
        "response": full_record.get("response", ""),
        "reward": simplify_reward(full_record.get("reward", {})),
        "critic": simplify_critic(full_record.get("critic", {})),
        "confidence": meta.get("confidence", 0.0),
    }


def simplify_preference_record(full_record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert full preference record to simplified format."""
    candidates = full_record.get("candidates", [])
    preference = full_record.get("preference", {})
    
    simplified_candidates = []
    for candidate in candidates:
        action = candidate.get("action") or {}
        reward = candidate.get("reward") or {}
        critic = candidate.get("critic") or {}
        
        simplified_candidates.append({
            "action_type": action.get("type", "explain"),
            "response": candidate.get("response", ""),
            "reward_total": reward.get("total", 0.0),
            "critic_confidence": critic.get("confidence", 0.0),
        })
    
    return {
        "observation": simplify_observation(full_record.get("observation", {})),
        "candidates": simplified_candidates,
        "chosen_index": preference.get("chosen", 0),
        "scores": preference.get("scores", []),
        "confidence": preference.get("confidence", 0.0),
        "reason": preference.get("reason", ""),
    }


def extract_agent_steps(full_observation: Dict[str, Any], action_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract step-by-step agent flow for UI display.

    When available, this uses the tutor's internal ``progress`` trace (including
    SRL multi-step execution) to build a detailed, ordered timeline. For older
    observations that do not include ``progress``, it falls back to a
    coarse-grained classification → policy → retrieval → action → response
    sequence.
    """

    steps: List[Dict[str, Any]] = []

    progress = full_observation.get("progress") or []

    if progress:
        # Rich SRL-aware path: classification → planning → retrieval → per-step
        # execution → decision → critique, then final response.
        for item in progress:
            stage = item.get("stage")

            if stage == "classification":
                steps.append(
                    {
                        "step": "classification",
                        "title": "Message Classification",
                        "data": dict(item),
                        "description": "Classified as {intent} intent with {affect} affect".format(
                            intent=item.get("intent", "unknown"),
                            affect=item.get("affect", "neutral"),
                        ),
                    }
                )

            elif stage == "planning":
                steps.append(
                    {
                        "step": "planning",
                        "title": "SRL Planning",
                        "data": dict(item),
                        "description": "Planned {action} with confidence {conf:.2f}".format(
                            action=item.get("intended_action", "explain"),
                            conf=float(item.get("confidence") or 0.0),
                        ),
                    }
                )

            elif stage == "retrieval":
                count = int(item.get("count") or 0)
                steps.append(
                    {
                        "step": "retrieval",
                        "title": "Knowledge Retrieval",
                        "data": dict(item),
                        "description": "Retrieved {count} chunks for query \"{query}\"".format(
                            count=count,
                            query=item.get("query", ""),
                        ),
                    }
                )

            elif stage == "step":
                # SRL multi-step execution from execute_plan_steps.step_progress
                idx = int(item.get("index") or 0)
                roles = item.get("roles") or []
                if not isinstance(roles, list):
                    roles = [roles]
                steps.append(
                    {
                        "step": f"srl_step_{idx}",
                        "title": "SRL Step {n}: {action}".format(
                            n=idx + 1,
                            action=item.get("action", "explain"),
                        ),
                        "data": dict(item),
                        "description": "Action {action} with roles {roles}; retrieved {retrieved} chunks".format(
                            action=item.get("action", "explain"),
                            roles=", ".join(roles) or "default",
                            retrieved=int(item.get("retrieved") or 0),
                        ),
                    }
                )

            elif stage == "decision":
                steps.append(
                    {
                        "step": "decision",
                        "title": "Final Action Decision",
                        "data": dict(item),
                        "description": "Chose {action} (cause: {cause}) with confidence {conf:.2f}".format(
                            action=item.get("action_type", "explain"),
                            cause=item.get("cause", "default"),
                            conf=float(item.get("confidence") or 0.0),
                        ),
                    }
                )

            elif stage == "critique":
                steps.append(
                    {
                        "step": "critique",
                        "title": "Self-Critique",
                        "data": dict(item),
                        "description": "Critique quality {quality:.2f}; should revise: {revise}".format(
                            quality=float(item.get("quality") or 0.0),
                            revise=item.get("should_revise"),
                        ),
                    }
                )

        # Always end with a response-generation step if we have result metadata.
        if action_result:
            action = full_observation.get("action") or {}
            steps.append(
                {
                    "step": "response",
                    "title": "Response Generation",
                    "data": {
                        "response": action_result.get("response", ""),
                        "confidence": action_result.get("confidence", 0.0),
                        "source_chunks": len(action_result.get("source_chunk_ids", [])),
                    },
                    "description": "Generated {action_type} response".format(
                        action_type=action.get("type", "explain"),
                    ),
                }
            )

        return steps

    # Fallback path for older observations without a progress trace.
    classifier = full_observation.get("classifier") or {}
    tutor = full_observation.get("tutor") or {}
    policy = full_observation.get("policy") or {}
    retrieval = full_observation.get("retrieval") or {}
    chunks = retrieval.get("chunks", [])
    action = full_observation.get("action") or {}

    # Step 1: Classification
    steps.append(
        {
            "step": "classification",
            "title": "Message Classification",
            "data": {
                "intent": classifier.get("intent", "unknown"),
                "affect": classifier.get("affect", "neutral"),
                "concept": classifier.get("concept", ""),
                "confidence": classifier.get("confidence", 0.0),
            },
            "description": "Classified as {intent} intent with {affect} affect".format(
                intent=classifier.get("intent", "unknown"),
                affect=classifier.get("affect", "neutral"),
            ),
        }
    )

    # Step 2: Policy Decision
    steps.append(
        {
            "step": "policy",
            "title": "Policy Decision",
            "data": {
                "focus_concept": tutor.get("focus_concept"),
                "concept_level": tutor.get("concept_level", "beginner"),
                "learning_path": tutor.get("learning_path", []),
                "cold_start": policy.get("cold_start", False),
            },
            "description": "Focus on {concept} at {level} level".format(
                concept=tutor.get("focus_concept", "concept"),
                level=tutor.get("concept_level", "beginner"),
            ),
        }
    )

    # Step 3: Retrieval
    steps.append(
        {
            "step": "retrieval",
            "title": "Knowledge Retrieval",
            "data": {
                "chunk_count": len(retrieval.get("chunk_ids", [])),
                "pedagogy_roles": retrieval.get("pedagogy_roles", []),
                "chunks": [
                    {
                        "id": c.get("id", ""),
                        "page": c.get("page_number"),
                        "score": c.get("score"),
                        "pedagogy_role": c.get("pedagogy_role"),
                    }
                    for c in chunks[:3]
                ],
            },
            "description": "Retrieved {count} relevant chunks".format(
                count=len(retrieval.get("chunk_ids", [])),
            ),
        }
    )

    # Step 4: Action Decision
    steps.append(
        {
            "step": "action",
            "title": "Action Selection",
            "data": {
                "action_type": action.get("type", "explain"),
                "override_applied": action.get("override_applied", False),
                "confidence": action.get("confidence", 0.0),
                "params": action.get("params", {}),
            },
            "description": "Selected {action_type} action{override}".format(
                action_type=action.get("type", "explain"),
                override=" (override)" if action.get("override_applied") else "",
            ),
        }
    )

    # Step 5: Response (from action_result if available)
    if action_result:
        steps.append(
            {
                "step": "response",
                "title": "Response Generation",
                "data": {
                    "response": action_result.get("response", ""),
                    "confidence": action_result.get("confidence", 0.0),
                    "source_chunks": len(action_result.get("source_chunk_ids", [])),
                },
                "description": "Generated {action_type} response".format(
                    action_type=action.get("type", "explain"),
                ),
            }
        )

    return steps


__all__ = [
    "simplify_observation",
    "simplify_reward",
    "simplify_critic",
    "simplify_sft_record",
    "simplify_preference_record",
    "extract_agent_steps",
]

