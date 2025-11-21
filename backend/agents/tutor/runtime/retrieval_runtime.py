"""Retrieval orchestration guided by policy and SRL plan."""

from typing import Any, Dict, List, Optional

from ..constants import logger
from ..retrieval import retrieve_chunks
from ..policy import role_sequence_for_level
from ..policy_decision import TutorPolicyDecision
from ..planning import TutorPlan
from ..decision_engine import ActionDecision
from .context import RetrievalContext


def run_retrieval(
    *,
    focus_concept: Optional[str],
    message: str,
    resource_id: Optional[str],
    concept_level: str,
    policy_decision: Optional[TutorPolicyDecision] = None,
    action_decision: Optional[ActionDecision] = None,
    srl_mode: bool = False,
    plan: Optional[TutorPlan] = None,
) -> RetrievalContext:
    """Orchestrate retrieval guided by policy, decision engine, and/or SRL plan.
    
    Returns RetrievalContext with chunks, query, roles, and chunk IDs.
    """
    role_sequence = role_sequence_for_level(concept_level)
    
    # Determine pedagogy roles
    policy_roles: List[str] = []
    if action_decision and action_decision.pedagogy_focus:
        policy_roles = action_decision.pedagogy_focus
    elif policy_decision is not None:
        try:
            policy_roles = list(policy_decision.pedagogy_focus or [])
        except Exception:
            policy_roles = []

    # Determine query
    query = None
    if action_decision and action_decision.retrieval_query:
        query = action_decision.retrieval_query
    elif policy_decision and policy_decision.retrieval_query:
        query = policy_decision.retrieval_query
    elif plan and getattr(plan, "retrieval_query", None):
        query = getattr(plan, "retrieval_query", None)
    
    if not query:
        query = focus_concept or message

    pedagogy_roles = policy_roles or list(getattr(plan, "pedagogy_focus", []) or []) or role_sequence

    # Determine max chunks from decision hints (if any)
    k = 15
    try:
        if action_decision and getattr(action_decision, "max_chunks", None) is not None:
            max_val = int(getattr(action_decision, "max_chunks"))
            if max_val > 0:
                k = max_val
    except Exception:
        k = 15

    # Handle explicit "no_retrieval" strategy: skip retrieval but keep query metadata.
    strategy = None
    try:
        if action_decision and getattr(action_decision, "retrieval_strategy", None):
            strategy = str(getattr(action_decision, "retrieval_strategy") or "").strip().lower()
    except Exception:
        strategy = None

    if strategy == "no_retrieval":
        chunks: List[Dict[str, Any]] = []
    else:
        chunks = retrieve_chunks(query, resource_id, pedagogy_roles, k=k)
    
    # Fallback strategies (only if retrieval is allowed)
    if strategy != "no_retrieval":
        if not chunks and focus_concept and query != focus_concept:
            chunks = retrieve_chunks(focus_concept, resource_id, pedagogy_roles, k=k)
        if not chunks and query != message:
            chunks = retrieve_chunks(message, resource_id, pedagogy_roles, k=k)

    retrieval_query = query
    retrieval_chunk_ids = [c.get("id") for c in (chunks or []) if c.get("id")]

    logger.info(
        "tutor_tool_retrieval_complete",
        extra={
            "query": retrieval_query,
            "pedagogy_roles": pedagogy_roles,
            "chunk_count": len(chunks or []),
            "chunk_ids": retrieval_chunk_ids,
            "srl_mode": srl_mode,
        },
    )

    return RetrievalContext(
        chunks=chunks or [],
        query=retrieval_query,
        pedagogy_roles=pedagogy_roles,
        chunk_ids=retrieval_chunk_ids,
    )
