from __future__ import annotations

from typing import Optional, List

from ..state import TutorSessionPolicy
from .plans import SessionPlan, SessionPlanEntry, ConceptPlan


def session_plan_from_policy(policy: TutorSessionPolicy) -> Optional[SessionPlan]:
    """Best-effort SessionPlan builder from TutorSessionPolicy.session_plan.

    Compatible with both legacy shapes (strategy + concept_plan list) and the
    richer SessionPlan.to_dict() representation.
    """

    raw = getattr(policy, "session_plan", None)
    if not isinstance(raw, dict):
        return None

    try:
        return SessionPlan.from_dict(raw)
    except Exception:
        # Fallback: only use the legacy concept_plan list and strategy fields.
        strategy = str(raw.get("strategy") or getattr(policy, "session_strategy", "learning_path") or "learning_path")
        plan_id = str(raw.get("plan_id") or "").strip()
        entries: List[SessionPlanEntry] = []
        raw_cp = raw.get("concept_plan")
        if isinstance(raw_cp, list):
            for cid in raw_cp:
                if isinstance(cid, str) and cid.strip():
                    entries.append(SessionPlanEntry(concept_id=cid.strip()))
        return SessionPlan(strategy=strategy, entries=entries, plan_id=plan_id)


def session_plan_to_policy(plan: SessionPlan, policy: TutorSessionPolicy) -> None:
    """Persist SessionPlan into TutorSessionPolicy.session_plan.

    This preserves a simple `concept_plan` list for backward compatibility
    with existing runtime code while storing the richer structure alongside.
    """

    data = plan.to_dict()
    # Ensure legacy consumers still see a simple list of concept ids.
    data["concept_plan"] = [entry.concept_id for entry in plan.entries]
    policy.session_plan = data
    # Keep session_strategy in sync when not already set.
    if not getattr(policy, "session_strategy", None):
        policy.session_strategy = plan.strategy


def concept_plan_from_policy(policy: TutorSessionPolicy) -> Optional[ConceptPlan]:
    """Extract a ConceptPlan from TutorSessionPolicy.srl_plan, if present.

    This does not interfere with existing TutorPlan usage; it only looks for
    an optional nested "concept_plan" key inside srl_plan.
    """

    raw_srl = getattr(policy, "srl_plan", None)
    if not isinstance(raw_srl, dict):
        return None

    cp_dict = raw_srl.get("concept_plan")
    if not isinstance(cp_dict, dict):
        return None

    try:
        return ConceptPlan.from_dict(cp_dict)
    except Exception:
        return None


def concept_plan_to_policy(plan: ConceptPlan, policy: TutorSessionPolicy) -> None:
    """Persist ConceptPlan into the nested concept_plan field of srl_plan.

    All existing srl_plan keys used by TutorPlan or legacy SRL code are left
    untouched; we only add or replace the "concept_plan" sub-dictionary.
    """

    raw_srl = getattr(policy, "srl_plan", None)
    if not isinstance(raw_srl, dict):
        raw_srl = {}

    raw_srl["concept_plan"] = plan.to_dict()
    policy.srl_plan = raw_srl
