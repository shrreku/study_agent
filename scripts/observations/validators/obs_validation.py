from __future__ import annotations

from typing import Any, Dict, Tuple


VALID_INTENTS = {"question", "answer", "reflection"}
VALID_AFFECTS = {"confused", "curious", "unsure", "thoughtful", "stuck", "neutral"}


def validate_observation_entry(entry: Dict[str, Any]) -> Tuple[bool, str | None]:
    """Lightweight structural and semantic validation for a single observation entry.

    Validation is intentionally conservative: if anything looks clearly wrong,
    we return (False, reason) so the caller can skip the entry or log it.
    """

    try:
        payload = entry.get("payload") or {}
        obs = entry.get("observation") or {}
    except Exception:
        return False, "missing_fields"

    # payload.message non-empty
    message = str(payload.get("message") or "").strip()
    if not message:
        return False, "empty_message"

    # retrieval: at least 2 chunks
    retrieval = obs.get("retrieval") or {}
    chunks = retrieval.get("chunks") or []
    if not isinstance(chunks, list) or len(chunks) < 2:
        return False, "too_few_chunks"

    # focus concept appears in at least one snippet
    tutor = obs.get("tutor") or {}
    focus = str(tutor.get("focus_concept") or "").strip()
    if not focus:
        return False, "missing_focus_concept"

    focus_lc = focus.lower()
    found = False
    for c in chunks:
        snippet = str((c or {}).get("snippet") or "").lower()
        if focus_lc and focus_lc in snippet:
            found = True
            break
    if not found:
        return False, "concept_not_in_snippet"

    # intent / affect checks
    classifier = obs.get("classifier") or {}
    intent = str(classifier.get("intent") or "").strip()
    affect = str(classifier.get("affect") or "").strip()
    if not intent:
        return False, "missing_intent"
    if not affect:
        return False, "missing_affect"

    if intent not in VALID_INTENTS:
        return False, "invalid_intent"
    if affect not in VALID_AFFECTS:
        return False, "invalid_affect"

    return True, None
