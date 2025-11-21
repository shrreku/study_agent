"""Mastery update orchestration."""

import os
from datetime import datetime
from typing import Any, Dict, Optional

from ..constants import logger
from ..tools.mastery_updater import MasteryUpdater, MasteryUpdate
from ..validators.assessment import assess_student_response
from .utils import safe_float


def setup_mastery_updater() -> Optional[MasteryUpdater]:
    """Initialize mastery updater if enabled."""
    enable_mastery_update = (os.getenv("TUTOR_MASTERY_REALTIME_UPDATE", "true").strip().lower() == "true")
    if not enable_mastery_update:
        logger.info("tutor_tool_mastery_updater_disabled")
        return None

    try:
        lr = float(os.getenv("TUTOR_MASTERY_LEARNING_RATE", "0.1") or 0.1)
    except Exception:
        lr = 0.1
    try:
        df = float(os.getenv("TUTOR_MASTERY_DECAY_FACTOR", "0.95") or 0.95)
    except Exception:
        df = 0.95
    try:
        mn = float(os.getenv("TUTOR_MASTERY_MIN_UPDATE", "0.02") or 0.02)
    except Exception:
        mn = 0.02
    try:
        mx = float(os.getenv("TUTOR_MASTERY_MAX_UPDATE", "0.3") or 0.3)
    except Exception:
        mx = 0.3

    updater = MasteryUpdater(
        learning_rate=lr,
        decay_factor=df,
        min_update=mn,
        max_update=mx,
    )
    
    logger.info(
        "tutor_tool_mastery_updater_initialized",
        extra={
            "learning_rate": lr,
            "decay_factor": df,
            "min_update": mn,
            "max_update": mx,
        },
    )
    
    return updater


def apply_mastery_update(
    *,
    cur: Any,
    mastery_updater: Optional[MasteryUpdater],
    user_id: str,
    target_concept: Optional[str],
    mastery_map: Dict[str, Dict[str, Any]],
    intent: str,
    affect: str,
    classification_confidence: Optional[float],
    message: str,
    chunks: list,
    policy_should_update: Optional[bool],
    dry_run: bool,
    question_context: Optional[str] = None,
) -> Optional[float]:
    """Apply mastery update based on interaction signals.
    
    IMPORTANT: As of Phase 5, LLM policy should NOT override mastery computation.
    Policy override (policy_should_update=False) is DEPRECATED and ignored.
    Mastery updates are driven purely by student interaction signals.
    
    Returns the mastery delta applied, or None if no update was made.
    """
    if not mastery_updater or not target_concept or dry_run:
        return None

    # NOTE: Removed policy override check. Policy should not block mastery updates.
    # Mastery is now purely signal-driven.

    try:
        current_mastery = float((mastery_map.get(target_concept) or {}).get("mastery", 0.0) or 0.0)
    except Exception:
        current_mastery = 0.0

    # Optionally evaluate student's response to derive correctness and quality signals
    ans_correct = None
    expl_quality = None

    if intent in {"answer", "reflection", "explanation"}:
        try:
            logger.info(
                "tutor_tool_mastery_assessment_start",
                extra={
                    "user_id": user_id,
                    "concept": target_concept,
                    "intent": intent,
                },
            )
            assess = assess_student_response(
                student_message=message,
                expected_concept=target_concept or "",
                reference_chunks=chunks,
                question_context=question_context,
            )
            if isinstance(assess, dict):
                if assess.get("correct") is not None:
                    ans_correct = bool(assess.get("correct"))
                try:
                    qv = assess.get("quality")
                    if qv is not None:
                        expl_quality = float(qv)
                except Exception:
                    pass
                logger.info(
                    "tutor_tool_mastery_assessment_result",
                    extra={
                        "user_id": user_id,
                        "concept": target_concept,
                        "answer_correct": ans_correct,
                        "explanation_quality": expl_quality,
                    },
                )
        except Exception:
            logger.exception("tutor_tool_mastery_assessment_failed")

    interaction_signals = {
        "affect": affect,
        "intent": intent,
        "classification_confidence": classification_confidence,
        "answer_correct": ans_correct,
        "explanation_quality": expl_quality,
    }

    logger.info(
        "tutor_tool_mastery_compute_delta",
        extra={
            "user_id": user_id,
            "concept": target_concept,
            "current_mastery": current_mastery,
            "signals": interaction_signals,
        },
    )

    update = mastery_updater.compute_mastery_delta(
        concept=target_concept,
        user_id=user_id,
        interaction_signals=interaction_signals,
        current_mastery=current_mastery,
    )

    if update.delta != 0.0:
        logger.info(
            "tutor_tool_mastery_applying_update",
            extra={
                "user_id": user_id,
                "concept": target_concept,
                "delta": update.delta,
                "reason": update.reason,
                "confidence": update.confidence,
            },
        )
        new_mastery = mastery_updater.apply_update(
            user_id=user_id,
            update=update,
            db_cursor=cur,
        )
        if target_concept in mastery_map:
            try:
                mastery_map[target_concept]["mastery"] = float(new_mastery)
            except Exception:
                mastery_map[target_concept]["mastery"] = new_mastery
        return update.delta
    else:
        logger.info(
            "tutor_tool_mastery_no_update",
            extra={
                "user_id": user_id,
                "concept": target_concept,
                "reason": update.reason,
            },
        )

    return None


def _apply_fixed_mastery_delta(
    *,
    cur: Any,
    mastery_updater: Optional[MasteryUpdater],
    user_id: str,
    target_concept: Optional[str],
    mastery_map: Dict[str, Dict[str, Any]],
    delta: float,
    reason: str,
) -> Optional[float]:
    if not mastery_updater or not target_concept or delta == 0.0:
        return None

    try:
        current_mastery = float((mastery_map.get(target_concept) or {}).get("mastery", 0.0) or 0.0)
    except Exception:
        current_mastery = 0.0

    update = MasteryUpdate(
        concept=target_concept,
        delta=delta,
        reason=reason,
        confidence=0.7,
        timestamp=datetime.utcnow(),
    )

    new_mastery = mastery_updater.apply_update(
        user_id=user_id,
        update=update,
        db_cursor=cur,
    )

    if target_concept in mastery_map:
        try:
            mastery_map[target_concept]["mastery"] = float(new_mastery)
        except Exception:
            mastery_map[target_concept]["mastery"] = new_mastery

    try:
        logger.info(
            "tutor_tool_fixed_mastery_update",
            extra={
                "user_id": user_id,
                "concept": target_concept,
                "delta": delta,
                "reason": reason,
                "current_mastery": current_mastery,
                "new_mastery": new_mastery,
            },
        )
    except Exception:
        pass

    return delta


def apply_srl_step_mastery_delta(
    *,
    cur: Any,
    mastery_updater: Optional[MasteryUpdater],
    user_id: str,
    target_concept: Optional[str],
    mastery_map: Dict[str, Dict[str, Any]],
) -> Optional[float]:
    """Apply a small fixed mastery bump for an SRL teaching step.

    The delta is intended to be conservative and is configurable via
    environment variables:

    - TUTOR_SRL_STEP_MASTERY_DELTA_BASE (default: 0.1)
    - TUTOR_SRL_STEP_MASTERY_DELTA_MAX (default: 0.2)

    For backwards compatibility with earlier design docs, the non-prefixed
    variants SRL_STEP_MASTERY_DELTA_BASE / SRL_STEP_MASTERY_DELTA_MAX are
    also honored if set.
    """

    if not mastery_updater or not target_concept:
        return None

    def _read_delta_env(name: str, default: float) -> float:
        try:
            raw = os.getenv(name)
            if raw is None or str(raw).strip() == "":
                return default
            return float(raw)
        except Exception:
            return default

    base_delta = _read_delta_env("TUTOR_SRL_STEP_MASTERY_DELTA_BASE", 0.1)
    base_delta = _read_delta_env("SRL_STEP_MASTERY_DELTA_BASE", base_delta)

    max_delta = _read_delta_env("TUTOR_SRL_STEP_MASTERY_DELTA_MAX", 0.2)
    max_delta = _read_delta_env("SRL_STEP_MASTERY_DELTA_MAX", max_delta)

    # Ensure delta is non-negative and does not exceed the configured max.
    if base_delta < 0.0:
        base_delta = 0.0
    if max_delta < 0.0:
        max_delta = 0.0
    if max_delta > 0.0:
        base_delta = min(base_delta, max_delta)

    if base_delta == 0.0:
        return None

    return _apply_fixed_mastery_delta(
        cur=cur,
        mastery_updater=mastery_updater,
        user_id=user_id,
        target_concept=target_concept,
        mastery_map=mastery_map,
        delta=base_delta,
        reason="srl_step",
    )


def get_quiz_delta_tables() -> (Dict[str, float], Dict[str, float]):
    """Return per-difficulty mastery deltas for SRL quiz answers.

    Returns a tuple of (correct_table, wrong_table), where each table maps
    difficulty labels ("easy", "medium", "hard") to mastery deltas.

    Defaults (can be tuned via env vars):

    - Correct answer deltas:
      - easy:   +0.05
      - medium: +0.10
      - hard:   +0.15
    - Wrong answer penalties:
      - easy:   -0.05
      - medium: -0.10
      - hard:   -0.15

    Environment-based overrides are supported in two forms:

    - TUTOR_SRL_QUIZ_CORRECT_DELTAS / TUTOR_SRL_QUIZ_WRONG_DELTAS
    - QUIZ_CORRECT_DELTA_BY_DIFFICULTY / QUIZ_WRONG_PENALTY_BY_DIFFICULTY

    Each env, if set, should be a comma-separated "difficulty:delta" list,
    e.g. "easy:0.05,medium:0.1,hard:0.15".
    """

    correct = {"easy": 0.05, "medium": 0.10, "hard": 0.15}
    wrong = {"easy": -0.05, "medium": -0.10, "hard": -0.15}

    def _parse_table_env(env_name: str, base: Dict[str, float]) -> Dict[str, float]:
        raw = os.getenv(env_name)
        if raw is None:
            return base
        text = str(raw).strip()
        if not text:
            return base
        out = dict(base)
        try:
            parts = [p for p in text.split(",") if p.strip()]
            for part in parts:
                if ":" not in part:
                    continue
                key, val = part.split(":", 1)
                key = key.strip().lower()
                if key not in out:
                    continue
                try:
                    out[key] = float(val)
                except Exception:
                    continue
            return out
        except Exception:
            return base

    # Apply overrides, allowing both TUTOR_* and legacy names.
    correct = _parse_table_env("TUTOR_SRL_QUIZ_CORRECT_DELTAS", correct)
    correct = _parse_table_env("QUIZ_CORRECT_DELTA_BY_DIFFICULTY", correct)

    wrong = _parse_table_env("TUTOR_SRL_QUIZ_WRONG_DELTAS", wrong)
    wrong = _parse_table_env("QUIZ_WRONG_PENALTY_BY_DIFFICULTY", wrong)

    return correct, wrong
