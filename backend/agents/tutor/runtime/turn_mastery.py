from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from ..constants import logger
from ..state import TutorSessionPolicy
from ..planning import TutorPlan
from .context import ClassificationContext, RetrievalContext
from .mastery_runtime import (
    setup_mastery_updater,
    apply_mastery_update,
    apply_srl_step_mastery_delta,
    get_quiz_delta_tables,
    _apply_fixed_mastery_delta,
)


def apply_mastery_and_quiz(
    *,
    cur: Any,
    user_id: str,
    session_id: str,
    turn_index: int,
    message: str,
    classification: ClassificationContext,
    focus_concept: Optional[str],
    inference_concept: Optional[str],
    concept_level: str,
    mastery_map: Dict[str, Any],
    retrieval: RetrievalContext,
    srl_mode: bool,
    plan: Optional[TutorPlan],
    final_action_params: Optional[Dict[str, Any]],
    action_type: str,
    policy_state: TutorSessionPolicy,
    last_turns: List[Dict[str, Any]],
    config,
    mcq_outcome: Optional[Dict[str, Any]],
    dry_run: bool,
    requested_override_type: Optional[str],
    tool_calls: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Apply mastery update, SRL step delta, and quiz delta.

    Returns (mastery_delta, srl_step_delta, quiz_delta).
    Mutates mastery_map and policy_state in-place via the MasteryUpdater.
    """

    logger.info(
        "tutor_stage_mastery_update_start turn=%s target=%s intent=%s affect=%s",
        turn_index,
        inference_concept or focus_concept,
        classification.intent,
        classification.affect,
        extra={
            "session_id": session_id,
            "user_id": user_id,
        },
    )

    # Extract question context from recent turns for better assessment
    question_context = ""
    if last_turns:
        try:
            last_tutor = last_turns[0] if last_turns else {}
            if last_tutor.get("role") == "tutor":
                question_context = last_tutor.get("response_text", "")
        except Exception:
            pass

    mastery_updater = setup_mastery_updater()
    mastery_delta = apply_mastery_update(
        cur=cur,
        mastery_updater=mastery_updater,
        user_id=user_id,
        target_concept=inference_concept or focus_concept,
        mastery_map=mastery_map,
        intent=classification.intent,
        affect=classification.affect,
        classification_confidence=classification.confidence,
        message=message,
        chunks=retrieval.chunks,
        policy_should_update=None,  # Deprecated as of Phase 5
        dry_run=dry_run,
        question_context=question_context,
    )

    # Optional SRL step heuristic: in step-by-step SRL mode with an active
    # SRL plan, apply a small additional mastery bump for teaching steps.
    srl_step_delta: Optional[float] = None
    if (
        srl_mode
        and plan is not None
        and isinstance(final_action_params, dict)
        and str(final_action_params.get("mode") or "").strip().lower() in {"srl_plan", "srl_plan_multi"}
        and action_type in {"explain", "hint", "review", "reflect"}
        and not dry_run
    ):
        srl_step_delta = apply_srl_step_mastery_delta(
            cur=cur,
            mastery_updater=mastery_updater,
            user_id=user_id,
            target_concept=inference_concept or focus_concept,
            mastery_map=mastery_map,
        )
        if srl_step_delta not in (None, 0.0):
            if mastery_delta is None:
                mastery_delta = srl_step_delta
            else:
                try:
                    mastery_delta = float(mastery_delta) + float(srl_step_delta)
                except Exception:
                    pass

    # Quiz-specific bonus/penalty for step-by-step SRL mode: when in quiz
    # phase and the student answers, apply an additional mastery delta based
    # on difficulty and correctness.
    quiz_delta: Optional[float] = None
    if (
        config.mode == "step_by_step"
        and getattr(policy_state, "quiz_phase", "") == "quiz"
        and classification.intent == "answer"
        and not dry_run
        and mastery_updater
    ):
        try:
            quiz_idx = int(getattr(policy_state, "quiz_question_index", 0) or 0)
        except Exception:
            quiz_idx = 0
        try:
            quiz_max = int(getattr(policy_state, "quiz_max_questions", 0) or 0)
        except Exception:
            quiz_max = 0

        # Prefer MCQ-derived difficulty and correctness when available.
        mcq_diff = None
        mcq_correct = None
        if mcq_outcome is not None:
            try:
                mcq_diff = str(mcq_outcome.get("difficulty") or "").strip().lower() or None
            except Exception:
                mcq_diff = None
            try:
                ac = mcq_outcome.get("answer_correct")
                if ac is not None:
                    mcq_correct = bool(ac)
            except Exception:
                mcq_correct = None

        if mcq_diff:
            difficulty = mcq_diff
        elif quiz_idx <= 0:
            difficulty = "easy"
        elif quiz_idx == 1:
            difficulty = "medium"
        else:
            difficulty = "hard"

        correct_table, wrong_table = get_quiz_delta_tables()
        if mcq_correct is not None:
            was_correct = mcq_correct
        else:
            was_correct = bool(mastery_delta is not None and mastery_delta > 0.0)
        table = correct_table if was_correct else wrong_table
        extra = float(table.get(difficulty, 0.0) or 0.0)

        if extra != 0.0:
            quiz_delta = _apply_fixed_mastery_delta(
                cur=cur,
                mastery_updater=mastery_updater,
                user_id=user_id,
                target_concept=inference_concept or focus_concept,
                mastery_map=mastery_map,
                delta=extra,
                reason=f"quiz_{difficulty}_{'correct' if was_correct else 'wrong'}",
            )
            if quiz_delta not in (None, 0.0):
                if mastery_delta is None:
                    mastery_delta = quiz_delta
                else:
                    try:
                        mastery_delta = float(mastery_delta) + float(quiz_delta)
                    except Exception:
                        pass

        try:
            policy_state.quiz_question_index = quiz_idx + 1
        except Exception:
            pass

        concept_for_mastery = inference_concept or focus_concept
        try:
            current_mastery_now = float(
                (mastery_map.get(concept_for_mastery) or {}).get("mastery", 0.0) or 0.0
            )
        except Exception:
            current_mastery_now = 0.0
        try:
            target_mastery = float(os.getenv("TUTOR_STEP_SRL_TARGET_MASTERY", "0.8") or 0.8)
        except Exception:
            target_mastery = 0.8

        quiz_done = False
        if quiz_max > 0 and getattr(policy_state, "quiz_question_index", 0) >= quiz_max:
            quiz_done = True
        if current_mastery_now >= target_mastery:
            quiz_done = True

        if quiz_done:
            try:
                policy_state.quiz_phase = ""
                policy_state.quiz_question_index = 0
                policy_state.quiz_max_questions = 0
            except Exception:
                pass

    # Explicit override: step_next_concept – mark the current focus
    # concept as having reached target mastery so the concept loop
    # advances on the next turn.
    if (
        config.mode == "step_by_step"
        and requested_override_type == "step_next_concept"
        and not dry_run
        and mastery_updater
    ):
        concept_for_override = inference_concept or focus_concept
        if concept_for_override:
            try:
                current_mastery_now = float(
                    (mastery_map.get(concept_for_override) or {}).get("mastery", 0.0) or 0.0
                )
            except Exception:
                current_mastery_now = 0.0
            try:
                target_mastery_override = float(
                    os.getenv("TUTOR_STEP_SRL_TARGET_MASTERY", "0.8") or 0.8
                )
            except Exception:
                target_mastery_override = 0.8

            if current_mastery_now < target_mastery_override:
                delta_needed = target_mastery_override - current_mastery_now
                try:
                    skip_delta = _apply_fixed_mastery_delta(
                        cur=cur,
                        mastery_updater=mastery_updater,
                        user_id=user_id,
                        target_concept=concept_for_override,
                        mastery_map=mastery_map,
                        delta=delta_needed,
                        reason="override_step_next_concept",
                    )
                    if skip_delta not in (None, 0.0):
                        if mastery_delta is None:
                            mastery_delta = skip_delta
                        else:
                            try:
                                mastery_delta = float(mastery_delta) + float(skip_delta)
                            except Exception:
                                pass
                except Exception:
                    pass

    if mastery_updater:
        try:
            tool_calls["mastery_updater"] = {
                "enabled": True,
                "learning_rate": mastery_updater.learning_rate,
                "decay_factor": mastery_updater.decay_factor,
                "min_update": mastery_updater.min_update,
                "max_update": mastery_updater.max_update,
            }
            logger.info(
                "tutor_stage_mastery_update_complete turn=%s delta=%s lr=%s",
                turn_index,
                mastery_delta,
                mastery_updater.learning_rate,
                extra={
                    "session_id": session_id,
                    "user_id": user_id,
                },
            )
        except Exception:
            pass

    return mastery_delta, srl_step_delta, quiz_delta
