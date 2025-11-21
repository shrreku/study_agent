from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _normalize_str(value: Any) -> str:
    try:
        text = str(value or "").strip()
    except Exception:
        text = ""
    return text


class BasicQuizEvaluator:
    def __call__(
        self,
        *,
        question: Dict[str, Any],
        user_answer: Any,
        correct_answer: Any,
    ) -> Tuple[Dict[str, Any], Optional[float]]:
        if not isinstance(question, dict):
            return {
                "question_id": None,
                "concept": None,
                "difficulty": None,
                "answer_correct": None,
                "chose_explain": False,
                "option_id": None,
            }, None

        qid = _normalize_str(question.get("question_id"))
        concept = question.get("concept")
        difficulty = None

        options = question.get("options") or []
        if not isinstance(options, list):
            options = []

        ua = user_answer if isinstance(user_answer, dict) else {}
        ans_qid = _normalize_str(ua.get("question_id"))
        opt_id = _normalize_str(ua.get("option_id"))

        correct_id = _normalize_str(correct_answer) or _normalize_str(question.get("correct_option_id"))
        explain_id = _normalize_str(question.get("explain_option_id"))

        if options and opt_id:
            for opt in options:
                if not isinstance(opt, dict):
                    continue
                if _normalize_str(opt.get("id")) == opt_id:
                    diff_raw = opt.get("difficulty")
                    if isinstance(diff_raw, str) and diff_raw.strip():
                        difficulty = diff_raw.strip().lower()
                    break

        if difficulty is None:
            diff_raw = question.get("difficulty")
            if isinstance(diff_raw, str) and diff_raw.strip():
                difficulty = diff_raw.strip().lower()

        # Match behaviour from legacy orchestrator: accept answers where question_id
        # is missing in the answer or matches the last question.
        if qid and ans_qid and qid != ans_qid:
            # Mismatched question IDs – treat as no outcome.
            return {
                "question_id": qid,
                "concept": concept,
                "difficulty": difficulty,
                "answer_correct": None,
                "chose_explain": False,
                "option_id": None,
            }, None

        chose_explain = bool(opt_id and explain_id and opt_id == explain_id)

        answer_correct: Optional[bool] = None
        if opt_id:
            if opt_id == correct_id:
                answer_correct = True
            elif not chose_explain:
                answer_correct = False

        outcome = {
            "question_id": qid or None,
            "concept": concept,
            "difficulty": difficulty,
            "answer_correct": answer_correct,
            "chose_explain": chose_explain,
            "option_id": opt_id or None,
        }

        if answer_correct is None:
            return outcome, None

        difficulty_weight = 1.0
        if difficulty == "easy":
            difficulty_weight = 0.8
        elif difficulty == "hard":
            difficulty_weight = 1.2

        base = 1.0 if answer_correct else -0.5
        quiz_delta = base * difficulty_weight
        return outcome, quiz_delta
