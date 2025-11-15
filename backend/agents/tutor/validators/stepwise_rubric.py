from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from config.tutor_rl import ValidatorConfig
from .types import ValidatorComponentResult, ValidatorContext


class TutoringStep(Enum):
    UNDERSTAND_STUDENT = "understand_student"
    SELECT_PEDAGOGY = "select_pedagogy"
    RETRIEVE_CONTENT = "retrieve_content"
    STRUCTURE_RESPONSE = "structure_response"
    GENERATE_OUTPUT = "generate_output"
    FORMATIVE_CHECK = "formative_check"


@dataclass
class StepScore:
    step: TutoringStep
    score: float
    weight: float
    evidence: Dict[str, Any]
    feedback: str


@dataclass
class StepwiseRubricResult:
    step_scores: List[StepScore]
    overall_score: float
    strong_steps: List[str]
    weak_steps: List[str]
    actionable_feedback: str


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


class StepwiseRubricValidator:
    def __init__(self, config: ValidatorConfig) -> None:
        self.config = config
        # Defaults from ticket
        default_weights = {
            TutoringStep.UNDERSTAND_STUDENT: 0.15,
            TutoringStep.SELECT_PEDAGOGY: 0.25,
            TutoringStep.RETRIEVE_CONTENT: 0.20,
            TutoringStep.STRUCTURE_RESPONSE: 0.20,
            TutoringStep.GENERATE_OUTPUT: 0.15,
            TutoringStep.FORMATIVE_CHECK: 0.05,
        }
        env_overrides = {
            TutoringStep.UNDERSTAND_STUDENT: _env_float("TUTOR_STEP_WEIGHT_UNDERSTAND", default_weights[TutoringStep.UNDERSTAND_STUDENT]),
            TutoringStep.SELECT_PEDAGOGY: _env_float("TUTOR_STEP_WEIGHT_PEDAGOGY", default_weights[TutoringStep.SELECT_PEDAGOGY]),
            TutoringStep.RETRIEVE_CONTENT: _env_float("TUTOR_STEP_WEIGHT_RETRIEVAL", default_weights[TutoringStep.RETRIEVE_CONTENT]),
            TutoringStep.STRUCTURE_RESPONSE: _env_float("TUTOR_STEP_WEIGHT_STRUCTURE", default_weights[TutoringStep.STRUCTURE_RESPONSE]),
            TutoringStep.GENERATE_OUTPUT: _env_float("TUTOR_STEP_WEIGHT_OUTPUT", default_weights[TutoringStep.GENERATE_OUTPUT]),
            TutoringStep.FORMATIVE_CHECK: _env_float("TUTOR_STEP_WEIGHT_FORMATIVE", default_weights[TutoringStep.FORMATIVE_CHECK]),
        }
        total = sum(max(0.0, w) for w in env_overrides.values()) or 1.0
        self.step_weights: Dict[TutoringStep, float] = {k: max(0.0, v) / total for k, v in env_overrides.items()}

    def evaluate(self, context: ValidatorContext) -> StepwiseRubricResult:
        steps: List[StepScore] = []
        steps.append(self._eval_understand_student(context))
        steps.append(self._eval_select_pedagogy(context))
        steps.append(self._eval_retrieve_content(context))
        steps.append(self._eval_structure_response(context))
        steps.append(self._eval_generate_output(context))
        steps.append(self._eval_formative_check(context))

        overall = 0.0
        strong: List[str] = []
        weak: List[str] = []
        for s in steps:
            overall += s.score * s.weight
            if s.score > 0.8:
                strong.append(s.step.value)
            if s.score < 0.5:
                weak.append(s.step.value)

        feedback = self._generate_feedback(steps, weak)
        return StepwiseRubricResult(
            step_scores=steps,
            overall_score=round(overall, 4),
            strong_steps=strong,
            weak_steps=weak,
            actionable_feedback=feedback,
        )

    def _eval_understand_student(self, context: ValidatorContext) -> StepScore:
        obs = context.observation
        classifier = obs.get("classifier", {})
        tutor = obs.get("tutor", {})

        score = 0.0
        evidence: Dict[str, Any] = {}

        conf = float(classifier.get("confidence") or 0.0)
        if conf > 0.7:
            score += 0.4
            evidence["classification"] = "high_confidence"
        elif conf > 0.5:
            score += 0.2
            evidence["classification"] = "medium_confidence"
        else:
            evidence["classification"] = "low_confidence"

        mastery_snapshot = tutor.get("mastery_snapshot") or {}
        if mastery_snapshot:
            score += 0.3
            evidence["mastery"] = f"{len(mastery_snapshot)} fields"

        learning_path = tutor.get("learning_path") or []
        if learning_path:
            score += 0.3
            evidence["learning_path"] = f"{len(learning_path)} concepts"

        return StepScore(
            step=TutoringStep.UNDERSTAND_STUDENT,
            score=min(1.0, score),
            weight=self.step_weights[TutoringStep.UNDERSTAND_STUDENT],
            evidence=evidence,
            feedback=self._format_step_feedback("Student Understanding", score, evidence),
        )

    def _eval_select_pedagogy(self, context: ValidatorContext) -> StepScore:
        obs = context.observation
        tutor = obs.get("tutor", {})
        action = obs.get("action", {})
        classifier = obs.get("classifier", {})

        score = 0.0
        evidence: Dict[str, Any] = {}

        action_type = action.get("type", "unknown")
        level = str(tutor.get("concept_level") or "beginner")
        affect = classifier.get("affect", "neutral")
        intent = classifier.get("intent", "unknown")

        if affect in {"confused", "frustrated"} and action_type in {"hint", "explain"}:
            score += 0.3
            evidence["affect_match"] = "appropriate_for_confusion"
        elif affect == "engaged" and action_type in {"ask", "reflect"}:
            score += 0.3
            evidence["affect_match"] = "appropriate_for_engagement"

        if intent == "question" and action_type in {"explain", "hint"}:
            score += 0.3
            evidence["intent_match"] = "answering_question"
        elif intent == "answer" and action_type in {"reflect", "ask"}:
            score += 0.3
            evidence["intent_match"] = "prompting_reflection"

        retrieval = obs.get("retrieval", {})
        roles = retrieval.get("pedagogy_roles", [])
        expected = self._expected_roles_for_level(level)
        overlap = len(set(roles) & set(expected))
        if expected:
            score += 0.4 * (overlap / len(expected))
            evidence["pedagogy_roles"] = f"{overlap}/{len(expected)} appropriate"

        return StepScore(
            step=TutoringStep.SELECT_PEDAGOGY,
            score=min(1.0, score),
            weight=self.step_weights[TutoringStep.SELECT_PEDAGOGY],
            evidence=evidence,
            feedback=self._format_step_feedback("Pedagogy Selection", score, evidence),
        )

    def _eval_retrieve_content(self, context: ValidatorContext) -> StepScore:
        obs = context.observation
        retrieval = obs.get("retrieval", {})
        tutor = obs.get("tutor", {})

        score = 0.0
        evidence: Dict[str, Any] = {}

        chunks = retrieval.get("chunks") or []
        chunk_ids = retrieval.get("chunk_ids") or []
        focus_concept = (tutor.get("focus_concept") or "").strip()
        level = str(tutor.get("concept_level") or "beginner")

        cnt = len(chunk_ids)
        if cnt >= 3:
            score += 0.3
            evidence["chunk_count"] = f"{cnt} chunks"
        elif cnt >= 1:
            score += 0.15
            evidence["chunk_count"] = f"{cnt} chunks (low)"
        else:
            evidence["chunk_count"] = "no_chunks"

        mentions = 0
        if focus_concept and chunks:
            lower = focus_concept.lower()
            for ch in chunks[:3]:
                snippet = str(ch.get("snippet") or "").lower()
                if lower and lower in snippet:
                    mentions += 1
        if mentions > 0:
            score += 0.3 * (mentions / max(1, min(3, len(chunks))))
            evidence["concept_mentions"] = f"{mentions}/{min(3, len(chunks))}"

        if self._check_difficulty_match(chunks, level):
            score += 0.4
            evidence["difficulty"] = "appropriate"
        else:
            evidence["difficulty"] = "may_be_too_advanced"

        return StepScore(
            step=TutoringStep.RETRIEVE_CONTENT,
            score=min(1.0, score),
            weight=self.step_weights[TutoringStep.RETRIEVE_CONTENT],
            evidence=evidence,
            feedback=self._format_step_feedback("Content Retrieval", score, evidence),
        )

    def _eval_structure_response(self, context: ValidatorContext) -> StepScore:
        text = context.response_text or ""
        score = 0.0
        evidence: Dict[str, Any] = {}

        words = len(text.split())
        if 50 <= words <= 200:
            score += 0.4
            evidence["length"] = f"{words} words (good)"
        elif 30 <= words <= 250:
            score += 0.2
            evidence["length"] = f"{words} words (acceptable)"
        else:
            evidence["length"] = f"{words} words (too_short/long)"

        paragraphs = text.split("\n\n")
        if len(paragraphs) >= 2:
            score += 0.3
            evidence["structure"] = f"{len(paragraphs)} paragraphs"

        flow_markers = {
            "first": ["first", "initially", "to start"],
            "then": ["then", "next", "after"],
            "finally": ["finally", "in conclusion", "overall"],
        }
        flow_score = 0
        tl = text.lower()
        for markers in flow_markers.values():
            if any(m in tl for m in markers):
                flow_score += 1
        if flow_score >= 2:
            score += 0.3
            evidence["flow"] = f"{flow_score}/3 stages marked"

        return StepScore(
            step=TutoringStep.STRUCTURE_RESPONSE,
            score=min(1.0, score),
            weight=self.step_weights[TutoringStep.STRUCTURE_RESPONSE],
            evidence=evidence,
            feedback=self._format_step_feedback("Response Structure", score, evidence),
        )

    def _eval_generate_output(self, context: ValidatorContext) -> StepScore:
        from .rubric import rubric_check

        classic = rubric_check(context, self.config)
        features = classic.details.get("features", {}) if isinstance(classic.details, dict) else {}
        denom = len(features) or 1
        score = sum(float(v or 0.0) for v in features.values()) / denom
        evidence = {
            "direct_answer": features.get("direct_answer", 0),
            "example": features.get("example", 0),
            "reasoning": features.get("reasoning", 0),
        }
        return StepScore(
            step=TutoringStep.GENERATE_OUTPUT,
            score=float(score),
            weight=self.step_weights[TutoringStep.GENERATE_OUTPUT],
            evidence=evidence,
            feedback=self._format_step_feedback("Output Quality", score, evidence),
        )

    def _eval_formative_check(self, context: ValidatorContext) -> StepScore:
        text = context.response_text or ""
        score = 0.0
        evidence: Dict[str, Any] = {}

        if text.strip().endswith("?"):
            score += 0.5
            evidence["question"] = "present"

        tl = text.lower()
        if any(m in tl for m in ["what do you think", "can you explain", "how would you", "try"]):
            score += 0.3
            evidence["reflection_prompt"] = "present"

        if any(m in tl for m in ["next", "then", "after this", "once you understand"]):
            score += 0.2
            evidence["next_steps"] = "present"

        return StepScore(
            step=TutoringStep.FORMATIVE_CHECK,
            score=min(1.0, score),
            weight=self.step_weights[TutoringStep.FORMATIVE_CHECK],
            evidence=evidence,
            feedback=self._format_step_feedback("Formative Assessment", score, evidence),
        )

    def _expected_roles_for_level(self, level: str) -> List[str]:
        lvl = (level or "").lower()
        if lvl in {"beginner", "developing"}:
            return ["definition", "explanation", "example"]
        if lvl == "proficient":
            return ["example", "application", "derivation"]
        return ["derivation", "proof", "application"]

    def _check_difficulty_match(self, chunks: List[Dict[str, Any]], level: str) -> bool:
        if not chunks:
            return False
        lvl = (level or "").lower()
        level_to_difficulty = {
            "beginner": ["introductory"],
            "developing": ["introductory", "intermediate"],
            "proficient": ["intermediate"],
            "mastering": ["intermediate", "advanced"],
        }
        expected = level_to_difficulty.get(lvl, ["intermediate"])
        for ch in chunks[:3]:
            tags = ch.get("tags") or {}
            diff = str(tags.get("difficulty", "intermediate")).lower()
            if diff not in expected:
                return False
        return True

    def _format_step_feedback(self, name: str, score: float, evidence: Dict[str, Any]) -> str:
        if score > 0.8:
            return f"{name}: Excellent ({score:.2f})"
        if score > 0.6:
            return f"{name}: Good ({score:.2f}) - {evidence}"
        if score > 0.4:
            return f"{name}: Needs improvement ({score:.2f}) - {evidence}"
        return f"{name}: Poor ({score:.2f}) - Critical issues: {evidence}"

    def _generate_feedback(self, steps: List[StepScore], weak_steps: List[str]) -> str:
        if not weak_steps:
            return "Strong performance across all reasoning steps."
        parts: List[str] = ["Areas for improvement:"]
        for s in steps:
            if s.step.value in weak_steps:
                parts.append(f"- {s.feedback}")
        return "\n".join(parts)


def stepwise_rubric_check(context: ValidatorContext, config: ValidatorConfig) -> ValidatorComponentResult:
    """Wrapper to expose stepwise rubric as a validator component.

    Returns a ValidatorComponentResult named "stepwise_rubric" with:
    - score: overall stepwise score
    - details: step_scores (list), step_scores_map (dict), feedback, strong/weak steps
    - flags: optional flags based on weak steps
    """
    validator = StepwiseRubricValidator(config)
    result = validator.evaluate(context)

    step_scores_payload: List[Dict[str, Any]] = [
        {
            "step": s.step.value,
            "score": float(s.score),
            "weight": float(s.weight),
            "evidence": s.evidence,
            "feedback": s.feedback,
        }
        for s in result.step_scores
    ]
    step_scores_map: Dict[str, Dict[str, Any]] = {
        s.step.value: {"score": float(s.score), "weight": float(s.weight)} for s in result.step_scores
    }

    details: Dict[str, Any] = {
        "step_scores": step_scores_payload,
        "step_scores_map": step_scores_map,
        "feedback": result.actionable_feedback,
        "strong_steps": list(result.strong_steps),
        "weak_steps": list(result.weak_steps),
    }

    flags: List[str] = []
    if len(result.weak_steps) >= 2:
        flags.append("stepwise_needs_improvement")

    return ValidatorComponentResult(
        name="stepwise_rubric",
        score=float(result.overall_score),
        details=details,
        flags=flags,
    )

