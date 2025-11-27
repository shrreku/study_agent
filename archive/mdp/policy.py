from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .session import SessionObservation, SessionMDPAction
from .concept import ConceptObservation
from .actions import ConceptMDPAction
from .plans import SessionPlan, ConceptPlan
from .tutor import TutorStepObservation, TutorStepMDPAction
from .pedagogical_tutor import PedagogicalTutorObservation, PedagogicalTutorAction


logger = logging.getLogger(__name__)


class SessionPolicy(Protocol):
    def decide(
        self,
        *,
        observation: SessionObservation,
        session_plan: Optional[SessionPlan],
    ) -> SessionMDPAction: ...


class ConceptPolicy(Protocol):
    def decide(
        self,
        *,
        observation: ConceptObservation,
        concept_plan: Optional[ConceptPlan],
    ) -> ConceptMDPAction: ...


class TutorStepPolicy(Protocol):
    def decide(
        self,
        *,
        observation: TutorStepObservation,
        concept_action: ConceptMDPAction,
    ) -> TutorStepMDPAction: ...


class PedagogicalTutorPolicy(Protocol):
    def decide(
        self,
        *,
        observation: PedagogicalTutorObservation,
        concept_plan: Optional[ConceptPlan],
    ) -> PedagogicalTutorAction: ...


@dataclass
class DefaultSessionPolicy:
    def decide(
        self,
        *,
        observation: SessionObservation,
        session_plan: Optional[SessionPlan],
    ) -> SessionMDPAction:
        # If we have no explicit plan entries but there are concepts, request a
        # session replanning step; otherwise terminate when there is nothing to do.
        if session_plan is None or not session_plan.entries:
            if observation.total_concepts <= 0:
                return SessionMDPAction.TERMINATE_SESSION
            return SessionMDPAction.REPLAN_SESSION

        if observation.total_concepts <= 0:
            return SessionMDPAction.TERMINATE_SESSION

        if observation.last_concept_termination_reason == "session_end":
            return SessionMDPAction.TERMINATE_SESSION

        if observation.current_index >= observation.total_concepts:
            return SessionMDPAction.TERMINATE_SESSION

        if observation.last_concept_termination_reason in {
            "mastery_reached",
            "max_steps_reached",
            "skip_next_concept",
        }:
            if observation.current_index < observation.total_concepts:
                return SessionMDPAction.ADVANCE_IN_PLAN
            return SessionMDPAction.TERMINATE_SESSION

        return SessionMDPAction.FOLLOW_PLAN_CONCEPT


@dataclass
class SRLConceptPolicy:
    def decide(
        self,
        *,
        observation: ConceptObservation,
        concept_plan: Optional[ConceptPlan],
    ) -> ConceptMDPAction:
        logger.info(
            f"SRLConceptPolicy.decide input: last_control={observation.last_control_type}, "
            f"mastery={observation.mastery}, target={observation.target_mastery}, "
            f"plan_len={len(concept_plan.steps) if concept_plan and concept_plan.steps else 0}"
        )

        # Strong precedence for explicit user controls.
        if observation.last_control_type in {"skip_to_quiz"}:
            logger.info("SRL Decision: JUMP_TO_ASSESSMENT (explicit skip)")
            return ConceptMDPAction.JUMP_TO_ASSESSMENT

        if observation.last_control_type in {"skip_to_next_concept"}:
            logger.info("SRL Decision: ADVANCE_CONCEPT (explicit skip)")
            return ConceptMDPAction.ADVANCE_CONCEPT

        if observation.last_control_type in {"end_session"}:
            logger.info("SRL Decision: TERMINATE_CONCEPT (explicit end)")
            return ConceptMDPAction.TERMINATE_CONCEPT

        if observation.last_control_type in {"replan_concept"}:
            logger.info("SRL Decision: REPLAN_CONCEPT (explicit replan)")
            return ConceptMDPAction.REPLAN_CONCEPT

        # If there is no concept-level plan yet, either advance (if mastery is
        # already high) or request a replanning step.
        if concept_plan is None or not concept_plan.steps:
            if observation.mastery is not None and observation.target_mastery is not None:
                if observation.mastery >= observation.target_mastery:
                    logger.info("SRL Decision: ADVANCE_CONCEPT (mastery reached, no plan)")
                    return ConceptMDPAction.ADVANCE_CONCEPT
            logger.info("SRL Decision: REPLAN_CONCEPT (no plan)")
            return ConceptMDPAction.REPLAN_CONCEPT

        # Quiz-phase behaviour: follow quiz steps until exhausted, then decide
        # whether to advance or terminate based on mastery.
        if observation.quiz_phase:
            if observation.quiz_question_index >= observation.quiz_max_questions > 0:
                if observation.mastery is not None and observation.target_mastery is not None:
                    if observation.mastery >= observation.target_mastery:
                        logger.info("SRL Decision: ADVANCE_CONCEPT (quiz done, mastery reached)")
                        return ConceptMDPAction.ADVANCE_CONCEPT
                logger.info("SRL Decision: TERMINATE_CONCEPT (quiz done)")
                return ConceptMDPAction.TERMINATE_CONCEPT
            logger.info("SRL Decision: FOLLOW_PLAN_STEP (quiz phase)")
            return ConceptMDPAction.FOLLOW_PLAN_STEP

        # Check for stagnation or need for replanning
        # If we are deep in the plan but mastery is still very low, maybe replan?
        # For now, we stick to the plan unless explicit signals.

        # Default: follow the concept plan step sequence.
        logger.info("SRL Decision: FOLLOW_PLAN_STEP (default)")
        return ConceptMDPAction.FOLLOW_PLAN_STEP


def make_session_policy(config: Any = None) -> SessionPolicy:
    return DefaultSessionPolicy()


def make_concept_policy(config: Any = None) -> ConceptPolicy:
    return SRLConceptPolicy()


@dataclass
class DefaultTutorStepPolicy:
    def decide(
        self,
        *,
        observation: TutorStepObservation,
        concept_action: ConceptMDPAction,
    ) -> TutorStepMDPAction:
        # If the concept policy explicitly requested a replan, honor it.
        if concept_action == ConceptMDPAction.REPLAN_CONCEPT:
            return TutorStepMDPAction.REPLAN_CONCEPT

        # If there is no usable plan, ask for a concept-level replan.
        if observation.plan_length <= 0:
            return TutorStepMDPAction.REPLAN_CONCEPT

        # If we are already past the end of the plan, request a replan so
        # the concept planner can refresh the SRL steps.
        if observation.at_end_of_plan:
            return TutorStepMDPAction.REPLAN_CONCEPT

        # Otherwise, execute the next SRL plan step.
        return TutorStepMDPAction.NEXT_STEP


def make_tutor_step_policy(config: Any = None) -> TutorStepPolicy:
    return DefaultTutorStepPolicy()


@dataclass
class DefaultPedagogicalTutorPolicy:
    def decide(
        self,
        *,
        observation: PedagogicalTutorObservation,
        concept_plan: Optional[ConceptPlan],
    ) -> PedagogicalTutorAction:
        logger.info(
            f"Pedagogical Decision Input: last_control={observation.last_control_type}, "
            f"mastery={observation.mastery}, plan_index={observation.plan_index}"
        )
        
        # Determine whether to auto-execute the current step or wait for an
        # explicit control signal from the user. The first pedagogical step of
        # a plan auto-executes; subsequent steps require an explicit
        # 'continue' / 'next' / 'yes' or a 'replan_concept' signal.

        is_first_pedagogical_step = (
            observation.plan_index == 0
            and not observation.at_end_of_plan
            and observation.num_explain_steps == 0
            and observation.num_practice_steps == 0
            and observation.num_quiz_steps == 0
        )

        should_wait = True
        if is_first_pedagogical_step:
            should_wait = False
        elif observation.last_control_type in {"continue", "next", "yes"}:
            should_wait = False
        elif observation.last_control_type == "replan_concept":
            # If we just replanned, auto-execute the first step of the new plan.
            should_wait = False

        if should_wait:
            logger.info("Pedagogical Decision: WAIT_FOR_CONFIRMATION (gating)")
            return PedagogicalTutorAction.WAIT_FOR_CONFIRMATION

        mastery = observation.mastery
        target = observation.target_mastery

        # If we have a specific plan step, try to align with it
        if concept_plan and 0 <= observation.plan_index < len(concept_plan.steps):
            step = concept_plan.steps[observation.plan_index]
            step_type = getattr(step, "step_type", "").lower()

            if step_type == "introduction":
                logger.info("Pedagogical Decision: EXPLAIN (intro)")
                return PedagogicalTutorAction.EXPLAIN
            elif step_type == "explanation":
                logger.info("Pedagogical Decision: EXPLAIN (explanation)")
                return PedagogicalTutorAction.EXPLAIN
            elif step_type == "example":
                logger.info("Pedagogical Decision: EXPLAIN (example)")
                return PedagogicalTutorAction.EXPLAIN  # Or a specific EXAMPLE action if we had one
            elif step_type == "practice":
                logger.info("Pedagogical Decision: ASK (practice)")
                return PedagogicalTutorAction.ASK_QUESTION  # Using generic ASK for now if PRACTICE not in enum
            elif step_type == "reflection":
                logger.info("Pedagogical Decision: REFLECT")
                return PedagogicalTutorAction.REFLECTION_PROMPT

        # Fallback to mastery-based logic if plan step is ambiguous or missing
        if mastery is not None and target is not None:
            gap = float(target) - float(mastery)
            if gap <= 0.05 and observation.at_end_of_plan:
                logger.info("Pedagogical Decision: SUMMARY (mastery close/reached)")
                return PedagogicalTutorAction.SUMMARY
            if gap > 0.1:
                logger.info("Pedagogical Decision: EXPLAIN (mastery gap)")
                return PedagogicalTutorAction.EXPLAIN

        logger.info("Pedagogical Decision: EXPLAIN (fallback)")
        return PedagogicalTutorAction.EXPLAIN


def make_pedagogical_tutor_policy(config: Any = None) -> PedagogicalTutorPolicy:
    return DefaultPedagogicalTutorPolicy()
