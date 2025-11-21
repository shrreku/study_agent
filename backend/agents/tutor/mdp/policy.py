from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .session import SessionObservation, SessionMDPAction
from .concept import ConceptObservation
from .actions import ConceptMDPAction
from .plans import SessionPlan, ConceptPlan
from .tutor import TutorStepObservation, TutorStepMDPAction
from .pedagogical_tutor import PedagogicalTutorObservation, PedagogicalTutorAction


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
        # Strong precedence for explicit user controls.
        if observation.last_control_type in {"skip_to_quiz"}:
            return ConceptMDPAction.JUMP_TO_ASSESSMENT

        if observation.last_control_type in {"skip_to_next_concept"}:
            return ConceptMDPAction.ADVANCE_CONCEPT

        if observation.last_control_type in {"end_session"}:
            return ConceptMDPAction.TERMINATE_CONCEPT

        if observation.last_control_type in {"replan_concept"}:
            return ConceptMDPAction.REPLAN_CONCEPT

        # If there is no concept-level plan yet, either advance (if mastery is
        # already high) or request a replanning step.
        if concept_plan is None or not concept_plan.steps:
            if observation.mastery is not None and observation.target_mastery is not None:
                if observation.mastery >= observation.target_mastery:
                    return ConceptMDPAction.ADVANCE_CONCEPT
            return ConceptMDPAction.REPLAN_CONCEPT

        # Quiz-phase behaviour: follow quiz steps until exhausted, then decide
        # whether to advance or terminate based on mastery.
        if observation.quiz_phase:
            if observation.quiz_question_index >= observation.quiz_max_questions > 0:
                if observation.mastery is not None and observation.target_mastery is not None:
                    if observation.mastery >= observation.target_mastery:
                        return ConceptMDPAction.ADVANCE_CONCEPT
                return ConceptMDPAction.TERMINATE_CONCEPT
            return ConceptMDPAction.FOLLOW_PLAN_STEP

        # Default: follow the concept plan step sequence.
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
        mastery = observation.mastery
        target = observation.target_mastery

        if mastery is not None and target is not None:
            gap = float(target) - float(mastery)
            if gap <= 0.05 and observation.at_end_of_plan:
                return PedagogicalTutorAction.SUMMARY
            if gap > 0.1:
                return PedagogicalTutorAction.EXPLAIN

        return PedagogicalTutorAction.EXPLAIN


def make_pedagogical_tutor_policy(config: Any = None) -> PedagogicalTutorPolicy:
    return DefaultPedagogicalTutorPolicy()
