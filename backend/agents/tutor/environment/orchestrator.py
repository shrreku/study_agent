from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..runtime.context import TurnContext
from ..mdp.plans import SessionPlan, ConceptPlan
from ..mdp.tools_factory import (
    make_session_planner_tool,
    make_concept_planner_tool,
    make_pedagogical_response_tool,
)
from .session_env import SessionEnvironment, SessionAction
from .concept_env import ConceptEnvironment, ConceptAction
from .tutor_env import TutorEnvironment, TutorAction
from .tools import EnvironmentStateManager, PlanCoordinator, ResponseBuilder

logger = logging.getLogger(__name__)


class EnvironmentOrchestrator:
    """Coordinates the 3-layer environment for a single tutor turn.

    This class is intentionally lightweight for the MVP:
    - Deterministic, stationary transitions in each environment.
    - Simple, rule-based policies wired directly in this class.
    - Buttons are treated as control signals carried in ``TurnContext.payload``.
    """

    def __init__(self, cur: Any, config: Optional[Any] = None) -> None:
        self.cur = cur
        self.state_manager = EnvironmentStateManager(cur)

        # Planning and response tools
        session_planner = make_session_planner_tool(config)
        concept_planner = make_concept_planner_tool(config)
        self.plan_coordinator = PlanCoordinator(session_planner, concept_planner)
        self.response_tool = make_pedagogical_response_tool(config)

    def _normalize_button_signal(self, payload: Dict[str, Any]) -> Optional[str]:
        """Extract a normalized control signal from the payload.

        We keep this intentionally small and button-focused for the MVP.
        """

        if not isinstance(payload, dict):
            return None

        # Primary boolean flag from API layer
        button_clicked = bool(payload.get("button_clicked"))
        step_control = payload.get("step_control") or {}
        confirmed = payload.get("confirmed_action")

        label: Optional[str] = None
        if isinstance(step_control, dict):
            raw_type = step_control.get("type")
            if isinstance(raw_type, str) and raw_type.strip():
                label = raw_type.strip().lower()

        if not label and isinstance(confirmed, str) and confirmed.strip():
            label = confirmed.strip().lower()

        if button_clicked:
            return label or "continue"

        return None

    def _decide_tutor_action(self, step: Any) -> TutorAction:
        """Map a concept plan step to a pedagogical tutor action.

        This is a simple heuristic mapper based on ``step.step_type``.
        """

        if step is None:
            return TutorAction.SUMMARY

        step_type = str(getattr(step, "step_type", "") or "").strip().lower()

        if step_type in {"explain"}:
            return TutorAction.EXPLAIN
        if step_type in {"example", "worked_example"}:
            return TutorAction.WORKED_EXAMPLE
        if step_type in {"practice", "guided_practice"}:
            return TutorAction.GUIDED_PRACTICE
        if step_type in {"summary"}:
            return TutorAction.SUMMARY
        if step_type in {"quiz", "mcq"}:
            return TutorAction.QUIZ_MCQ

        return TutorAction.EXPLAIN

    def run_turn(self, ctx: TurnContext) -> Dict[str, Any]:
        """Run a full 3-layer environment turn.

        This function:
        1. Loads session + concept state from persistence.
        2. Runs the session, concept, and tutor environments.
        3. Persists updated state.
        4. Returns a frontend-compatible response.
        """

        session_id = ctx.session_id
        user_id = ctx.user_id
        payload = ctx.payload or {}

        logger.info(
            "environment_turn_start session_id=%s user_id=%s turn_index=%s",
            session_id,
            user_id,
            ctx.turn_index,
        )

        control_signal = self._normalize_button_signal(payload)

        # ------------------------------------------------------------------
        # 1. Session layer
        # ------------------------------------------------------------------
        raw_session_state = self.state_manager.load_session_state(session_id, user_id)

        raw_plan = raw_session_state.get("session_plan")
        session_plan: Optional[SessionPlan] = None
        if isinstance(raw_plan, dict):
            try:
                session_plan = SessionPlan.from_dict(raw_plan)
            except Exception:
                session_plan = None

        mastery_map = raw_session_state.get("mastery_map") or {}

        session_env = SessionEnvironment(
            session_id=session_id,
            user_id=user_id,
            session_plan=session_plan,
            mastery_map=mastery_map,
        )

        s_state = session_env.get_state()
        s_state.current_concept_index = int(raw_session_state.get("current_concept_index", 0) or 0)
        s_state.turn_count = int(raw_session_state.get("turn_count", 0) or 0)

        # If there is no plan yet and we have target concepts, generate a simple one.
        if (not s_state.session_plan or not s_state.session_plan.entries) and ctx.target_concepts:
            logger.info("environment_session_generate_plan target_concepts=%s", ctx.target_concepts)
            plan = self.plan_coordinator.generate_session_plan(
                user_id=user_id,
                session_id=session_id,
                target_concepts=ctx.target_concepts,
                mastery_map=mastery_map,
                strategy="sequential",
            )
            s_state.session_plan = plan

        # Decide whether to stay on the current concept or advance based on
        # the persisted concept termination flag.
        current_concept_id = s_state.current_concept_id
        current_concept_terminated = False
        if s_state.session_plan and s_state.session_plan.entries and current_concept_id:
            try:
                last_concept_state = self.state_manager.load_concept_state(
                    session_id=session_id,
                    user_id=user_id,
                    concept_id=current_concept_id,
                )
                current_concept_terminated = bool(last_concept_state.get("terminated"))
            except Exception:
                current_concept_terminated = False

        # Session decision:
        # - No plan or past end → END_SESSION
        # - Current concept completed → ADVANCE_CONCEPT
        # - Otherwise → START_CONCEPT (continue current concept)
        if not s_state.session_plan or not s_state.session_plan.entries:
            session_action = SessionAction.END_SESSION
        elif s_state.current_concept_index >= len(s_state.session_plan.entries):
            session_action = SessionAction.END_SESSION
        elif current_concept_terminated:
            session_action = SessionAction.ADVANCE_CONCEPT
        else:
            session_action = SessionAction.START_CONCEPT

        session_transition = session_env.step(session_action)
        logger.info(
            "rl_session_transition session_id=%s user_id=%s action=%s concept_index=%s terminated=%s reason=%s",
            session_id,
            user_id,
            session_action.value,
            session_env.get_state().current_concept_index,
            session_transition.terminated,
            session_transition.termination_reason,
        )
        focus_concept = session_transition.outputs.get("focus_concept")

        if session_transition.terminated or not focus_concept:
            # Persist and return a simple session-complete response.
            self.state_manager.save_session_state(session_env)
            logger.info("environment_session_terminated session_id=%s", session_id)
            return ResponseBuilder.build_session_complete_response(session_env)

        concept_id = str(focus_concept)

        # ------------------------------------------------------------------
        # 2. Concept layer
        # ------------------------------------------------------------------
        raw_concept_state = self.state_manager.load_concept_state(
            session_id=session_id,
            user_id=user_id,
            concept_id=concept_id,
        )

        raw_cplan = raw_concept_state.get("concept_plan")
        concept_plan: Optional[ConceptPlan] = None
        if isinstance(raw_cplan, dict):
            try:
                concept_plan = ConceptPlan.from_dict(raw_cplan)
            except Exception:
                concept_plan = None

        current_mastery = float(raw_concept_state.get("current_mastery") or 0.0)

        concept_env = ConceptEnvironment(
            session_id=session_id,
            user_id=user_id,
            concept_id=concept_id,
            concept_plan=concept_plan,
            current_mastery=current_mastery,
            target_mastery=0.8,
        )

        c_state = concept_env.get_state()
        c_state.current_step_index = int(raw_concept_state.get("current_step_index", 0) or 0)
        c_state.phase = str(raw_concept_state.get("phase") or "learning")

        # Ensure a plan exists by calling the planner when needed.
        if c_state.concept_plan is None:
            logger.info("environment_concept_generate_plan concept_id=%s", concept_id)
            _ = concept_env.step(
                ConceptAction.GENERATE_PLAN,
                planner=self.plan_coordinator.generate_concept_plan,
                context_obs={
                    "student_message": ctx.message,
                    "current_mastery": c_state.current_mastery,
                    "target_mastery": c_state.target_mastery,
                },
            )
            c_state = concept_env.get_state()

        # Concept decision: button click → advance, otherwise execute current step.
        if control_signal:
            concept_action = ConceptAction.ADVANCE_STEP
        else:
            concept_action = ConceptAction.EXECUTE_STEP

        concept_transition = concept_env.step(
            concept_action,
            planner=self.plan_coordinator.generate_concept_plan,
            context_obs={
                "student_message": ctx.message,
                "current_mastery": c_state.current_mastery,
                "target_mastery": c_state.target_mastery,
            },
        )
        logger.info(
            "rl_concept_transition session_id=%s user_id=%s concept_id=%s action=%s step_index=%s terminated=%s reason=%s mastery=%s",
            session_id,
            user_id,
            concept_id,
            concept_action.value,
            concept_env.get_state().current_step_index,
            concept_transition.terminated,
            concept_transition.termination_reason,
            concept_env.get_state().current_mastery,
        )

        current_step = concept_transition.outputs.get("current_step")

        # ------------------------------------------------------------------
        # 3. Tutor layer
        # ------------------------------------------------------------------
        tutor_env = TutorEnvironment(
            session_id=session_id,
            user_id=user_id,
            concept_id=concept_id,
            response_tool=self.response_tool,
            current_step=current_step,
            concept_plan=concept_env.get_state().concept_plan,
            plan_index=concept_env.get_state().current_step_index,
            phase=concept_env.get_state().phase,
        )

        tutor_action = self._decide_tutor_action(current_step)

        tutor_transition = tutor_env.step(
            tutor_action,
            control_signal=control_signal,
            context_obs={
                "student_message": ctx.message,
            },
        )
        logger.info(
            "rl_tutor_transition session_id=%s user_id=%s concept_id=%s action=%s control_signal=%s turn_in_step=%s",
            session_id,
            user_id,
            concept_id,
            tutor_action.value,
            control_signal,
            tutor_env.get_state().turn_in_step,
        )
        tutor_outputs = tutor_transition.outputs

        # ------------------------------------------------------------------
        # 4. Persistence and response
        # ------------------------------------------------------------------
        self.state_manager.save_session_state(session_env)
        self.state_manager.save_concept_state(concept_env)

        response = ResponseBuilder.build_response(
            tutor_outputs,
            session_env=session_env,
            concept_env=concept_env,
            tutor_env=tutor_env,
        )

        # Convenience flags in debug
        debug = response.setdefault("debug", {})
        debug.setdefault("session_complete", session_env.is_terminated())
        debug.setdefault("concept_complete", concept_env.is_terminated())

        logger.info(
            "environment_turn_complete session_id=%s ui_mode=%s",
            session_id,
            response.get("ui_mode"),
        )

        return response


def run_environment_turn(ctx: TurnContext, cur: Any) -> Dict[str, Any]:
    """Main entry point for the environment-based tutor orchestrator.

    This matches the public interface expected by the API layer and
    QUICKSTART examples.
    """

    orchestrator = EnvironmentOrchestrator(cur)
    return orchestrator.run_turn(ctx)
