from __future__ import annotations

import os
from typing import Optional

from ..constants import logger
from ..decision_engine import TutorDecisionEngine, ActionDecision
from ..context_model import TutorContext
from ..state_machine import TutorStateManager, TutorState
from ..state import TutorSessionPolicy
from ..classifiers import TurnSignalsClassifier
from ..turn_signals_classifier import classify_turn_signals


class StepByStepSRLRuntime:
    """Step-by-step SRL teaching runtime.

    This runtime is responsible for coordinating step-by-step mode behavior.
    For now it primarily wraps the unified TutorDecisionEngine in step_by_step
    mode, while enforcing a configurable target mastery threshold and exposing
    env-driven mastery step delta knobs for future use.

    In the V3 architecture this class is treated as an internal implementation
    detail of the StepEngine; callers should route decisions through
    ``StepEngine.decide_step(control_mode="step", ...)`` rather than invoking
    this runtime directly from the orchestrator or API layers.
    """

    def __init__(
        self,
        *,
        mode: str,
        decision_engine: TutorDecisionEngine,
        target_mastery: float,
        step_delta_base: float,
        step_delta_max: float,
    ) -> None:
        self.mode = (mode or "").strip().lower()
        self.decision_engine = decision_engine
        self.target_mastery = float(target_mastery)
        self.step_delta_base = float(step_delta_base)
        self.step_delta_max = float(step_delta_max)
        self.turn_signals_classifier: TurnSignalsClassifier = _DefaultTurnSignalsClassifier()

    @classmethod
    def from_env(
        cls,
        *,
        mode: str,
        decision_engine: TutorDecisionEngine,
    ) -> "StepByStepSRLRuntime":
        """Build runtime with env-configured mastery parameters.

        Env vars:
            TUTOR_STEP_SRL_TARGET_MASTERY: float in [0,1], default 0.8
            TUTOR_STEP_SRL_STEP_DELTA_BASE: float, default 0.1
            TUTOR_STEP_SRL_STEP_DELTA_MAX: float, default 0.2
        """

        def _get_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except Exception:
                try:
                    logger.warning(
                        "invalid_step_srl_env_value", extra={"env": name, "value": raw}
                    )
                except Exception:
                    pass
                return default

        target_mastery = _get_float("TUTOR_STEP_SRL_TARGET_MASTERY", 0.8)
        step_delta_base = _get_float("TUTOR_STEP_SRL_STEP_DELTA_BASE", 0.1)
        step_delta_max = _get_float("TUTOR_STEP_SRL_STEP_DELTA_MAX", 0.2)

        try:
            logger.info(
                "step_srl_runtime_init",
                extra={
                    "mode": (mode or "").strip().lower(),
                    "target_mastery": target_mastery,
                    "step_delta_base": step_delta_base,
                    "step_delta_max": step_delta_max,
                },
            )
        except Exception:
            pass

        return cls(
            mode=mode,
            decision_engine=decision_engine,
            target_mastery=target_mastery,
            step_delta_base=step_delta_base,
            step_delta_max=step_delta_max,
        )

    def decide_action(
        self,
        context: TutorContext,
        state_manager: TutorStateManager,
        policy_state: TutorSessionPolicy,
        override_type: Optional[str] = None,
        override_params: Optional[dict] = None,
    ) -> ActionDecision:
        """Produce an ActionDecision for step-by-step SRL mode.

        Behavior:
        - If the current focus concept already meets or exceeds the
          env-configured target mastery threshold, transition the
          state machine into CLOSURE and delegate to the unified
          TutorDecisionEngine (which will emit a closure preview).
        - Otherwise, delegate directly to the unified TutorDecisionEngine,
          which will route into its step-by-step planning logic when
          mode is "step_by_step".
        """

        focus_concept = context.focus_concept
        current_mastery: Optional[float] = None

        turn_signals = None
        message = getattr(context, "message", None)
        has_message = isinstance(message, str) and bool(message.strip())
        if has_message:
            try:
                turn_signals = self.turn_signals_classifier.classify(context)
            except Exception:
                turn_signals = None

        if turn_signals is not None:
            try:
                context.turn_signals = turn_signals
            except Exception:
                pass

            try:
                if getattr(turn_signals, "wants_closure", False):
                    state_manager.current_state = TutorState.CLOSURE
            except Exception:
                pass

        # Normalize override type (if any) for control-surface behaviors.
        try:
            normalized_override = (
                override_type.strip().lower()  # type: ignore[union-attr]
                if isinstance(override_type, str)
                else ""
            )
        except Exception:
            normalized_override = ""

        # Explicit session_end override: move state machine to CLOSURE so the
        # unified decision engine emits a wrap-up / closure response.
        if normalized_override == "session_end":
            try:
                logger.info(
                    "step_srl_override_session_end",
                    extra={
                        "concept": focus_concept,
                        "state": getattr(state_manager, "current_state", None),
                    },
                )
            except Exception:
                pass

            try:
                state_manager.current_state = TutorState.CLOSURE
            except Exception:
                pass

            return self.decision_engine.decide_action(context, state_manager)

        # Explicit step_skip_to_assessment override: treat the current SRL plan
        # as exhausted so that the step-by-step decision path transitions into
        # an assessment / quiz question for this concept.
        if normalized_override == "step_skip_to_assessment":
            try:
                plan_obj = getattr(context, "current_plan", None)
            except Exception:
                plan_obj = None
            steps_len = 0
            if plan_obj is not None:
                try:
                    steps = getattr(plan_obj, "steps", None) or []
                except Exception:
                    steps = []
                try:
                    steps_len = len(steps)
                except Exception:
                    steps_len = 0

            if steps_len > 0:
                try:
                    context.plan_step_index = steps_len
                except Exception:
                    pass
                try:
                    policy_state.srl_plan_step_index = steps_len
                except Exception:
                    pass

                try:
                    logger.info(
                        "step_srl_override_skip_to_assessment",
                        extra={
                            "concept": focus_concept,
                            "steps_len": steps_len,
                        },
                    )
                except Exception:
                    pass

        if focus_concept:
            try:
                current_mastery = float(
                    (context.mastery_map.get(focus_concept) or {}).get("mastery", 0.0)
                    or 0.0
                )
            except Exception:
                current_mastery = 0.0

        if (
            focus_concept
            and current_mastery is not None
            and current_mastery >= self.target_mastery
        ):
            # Concept already at or above target mastery: move to CLOSURE and
            # let the unified decision engine apply its state mandate.
            try:
                logger.info(
                    "step_srl_target_mastery_reached",
                    extra={
                        "concept": focus_concept,
                        "mastery": current_mastery,
                        "target_mastery": self.target_mastery,
                    },
                )
            except Exception:
                pass

            try:
                state_manager.current_state = TutorState.CLOSURE
            except Exception:
                pass

        # Delegate to the unified decision engine, which will use its
        # step-by-step decision path when mode is "step_by_step" and will
        # honor any state changes (e.g., CLOSURE) we applied above.
        return self.decision_engine.decide_action(context, state_manager)


class _DefaultTurnSignalsClassifier:
    def classify(self, context: TutorContext):
        return classify_turn_signals(context)
