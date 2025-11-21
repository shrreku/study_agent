"""
Unified Decision Engine: Coordinates all decision-making layers.

Consolidates policy_llm, planning, and heuristics into a single,
state-aware decision coordinator with clear priority rules.

Decision Priority:
1. State-mandated actions (ORIENTATION, CLOSURE)
2. Safety gates (prerequisite, cold start, confusion)
3. Mode-specific logic (simple, intelligent, step_by_step)
4. Heuristic fallback
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .state_machine import TutorStateManager, TutorState
from .context_model import TutorContext


logger = logging.getLogger("tutor.decision_engine")


@dataclass
class ActionDecision:
    """Decision to take an action."""

    action: str  # "explain", "ask", "hint", "reflect", "review", "preview", "orient"
    rationale: str  # Why this action?
    retrieval_query: Optional[str]  # Query for retrieval if needed
    grounding_mode: str  # "llm_integrated", "explicit_citation", "none"
    confidence: float
    pedagogy_focus: Optional[List[str]] = None  # ["definition", "example", "application"]
    state_after: Optional[TutorState] = None  # Expected state after execution
    generated_plan: Optional[Dict[str, Any]] = None  # If a plan was generated
    cold_start: bool = False
    retrieval_strategy: Optional[str] = None  # Optional retrieval strategy hint
    max_chunks: Optional[int] = None  # Optional cap on retrieval chunk count


class TutorDecisionEngine:
    """
    Unified decision engine coordinating policy, planning, and heuristics.

    Priority-based routing:
    1. State enforcement (ORIENTATION must orient, CLOSURE must wrap)
    2. Safety gates (prerequisite check, cold start, confusion detection)
    3. Mode-specific logic (simple, intelligent, step_by_step)
    4. Heuristic fallback
    """

    def __init__(
        self,
        mode: str = "intelligent",
        enable_llm_policy: bool = True,
        enable_srl_planning: bool = True,
    ):
        """
        Initialize decision engine.

        Args:
            mode: "simple", "intelligent", or "step_by_step"
            enable_llm_policy: Use LLM for policy decisions
            enable_srl_planning: Use SRL planner for multi-step planning
        """
        self.mode = mode.lower()
        self.enable_llm_policy = enable_llm_policy and mode != "simple"
        self.enable_srl_planning = enable_srl_planning and mode != "simple"

        # Import here to avoid circular dependencies
        if self.enable_llm_policy:
            try:
                from .policy_llm import TutorPolicyLLM

                self.policy_llm: Optional[Any] = TutorPolicyLLM()
            except ImportError:
                logger.warning("Could not import TutorPolicyLLM")
                self.policy_llm = None
        else:
            self.policy_llm = None

        if self.enable_srl_planning:
            try:
                from .planning import TutorPlanner

                self.planner: Optional[Any] = TutorPlanner()
            except ImportError:
                logger.warning("Could not import TutorPlanner")
                self.planner = None
        else:
            self.planner = None

        logger.info(
            f"tutor_decision_engine_init "
            f"mode={self.mode} "
            f"policy={self.enable_llm_policy} "
            f"planning={self.enable_srl_planning}"
        )

    def decide_action(
        self,
        context: TutorContext,
        state_manager: TutorStateManager,
    ) -> ActionDecision:
        """
        Make a unified decision about what action to take.

        Follows priority chain:
        1. State-mandated actions
        2. Safety gates
        3. Mode-specific logic
        4. Fallback

        Args:
            context: Complete decision context
            state_manager: Current state machine

        Returns:
            ActionDecision with action and reasoning
        """

        logger.info(
            f"tutor_decision_start "
            f"state={context.current_state.value} "
            f"intent={context.intent} "
            f"concept={context.focus_concept}"
        )

        # LAYER 1: State-mandated actions (highest priority)
        decision = self._check_state_mandate(context, state_manager)
        if decision:
            logger.info(
                f"tutor_decision_state_mandate "
                f"action={decision.action} "
                f"reason={decision.rationale}"
            )
            return decision

        # LAYER 2: Safety gates
        decision = self._check_safety_gates(context, state_manager)
        if decision:
            logger.info(
                f"tutor_decision_safety_gate "
                f"action={decision.action} "
                f"reason={decision.rationale}"
            )
            return decision

        # LAYER 3: Mode-specific logic
        if self.mode == "simple":
            decision = self._heuristic_decision(context, state_manager)
        elif self.mode == "intelligent" or self.mode == "debug":
            decision = self._intelligent_decision(context, state_manager)
        elif self.mode == "step_by_step":
            decision = self._step_by_step_decision(context, state_manager)
        else:
            logger.warning(f"Unknown mode: {self.mode}, using heuristic")
            decision = self._heuristic_decision(context, state_manager)

        logger.info(
            f"tutor_decision_final "
            f"action={decision.action} "
            f"rationale={decision.rationale[:50]}..."
        )
        return decision

    def _check_state_mandate(
        self,
        context: TutorContext,
        state_manager: TutorStateManager,
    ) -> Optional[ActionDecision]:
        """Check if current state mandates a specific action."""

        # ORIENTATION: Must orient
        if context.current_state == TutorState.ORIENTATION:
            return ActionDecision(
                action="orient",
                rationale="ORIENTATION state: must greet and select concept",
                retrieval_query=None,
                grounding_mode="llm_integrated",
                confidence=0.95,
            )

        # CLOSURE: Must wrap up
        if context.current_state == TutorState.CLOSURE:
            return ActionDecision(
                action="preview",  # Session preview/wrap-up
                rationale="CLOSURE state: wrapping up session",
                retrieval_query=None,
                grounding_mode="none",
                confidence=0.95,
            )

        return None

    def _check_safety_gates(
        self,
        context: TutorContext,
        state_manager: TutorStateManager,
    ) -> Optional[ActionDecision]:
        """Check safety gates before proceeding with normal decision."""

        # Gate 1: Student is confused
        if context.affect == "confused":
            return ActionDecision(
                action="hint",
                rationale="Safety: Student confused, provide hint",
                retrieval_query=context.focus_concept,
                grounding_mode="llm_integrated",
                confidence=0.8,
            )

        # Gate 2: Student is frustrated
        if context.affect == "frustrated":
            return ActionDecision(
                action="reflect",
                rationale="Safety: Student frustrated, reflect on feelings",
                retrieval_query=None,
                grounding_mode="none",
                confidence=0.8,
            )

        # Gate 3: Prerequisite not mastered (in REVIEW state)
        if context.current_state == TutorState.REVIEW:
            if context.prerequisites:
                prereq = context.prerequisites[0]  # First prerequisite
                prereq_mastery = (
                    context.mastery_map.get(prereq, {}).get("mastery", 0.0)
                )
                if prereq_mastery < 0.5:
                    return ActionDecision(
                        action="explain",
                        rationale=f"Safety: Prerequisite '{prereq}' not mastered, explaining",
                        retrieval_query=prereq,
                        pedagogy_focus=["definition", "explanation"],
                        grounding_mode="llm_integrated",
                        confidence=0.85,
                        state_after=TutorState.REVIEW,
                    )

        # Gate 4: Cold Start
        if context.cold_start_eligible:
             return ActionDecision(
                action="explain",
                rationale="Safety: Cold start for new concept",
                retrieval_query=context.focus_concept,
                pedagogy_focus=["definition", "explanation", "example"],
                grounding_mode="llm_integrated",
                confidence=0.9,
                cold_start=True,
            )

        return None

    def _intelligent_decision(
        self,
        context: TutorContext,
        state_manager: TutorStateManager,
    ) -> ActionDecision:
        """
        Intelligent mode: Use policy + planning intelligently.

        Flow:
        1. Check if policy suggests multi-step planning
        2. If yes, generate plan and execute first step
        3. If no, use policy decision directly
        4. Fall back to heuristics if policy unavailable
        """

        # Try LLM policy if available
        if self.policy_llm:
            try:
                # Get policy recommendation
                policy_obs = context.to_policy_observation()
                policy_decision = self.policy_llm.decide(policy_obs)

                if policy_decision:
                    # Check if policy wants multi-step planning
                    use_planning = (
                        policy_decision.use_srl_planning
                        and self.planner
                    )

                    if use_planning:
                        # Generate and execute plan
                        planning_obs = context.to_planning_observation()
                        plan_obj = self.planner.generate_plan(
                            observation=planning_obs,
                            student_state={"mastery_map": context.mastery_map, "learning_path": context.learning_path},
                            available_actions=[
                                "explain",
                                "ask",
                                "hint",
                                "reflect",
                                "review",
                            ],
                        )

                        # Planner may return a TutorPlan dataclass or a plain dict.
                        if isinstance(plan_obj, dict):
                            plan = plan_obj
                        elif hasattr(plan_obj, "__dict__"):
                            plan = dict(plan_obj.__dict__)
                        else:
                            plan = {}

                        if plan and plan.get("steps"):
                            # Execute first step
                            first_step = plan["steps"][0]
                            return ActionDecision(
                                action=first_step.get("action", "explain"),
                                rationale=f"Policy + Plan: {first_step.get('reasoning', 'executing plan')}",
                                retrieval_query=first_step.get("retrieval_query") or context.focus_concept,
                                pedagogy_focus=first_step.get("pedagogy_focus"),
                                grounding_mode="llm_integrated",
                                confidence=0.8,
                                generated_plan=plan,
                            )

                    else:
                        # Policy says direct action (no planning)
                        next_action = (policy_decision.next_action or "explain").strip().lower()
                        
                        # Fix for "I notice you haven't shared your thoughts" hallucination:
                        # If policy says REFLECT but student just answered, force check
                        # Or if next_action is REFLECT but message is empty (sanity check)
                        if next_action == "reflect" and context.intent == "answer":
                             # If the user answered, we should probably just acknowledge and potentially ASK another question,
                             # or REFLECT on their answer. 
                             pass

                        # Enhance query generation
                        query = policy_decision.retrieval_query
                        
                        # If asking or explaining, ensure query is useful
                        if not query:
                            if next_action == "ask":
                                 query = f"{context.focus_concept} key concepts and examples"
                            elif next_action == "explain":
                                 # Check user message for specific requests like "example"
                                 msg_lower = (context.message or "").lower()
                                 if "example" in msg_lower or "instance" in msg_lower:
                                     query = f"{context.focus_concept} real world examples application"
                                 else:
                                     query = context.focus_concept
                        
                        # Improve query for "real world example" requests specifically
                        if context.intent == "question" and "example" in (context.message or "").lower():
                             query = f"{context.focus_concept} real world examples application"

                        pedagogy = policy_decision.pedagogy_focus
                        if not pedagogy and next_action == "explain":
                             pedagogy = ["definition", "explanation", "example"]
                             # Refine based on message
                             if "example" in (context.message or "").lower():
                                 pedagogy = ["example", "application"]

                        return ActionDecision(
                            action=next_action,
                            rationale=f"Policy: {policy_decision.mode}",
                            retrieval_query=query,
                            pedagogy_focus=pedagogy,
                            grounding_mode="llm_integrated",
                            confidence=0.85,
                        )

            except Exception as e:
                logger.warning(f"Policy error: {e}, falling back to classifier suggestion/heuristic")

        # If policy unavailable, try classifier-style action suggestion
        action_suggestion = getattr(context, "action_suggestion", None)
        retrieval_suggestion = getattr(context, "retrieval_suggestion", None)
        if action_suggestion is not None:
            try:
                next_action = (action_suggestion.next_action or "explain").strip().lower()
            except Exception:
                next_action = "explain"

            pedagogy = getattr(action_suggestion, "pedagogy_focus", None)

            query = None
            strategy = None
            max_chunks: Optional[int] = None
            if retrieval_suggestion is not None:
                query = getattr(retrieval_suggestion, "retrieval_query", None) or None
                strategy = getattr(retrieval_suggestion, "strategy", None) or None
                try:
                    max_val = getattr(retrieval_suggestion, "max_chunks", None)
                    if max_val is not None:
                        max_int = int(max_val)
                        if max_int > 0:
                            max_chunks = max_int
                except Exception:
                    max_chunks = None

                if not pedagogy:
                    sections = getattr(retrieval_suggestion, "prefer_sections", None)
                    if isinstance(sections, list) and sections:
                        cleaned_sections = []
                        for item in sections:
                            try:
                                v = str(item or "").strip()
                            except Exception:
                                v = ""
                            if v:
                                cleaned_sections.append(v)
                        if cleaned_sections:
                            pedagogy = cleaned_sections

            if not query:
                query = context.focus_concept

            desired_state = getattr(action_suggestion, "desired_state_after", None)
            state_after = None
            if isinstance(desired_state, str):
                try:
                    state_after = TutorState(desired_state)
                except Exception:
                    state_after = None

            rationale = "Classifier suggestion"
            phase = getattr(context, "phase_suggestion", None)
            if phase is not None:
                suggested_state = getattr(phase, "suggested_state", None)
                if suggested_state:
                    rationale = f"Classifier suggestion: phase={suggested_state}"

            return ActionDecision(
                action=next_action,
                rationale=rationale,
                retrieval_query=query,
                pedagogy_focus=pedagogy,
                grounding_mode="llm_integrated",
                confidence=0.75,
                state_after=state_after,
                retrieval_strategy=strategy,
                max_chunks=max_chunks,
            )

        # Fallback to heuristics
        return self._heuristic_decision(context, state_manager)

    def _step_by_step_decision(
        self,
        context: TutorContext,
        state_manager: TutorStateManager,
    ) -> ActionDecision:
        """
        Step-by-step mode: Follow generated plan step-by-step.
        """
        # Check for existing plan
        if context.current_plan:
            steps = context.current_plan.get("steps", [])
            idx = context.plan_step_index
            
            if 0 <= idx < len(steps):
                step = steps[idx]
                return ActionDecision(
                    action=step.get("action", "explain"),
                    rationale=f"Plan Step {idx+1}/{len(steps)}: {step.get('reasoning', 'executing step')}",
                    retrieval_query=step.get("retrieval_query"),
                    pedagogy_focus=step.get("pedagogy_focus"),
                    grounding_mode="llm_integrated",
                    confidence=0.85,
                    generated_plan=None, # Existing plan
                )
            else:
                 # Plan exhausted
                 return ActionDecision(
                    action="ask",
                    rationale="Plan completed: Checking understanding of covered topic",
                    retrieval_query=context.focus_concept,
                    pedagogy_focus=["concept_check"],
                    grounding_mode="llm_integrated",
                    confidence=0.8,
                )

        # No plan -> Generate new one
        if self.planner:
            try:
                planning_obs = context.to_planning_observation()
                plan_obj = self.planner.generate_plan(
                    observation=planning_obs,
                    student_state={"mastery_map": context.mastery_map, "learning_path": context.learning_path},
                    available_actions=[
                        "explain",
                        "ask",
                        "hint",
                        "reflect",
                        "review",
                    ],
                )

                # Planner may return a TutorPlan dataclass or a plain dict.
                if isinstance(plan_obj, dict):
                    plan = plan_obj
                elif hasattr(plan_obj, "__dict__"):
                    plan = dict(plan_obj.__dict__)
                else:
                    plan = {}

                if plan and plan.get("steps"):
                    first_step = plan["steps"][0]
                    return ActionDecision(
                        action=first_step.get("action", "explain"),
                        rationale=f"New Plan Step 1: {first_step.get('reasoning', 'starting plan')}",
                        retrieval_query=first_step.get("retrieval_query"),
                        pedagogy_focus=first_step.get("pedagogy_focus"),
                        grounding_mode="llm_integrated",
                        confidence=0.8,
                        generated_plan=plan,
                    )

            except Exception as e:
                logger.warning(f"Planning error: {e}, falling back to heuristic")

        return self._heuristic_decision(context, state_manager)

    def _heuristic_decision(
        self,
        context: TutorContext,
        state_manager: TutorStateManager,
    ) -> ActionDecision:
        """
        Heuristic fallback: Simple rule-based decision based on state and context.

        State-specific rules:
        - TEACHING: Alternate explain/ask based on counters
        - ASSESSMENT: Ask or reflect based on intent
        - REVIEW: Explain with examples
        """

        current_state = context.current_state

        # TEACHING state heuristics
        if current_state == TutorState.TEACHING:
            consecutive_explains = state_manager.state_counters.get(
                "consecutive_explains", 0
            )

            # After 2 explains, ask a question
            if consecutive_explains >= 2:
                return ActionDecision(
                    action="ask",
                    rationale="Heuristic: Asked student after 2 explains to check understanding",
                    retrieval_query=context.focus_concept,
                    grounding_mode="llm_integrated",
                    confidence=0.75,
                )

            # Default: explain
            return ActionDecision(
                action="explain",
                rationale="Heuristic: Continue explaining",
                retrieval_query=context.focus_concept,
                grounding_mode="llm_integrated",
                confidence=0.7,
            )

        # ASSESSMENT state heuristics
        if current_state == TutorState.ASSESSMENT:
            # Student answered → reflect on answer
            if context.intent == "answer":
                return ActionDecision(
                    action="reflect",
                    rationale="Heuristic: Reflect on student answer in ASSESSMENT",
                    retrieval_query=context.focus_concept,
                    grounding_mode="llm_integrated",
                    confidence=0.8,
                )

            # No answer yet → prompt again
            return ActionDecision(
                action="ask",
                rationale="Heuristic: Prompting for answer in ASSESSMENT",
                retrieval_query=context.focus_concept,
                grounding_mode="llm_integrated",
                confidence=0.75,
            )

        # REVIEW state heuristics
        if current_state == TutorState.REVIEW:
            return ActionDecision(
                action="explain",
                rationale="Heuristic: Explaining prerequisite in REVIEW state",
                retrieval_query=context.prerequisites[0] if context.prerequisites else context.focus_concept,
                grounding_mode="llm_integrated",
                confidence=0.7,
            )

        # Default fallback
        return ActionDecision(
            action="explain",
            rationale="Heuristic: Default to explain",
            retrieval_query=context.focus_concept,
            grounding_mode="llm_integrated",
            confidence=0.6,
        )
