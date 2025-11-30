"""
Tutor MDP Orchestrator v2

This module orchestrates the 3-layer tutoring system:
- Level 1: Session Planning (SessionPlanner)
- Level 2: Concept Planning (ConceptPolicy)  
- Level 3: Tutor MDP (TutorPolicy) - The trainable component

The orchestrator:
1. Maintains TutorState across the session
2. Creates TutorObservation for policy input
3. Executes TutorAction from policy
4. Computes TutorReward for training
5. Logs TutorTransition for PPO training data
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict
from typing import Dict, List, Optional, Any, Callable

from mdp.schemas_v2 import (
    TutorState,
    TutorObservation,
    TutorAction,
    TutorReward,
    TutorTransition,
    StudentAnalysis,
    ConversationTurn,
    PedagogicalAction,
    StudentIntent,
    CorrectnessLevel,
    RecommendedAction,
    StudentProfile,
    Plan,
    PlanStep,
    SessionPlan,
    ConceptPolicy,
    TutorPolicy,
    log_transition,
)
from mdp.llm_client import LLMClient
from mdp.rag import RAGTools
from mdp.mastery import MasteryModel

logger = logging.getLogger(__name__)


class TutorOrchestrator:
    """
    Main orchestrator for the Tutor MDP.
    
    Manages:
    - State tracking (TutorState)
    - Policy execution (TutorPolicy)
    - Reward computation (TutorReward)
    - Transition logging (TutorTransition)
    """
    
    def __init__(
        self,
        concept_policy: ConceptPolicy,
        tutor_policy: TutorPolicy,
        llm_client: LLMClient,
        trajectory_callback: Optional[Callable[[TutorTransition], None]] = None,
    ):
        """
        Initialize orchestrator.
        
        Args:
            concept_policy: Level 2 policy for concept planning
            tutor_policy: Level 3 policy for pedagogical decisions
            llm_client: LLM client for response generation
            trajectory_callback: Optional callback for logging transitions
        """
        self.concept_policy = concept_policy
        self.tutor_policy = tutor_policy
        self.llm = llm_client
        self.rag = RAGTools()
        self.mastery_model = MasteryModel()
        self.trajectory_callback = trajectory_callback
        
        # Internal state
        self.state = TutorState()
        self.transitions: List[TutorTransition] = []
        
        logger.info("TutorOrchestrator initialized", extra={
            "concept_policy": type(concept_policy).__name__,
            "tutor_policy": type(tutor_policy).__name__,
        })
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    def start_session(
        self, 
        student_profile: StudentProfile, 
        initial_concept: str
    ) -> Dict[str, Any]:
        """
        Start a new tutoring session.
        
        Args:
            student_profile: Student information
            initial_concept: Concept to teach
            
        Returns:
            Initial tutor response
        """
        session_id = str(uuid.uuid4())[:8]
        
        self.state = TutorState(
            session_id=session_id,
            student_id=student_profile.student_id,
            concept_id=initial_concept,
        )
        
        # Get initial mastery
        mastery_map = self.mastery_model.get_mastery(
            student_profile.student_id, [initial_concept]
        )
        self.state.mastery_current = mastery_map.get(initial_concept, 0.0)
        self.state.mastery_target = 0.8
        
        # Generate concept plan (Level 2)
        plan = self._generate_concept_plan(initial_concept, student_profile)
        self.state.plan = plan
        self.state.current_step_index = 0
        
        logger.info("session_started", extra={
            "session_id": session_id,
            "student_id": student_profile.student_id,
            "concept": initial_concept,
            "initial_mastery": self.state.mastery_current,
            "plan_steps": len(plan.steps) if plan else 0,
        })
        
        # Render first step
        return self._render_step()
    
    def start_with_session_plan(
        self, 
        student_profile: StudentProfile, 
        session_plan: SessionPlan
    ) -> Dict[str, Any]:
        """Start session with Level 1 session plan"""
        if not session_plan.steps:
            return {"status": "error", "error": "Empty session plan"}
        
        session_id = session_plan.session_id or str(uuid.uuid4())[:8]
        first_concept = session_plan.steps[0].concept_name
        
        self.state = TutorState(
            session_id=session_id,
            student_id=student_profile.student_id,
            concept_id=first_concept,
            session_plan=session_plan,
            session_step_index=0,
        )
        
        # Get mastery and generate plan
        mastery_map = self.mastery_model.get_mastery(
            student_profile.student_id, [first_concept]
        )
        self.state.mastery_current = mastery_map.get(first_concept, 0.0)
        
        plan = self._generate_concept_plan(first_concept, student_profile)
        self.state.plan = plan
        
        logger.info("session_started_with_plan", extra={
            "session_id": session_id,
            "total_concepts": len(session_plan.steps),
            "first_concept": first_concept,
        })
        
        return self._render_step()
    
    # =========================================================================
    # MESSAGE HANDLING (Main MDP Loop)
    # =========================================================================
    
    def handle_message(
        self, 
        message: str, 
        student_profile: StudentProfile
    ) -> Dict[str, Any]:
        """
        Handle student message - main MDP step.
        
        This is where the MDP loop happens:
        1. Observe: Analyze message, create observation
        2. Act: Policy selects action
        3. Execute: Generate response for action
        4. Reward: Compute reward
        5. Log: Record transition
        
        Args:
            message: Student's message
            student_profile: Student information
            
        Returns:
            Tutor response dict
        """
        if not self.state.plan or self.state.current_step_index >= len(self.state.plan.steps):
            return self._handle_session_end()
        
        # Record state before action
        mastery_before = self.state.mastery_current
        step_before = self.state.current_step_index
        
        # ===== STEP 1: ANALYZE MESSAGE =====
        analysis = self._analyze_message(message)
        
        logger.info("message_analyzed", extra={
            "message_preview": message[:50],
            "intent": analysis.intent.value,
            "correctness": analysis.correctness.value,
            "correctness_score": analysis.correctness_score,
        })
        
        # ===== STEP 2: UPDATE MASTERY =====
        new_mastery = self.mastery_model.update_mastery(
            student_profile.student_id,
            self.state.concept_id,
            analysis.correctness_score,
            hints_used=self.state.hints_given,
        )
        mastery_delta = new_mastery - self.state.mastery_current
        self.state.mastery_current = new_mastery
        
        # Track errors
        if analysis.correctness == CorrectnessLevel.INCORRECT:
            self.state.consecutive_incorrect += 1
            if analysis.key_errors:
                self.state.errors_tracked.extend(analysis.key_errors[:2])
        else:
            self.state.consecutive_incorrect = 0
        
        # ===== STEP 3: CREATE OBSERVATION =====
        # Use retrieval_queries (array) if available, fallback to retrieval_query (string)
        queries = getattr(analysis, 'retrieval_queries', None) or (
            [analysis.retrieval_query] if analysis.retrieval_query else None
        )
        retrieved_context = self._get_rag_context(queries)
        observation = TutorObservation.from_state(
            self.state, analysis, message, retrieved_context
        )
        
        logger.info("observation_created", extra={
            "concept": observation.concept_id,
            "step": f"{observation.step_index + 1}/{observation.steps_total}",
            "mastery": f"{observation.mastery_current:.2f}",
            "intent": observation.student_intent.value,
        })
        
        # ===== STEP 4: POLICY DECISION =====
        action = self.tutor_policy.select_action(observation)
        
        logger.info("policy_action_selected", extra={
            "action": action.action.value,
            "thinking_preview": action.thinking[:80] if action.thinking else "",
            "confidence": action.confidence,
        })
        
        # ===== STEP 5: EXECUTE ACTION =====
        response_result = self._execute_action(action, observation, retrieved_context)
        
        # ===== STEP 6: UPDATE STATE =====
        self._update_state_after_action(action, message, analysis, mastery_delta)
        
        # ===== STEP 7: COMPUTE REWARD =====
        reward = self._compute_reward(
            action, observation, mastery_delta, retrieved_context
        )
        
        logger.info("reward_computed", extra={
            "total": reward.compute_total(),
            "mastery_delta": reward.mastery_delta,
            "pedagogical_quality": reward.pedagogical_quality,
            "no_leakage": reward.no_answer_leakage,
            "flags": reward.flags,
        })
        
        # ===== STEP 8: LOG TRANSITION =====
        transition = TutorTransition(
            session_id=self.state.session_id,
            concept_id=self.state.concept_id,
            turn_number=self.state.turn_count,
            observation=observation,
            action=action,
            reward=reward,
            mastery_before=mastery_before,
            mastery_after=new_mastery,
            step_before=step_before,
            step_after=self.state.current_step_index,
            is_terminal=response_result.get("status") == "done",
        )
        
        self._log_transition(transition)
        
        return response_result
    
    def handle_button(
        self, 
        button: str, 
        student_profile: StudentProfile
    ) -> Dict[str, Any]:
        """Handle button click (continue, replan)"""
        if button == "replan":
            return self._handle_replan(student_profile)
        elif button == "continue":
            return self._handle_advance()
        else:
            return {"error": f"Unknown button: {button}"}
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _generate_concept_plan(
        self, 
        concept: str, 
        student_profile: StudentProfile
    ) -> Plan:
        """Generate concept plan using Level 2 policy"""
        state_c = {
            "concept_id": concept,
            "student_profile": asdict(student_profile),
            "constraints": {"max_steps": 5},
        }
        
        try:
            plan = self.concept_policy.generate_plan(state_c)
            logger.info("concept_plan_generated", extra={
                "concept": concept,
                "steps": len(plan.steps),
            })
            return plan
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            # Fallback plan
            return Plan(
                plan_id=str(uuid.uuid4())[:8],
                concept=concept,
                steps=[
                    PlanStep(1, concept, "explain", "Introduce the concept"),
                    PlanStep(2, concept, "example", "Show an example"),
                    PlanStep(3, concept, "question", "Check understanding"),
                ],
            )
    
    def _analyze_message(self, message: str) -> StudentAnalysis:
        """Analyze student message"""
        from mdp.input_analysis import InputAnalyzer
        
        current_step = self.state.current_step
        if not current_step:
            return StudentAnalysis()
        
        analyzer = InputAnalyzer(self.llm)
        raw_analysis = analyzer.analyze(message, current_step)
        
        # Convert to StudentAnalysis
        intent = StudentIntent.ACKNOWLEDGE
        intent_str = raw_analysis.get("intent", "").lower()
        if "answer" in intent_str:
            intent = StudentIntent.ANSWER
        elif "question" in intent_str:
            intent = StudentIntent.QUESTION
        elif "confusion" in intent_str:
            intent = StudentIntent.CONFUSION
        elif "off_topic" in intent_str:
            intent = StudentIntent.OFF_TOPIC
        
        correctness = CorrectnessLevel.NOT_APPLICABLE
        corr_str = raw_analysis.get("correctness", "").lower()
        if "correct" in corr_str and "incorrect" not in corr_str:
            correctness = CorrectnessLevel.CORRECT
        elif "partial" in corr_str:
            correctness = CorrectnessLevel.PARTIAL
        elif "incorrect" in corr_str:
            correctness = CorrectnessLevel.INCORRECT
        
        # Parse recommended_action
        rec_action = RecommendedAction.STAY
        rec_str = raw_analysis.get("recommended_action", "stay").lower()
        if rec_str == "advance":
            rec_action = RecommendedAction.ADVANCE
        elif rec_str == "reply":
            rec_action = RecommendedAction.REPLY
        elif rec_str == "replan":
            rec_action = RecommendedAction.REPLAN
        else:
            rec_action = RecommendedAction.STAY
        
        # Get retrieval queries (new array format) or fallback to legacy single query
        retrieval_queries = raw_analysis.get("retrieval_queries", [])
        if not retrieval_queries and raw_analysis.get("retrieval_query"):
            retrieval_queries = [raw_analysis.get("retrieval_query")]
        
        return StudentAnalysis(
            intent=intent,
            correctness=correctness,
            correctness_score=raw_analysis.get("correctness_score", 0.0),
            sentiment=raw_analysis.get("sentiment", "neutral"),
            retrieval_query=raw_analysis.get("retrieval_query"),  # Keep for backward compat
            retrieval_queries=retrieval_queries,
            feedback_hint=raw_analysis.get("feedback", ""),
            recommended_action=rec_action,
        )
    
    def _get_rag_context(self, queries: Optional[List[str]] = None) -> str:
        """
        Get RAG context for current step using multi-query search.
        
        Args:
            queries: List of 2-3 word queries from input analysis
            
        Returns:
            Formatted context string
        """
        # Build default queries if none provided
        if not queries:
            queries = []
            current_step = self.state.current_step
            if current_step:
                # Generate 2-3 word queries from step context
                queries.append(current_step.concept)
                if current_step.pedagogy and current_step.pedagogy.lower() not in ["explain", "intro"]:
                    queries.append(f"{current_step.concept} {current_step.pedagogy}")
            else:
                queries.append(self.state.concept_id)
        
        # Ensure queries is a list
        if isinstance(queries, str):
            queries = [queries]
        
        try:
            # Use multi-query search for better coverage
            if len(queries) > 1:
                context = self.rag.search_multi_query(queries, limit_per_query=2)
            else:
                context = self.rag.search_context(queries[0] if queries else self.state.concept_id)
            
            self.state.last_retrieval_query = ", ".join(queries) if queries else ""
            return context if context else ""
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            return ""
    
    def _execute_action(
        self, 
        action: TutorAction, 
        observation: TutorObservation,
        retrieved_context: str
    ) -> Dict[str, Any]:
        """Execute the selected action and generate response"""
        
        # Flow control actions
        if action.action == PedagogicalAction.ADVANCE_STEP:
            return self._handle_advance()
        elif action.action == PedagogicalAction.STAY_ON_STEP:
            return self._render_step(feedback=action.thinking)
        elif action.action == PedagogicalAction.REPLAN:
            return self._handle_replan(StudentProfile(self.state.student_id))
        elif action.action == PedagogicalAction.CONCLUDE:
            return self._handle_session_end()
        
        # Content actions - generate response
        if action.response_text:
            response_text = action.response_text
        else:
            response_text = self.tutor_policy.generate_response(
                observation, action.action, retrieved_context
            )
            action.response_text = response_text
        
        # Track hints
        if action.action in [PedagogicalAction.GIVE_HINT, PedagogicalAction.WORKED_EXAMPLE]:
            self.state.hints_given += 1
        
        # Track questions
        if action.action in [PedagogicalAction.SOCRATIC_QUESTION, PedagogicalAction.CONCEPT_CHECK]:
            self.state.questions_asked += 1
        
        return {
            "status": "rendered",
            "step": self.state.current_step,
            "action": action.action.value,
            "rendered_content": response_text,
            "thinking": action.thinking,
            "confidence": action.confidence,
        }
    
    def _update_state_after_action(
        self, 
        action: TutorAction,
        message: str,
        analysis: StudentAnalysis,
        mastery_delta: float
    ):
        """Update state after action execution"""
        # Add student turn to history
        self.state.add_turn(
            role="student",
            content=message,
            analysis=analysis.to_dict(),
            mastery_delta=mastery_delta,
        )
        
        # Add tutor turn to history
        self.state.add_turn(
            role="tutor",
            content=action.response_text[:200],
            action=action.action.value,
        )
        
        # Handle step advancement for ADVANCE action
        if action.action == PedagogicalAction.ADVANCE_STEP:
            self.state.current_step_index += 1
    
    def _compute_reward(
        self, 
        action: TutorAction,
        observation: TutorObservation,
        mastery_delta: float,
        retrieved_context: str,
        use_llm_judge: bool = False
    ) -> TutorReward:
        """
        Compute composite reward for the action taken.
        
        Reward components (based on pedagogical RL literature):
        1. Mastery delta - verifiable from student performance
        2. Scaffolding quality - bonus for guiding vs telling
        3. Answer leakage - hard penalty for giving away answers
        4. Grounding - response uses retrieved context
        5. Efficiency - fewer turns/hints to achieve mastery
        6. Action appropriateness - right action for student state
        """
        import re
        
        reward = TutorReward(mastery_delta=mastery_delta)
        
        # ===== 1. SCAFFOLDING CHECK =====
        reward.scaffolding_used = action.action in PedagogicalAction.scaffolding_actions()
        
        # ===== 2. ANSWER LEAKAGE CHECK (Hard Penalty) =====
        LEAKAGE_PATTERNS = [
            r"the answer is",
            r"the correct answer",
            r"the solution is",
            r"= \d+",  # Direct numerical answer like "= 42"
            r"therefore[,\s]+\w+ equals",
            r"so the result is",
            r"it's simply",
            r"just remember that .+ is \d+",
        ]
        
        if action.response_text:
            response_lower = action.response_text.lower()
            for pattern in LEAKAGE_PATTERNS:
                if re.search(pattern, response_lower):
                    reward.no_answer_leakage = False
                    reward.flags.append("ANSWER_LEAKAGE")
                    break
        
        # ===== 3. PEDAGOGICAL QUALITY =====
        # Score based on scaffolding indicators in response
        SCAFFOLDING_INDICATORS = [
            r"\?$",  # Ends with question
            r"what do you think",
            r"can you",
            r"try to",
            r"consider",
            r"think about",
            r"let's see",
            r"what if",
            r"how would",
            r"why do you think",
        ]
        
        if action.response_text:
            response_lower = action.response_text.lower()
            scaffolding_count = sum(
                1 for p in SCAFFOLDING_INDICATORS if re.search(p, response_lower)
            )
            base_ped_score = min(1.0, scaffolding_count * 0.15)
            
            # Bonus for scaffolding action type
            if reward.scaffolding_used:
                base_ped_score = min(1.0, base_ped_score + 0.3)
            
            reward.pedagogical_quality = max(0.4, base_ped_score)  # Floor at 0.4
        else:
            reward.pedagogical_quality = 0.5
        
        # ===== 4. GROUNDING CHECK =====
        if retrieved_context and action.response_text:
            # Filter common words for meaningful overlap
            common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "to", 
                          "of", "in", "for", "on", "with", "at", "by", "from", "as",
                          "this", "that", "it", "and", "but", "or", "not", "you", "your"}
            
            context_words = set(retrieved_context.lower().split()) - common_words
            response_words = set(action.response_text.lower().split()) - common_words
            
            if response_words:
                overlap = len(context_words & response_words)
                coverage = overlap / len(response_words)
                reward.grounding_score = min(1.0, coverage * 2)  # Scale up
            else:
                reward.grounding_score = 0.5
        else:
            reward.grounding_score = 0.6  # No context available, be lenient
        
        # ===== 5. EFFICIENCY =====
        # Reward positive mastery changes, penalize stagnation
        if mastery_delta > 0.05:
            reward.efficiency_score = min(1.0, 0.6 + mastery_delta * 3)
        elif mastery_delta > 0:
            reward.efficiency_score = 0.5 + mastery_delta * 2
        elif observation.consecutive_incorrect > 2:
            reward.efficiency_score = 0.3
            reward.flags.append("CONSECUTIVE_ERRORS")
        else:
            reward.efficiency_score = 0.5
        
        # Hint efficiency penalty (diminishing returns after 3 hints)
        if self.state.hints_given > 3:
            reward.efficiency_score *= max(0.5, 1.0 - (self.state.hints_given - 3) * 0.1)
            reward.flags.append("MANY_HINTS")
        
        # ===== 6. ACTION APPROPRIATENESS =====
        reward.action_appropriate = True  # Default
        
        # Correct answer should advance
        if observation.student_correctness == CorrectnessLevel.CORRECT:
            if action.action != PedagogicalAction.ADVANCE_STEP:
                reward.action_appropriate = False
                reward.flags.append("SHOULD_ADVANCE")
        
        # Incorrect answer should get help, not advance
        elif observation.student_correctness == CorrectnessLevel.INCORRECT:
            if action.action == PedagogicalAction.ADVANCE_STEP:
                reward.action_appropriate = False
                reward.flags.append("PREMATURE_ADVANCE")
            elif action.action in [PedagogicalAction.GIVE_HINT, 
                                   PedagogicalAction.CORRECT_MISCONCEPTION,
                                   PedagogicalAction.WORKED_EXAMPLE]:
                reward.action_appropriate = True
        
        # Bonus for appropriate action
        if reward.action_appropriate:
            reward.pedagogical_quality = min(1.0, reward.pedagogical_quality + 0.1)
        else:
            reward.pedagogical_quality = max(0.2, reward.pedagogical_quality - 0.2)
        
        # ===== 7. RESPONSE LENGTH =====
        if action.response_text:
            word_count = len(action.response_text.split())
            if word_count > 150:
                reward.response_concise = False
                reward.flags.append("RESPONSE_TOO_LONG")
                reward.efficiency_score *= 0.9
        
        logger.debug(f"reward_computed total={reward.compute_total():.3f}", extra={
            "mastery_delta": reward.mastery_delta,
            "scaffolding": reward.scaffolding_used,
            "no_leakage": reward.no_answer_leakage,
            "ped_quality": reward.pedagogical_quality,
            "grounding": reward.grounding_score,
            "efficiency": reward.efficiency_score,
            "flags": reward.flags,
        })
        
        return reward
    
    def _render_step(self, feedback: Optional[str] = None) -> Dict[str, Any]:
        """Render current step content"""
        if not self.state.plan or self.state.current_step_index >= len(self.state.plan.steps):
            return self._handle_session_end()
        
        step = self.state.current_step
        if not step:
            return {"error": "No current step"}
        
        # Get context
        context = self._get_rag_context()
        
        # Generate response for step
        from mdp.response_generator import ResponseGenerator
        generator = ResponseGenerator(self.llm, self.rag)
        
        rendered = generator.generate_pedagogical_response(
            step_concept=step.concept,
            step_pedagogy=step.pedagogy,
            step_content=step.content,
            student_history=[],
            plan_index=self.state.current_step_index,
            plan_length=len(self.state.plan.steps),
            feedback_override=feedback,
            student_mastery=self.state.mastery_current,
        )
        
        logger.info("step_rendered", extra={
            "step": f"{self.state.current_step_index + 1}/{len(self.state.plan.steps)}",
            "pedagogy": step.pedagogy,
            "concept": step.concept,
        })
        
        return {
            "status": "rendered",
            "step": step,
            "rendered_content": rendered.get("content"),
            "debug_json": rendered.get("raw_json"),
        }
    
    def _handle_advance(self) -> Dict[str, Any]:
        """Handle advancing to next step"""
        self.state.current_step_index += 1
        
        # Check if concept is done
        if self.state.plan and self.state.current_step_index >= len(self.state.plan.steps):
            # Try to advance session
            if self._try_advance_session():
                return self._render_step(
                    feedback=f"Great progress! Let's move to: {self.state.concept_id}"
                )
            return self._handle_session_end()
        
        logger.info("step_advanced", extra={
            "new_step": self.state.current_step_index,
            "total_steps": len(self.state.plan.steps) if self.state.plan else 0,
        })
        
        return self._render_step()
    
    def _handle_replan(self, student_profile: StudentProfile) -> Dict[str, Any]:
        """Handle replanning request"""
        concept = self.state.concept_id
        
        new_plan = self._generate_concept_plan(concept, student_profile)
        self.state.plan = new_plan
        self.state.current_step_index = 0
        
        logger.info("plan_regenerated", extra={
            "concept": concept,
            "new_steps": len(new_plan.steps),
        })
        
        return self._render_step(feedback="I've adjusted the lesson plan based on your needs.")
    
    def _try_advance_session(self) -> bool:
        """Try to advance to next concept in session plan"""
        if not self.state.session_plan:
            return False
        
        next_idx = self.state.session_step_index + 1
        if next_idx >= len(self.state.session_plan.steps):
            return False
        
        self.state.session_step_index = next_idx
        next_step = self.state.session_plan.steps[next_idx]
        self.state.concept_id = next_step.concept_name
        
        # Get mastery for new concept
        mastery_map = self.mastery_model.get_mastery(
            self.state.student_id, [next_step.concept_name]
        )
        self.state.mastery_current = mastery_map.get(next_step.concept_name, 0.0)
        
        # Generate new concept plan
        profile = StudentProfile(self.state.student_id)
        new_plan = self._generate_concept_plan(next_step.concept_name, profile)
        self.state.plan = new_plan
        self.state.current_step_index = 0
        
        logger.info("session_advanced", extra={
            "new_concept": next_step.concept_name,
            "session_step": f"{next_idx + 1}/{len(self.state.session_plan.steps)}",
        })
        
        return True
    
    def _handle_session_end(self) -> Dict[str, Any]:
        """Handle end of session"""
        logger.info("session_ended", extra={
            "session_id": self.state.session_id,
            "total_turns": self.state.turn_count,
            "final_mastery": self.state.mastery_current,
            "transitions_logged": len(self.transitions),
        })
        
        return {
            "status": "done",
            "session_id": self.state.session_id,
            "final_mastery": self.state.mastery_current,
            "total_turns": self.state.turn_count,
        }
    
    def _log_transition(self, transition: TutorTransition):
        """Log transition for training"""
        self.transitions.append(transition)
        log_transition(transition, logger)
        
        if self.trajectory_callback:
            try:
                self.trajectory_callback(transition)
            except Exception as e:
                logger.error(f"Trajectory callback failed: {e}")
    
    # =========================================================================
    # EXPORT METHODS
    # =========================================================================
    
    def get_transitions(self) -> List[Dict[str, Any]]:
        """Get all logged transitions as dicts"""
        return [t.to_dict() for t in self.transitions]
    
    def get_ppo_training_data(self) -> List[Dict[str, Any]]:
        """Get transitions formatted for PPO training"""
        return [t.to_ppo_format() for t in self.transitions]
    
    def export_trajectory(self, filepath: str):
        """Export trajectory to JSONL file"""
        with open(filepath, "a") as f:
            for t in self.transitions:
                f.write(json.dumps(t.to_dict()) + "\n")
        
        logger.info(f"Exported {len(self.transitions)} transitions to {filepath}")
