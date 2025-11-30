"""
Curriculum Runner for End-to-End RL Training Data Generation

This module orchestrates complete tutoring sessions covering all concepts
from ingested resources. It:

1. Loads resources and extracts concept sequences
2. Generates session plans respecting prerequisites
3. Simulates complete tutoring sessions with student models
4. Logs transitions for RL training (SFT, DPO, PPO formats)

The goal is to generate diverse, high-quality training data where the
tutor learns to:
- Improve student mastery efficiently
- Use appropriate pedagogical strategies
- Avoid answer leakage
- Adapt to different student types
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from mdp.schemas_v2 import (
    TutorState,
    TutorObservation,
    TutorAction,
    TutorReward,
    TutorTransition,
    StudentProfile,
    SessionPlan,
    SessionStep,
    Plan,
    PlanStep,
    PedagogicalAction,
    StudentAnalysis,
    StudentIntent,
    CorrectnessLevel,
    RecommendedAction,
)
from mdp.student_models import (
    LLMStudentSimulator,
    LearningProfile,
    LearnerType,
    StudentPopulation,
    create_student,
)
from mdp.trajectory import TrajectoryLogger, TrajectoryStats
from mdp.session_planning import SessionPlanner
from mdp.planning import LLMConceptPolicy
from mdp.policies_v2 import LLMTutorPolicy, RuleBasedTutorPolicy
from mdp.input_analysis import InputAnalyzer
from mdp.llm_client import LLMClient
from mdp.llm_config import LLMConfig, LLMClientManager
from mdp.rag import RAGTools

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SessionResult:
    """Result of a simulated tutoring session"""
    session_id: str
    student_id: str
    learner_type: str
    concepts_covered: List[str]
    
    # Mastery progression
    initial_mastery: Dict[str, float] = field(default_factory=dict)
    final_mastery: Dict[str, float] = field(default_factory=dict)
    
    # Session metrics
    total_turns: int = 0
    total_concepts: int = 0
    concepts_mastered: int = 0  # Reached target mastery
    
    # Trajectory data
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timing
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum-based data generation"""
    # Resource configuration
    resource_ids: List[str] = field(default_factory=list)
    concept_ids: List[str] = field(default_factory=list)  # Override: specific concepts
    
    # Session parameters
    max_turns_per_concept: int = 20
    target_mastery: float = 0.8
    max_concepts_per_session: int = 10
    
    # Student configuration
    students_per_concept: int = 3  # Number of different students per concept
    learner_types: Optional[List[str]] = None  # Specific types to use
    use_llm_students: bool = True
    
    # LLM Model configuration - specify different models for components
    tutor_model: Optional[str] = None      # Policy + Response (e.g., "gpt-4o", "claude-3-5-sonnet")
    student_model: Optional[str] = None    # Student simulator (e.g., "gemini-2.0-flash")
    analyzer_model: Optional[str] = None   # Input analysis (can use cheaper model)
    planner_model: Optional[str] = None    # Session planning
    
    # Output configuration
    output_dir: str = "data/trajectories"
    output_prefix: str = "curriculum"
    
    # Quality filters
    min_turns_per_concept: int = 3
    require_mastery_improvement: bool = True
    
    def to_llm_config(self) -> LLMConfig:
        """Convert to LLMConfig for client manager."""
        return LLMConfig(
            planner_model=self.planner_model,
            policy_model=self.tutor_model,
            response_model=self.tutor_model,
            analyzer_model=self.analyzer_model,
            student_model=self.student_model,
        )


# =============================================================================
# CURRICULUM RUNNER
# =============================================================================

class CurriculumRunner:
    """
    Runs complete curriculum covering all concepts with diverse students.
    
    This is the main orchestrator for training data generation.
    """
    
    def __init__(
        self,
        config: CurriculumConfig,
        llm_client: Optional[LLMClient] = None,
        llm_manager: Optional[LLMClientManager] = None,
    ):
        self.config = config
        
        # Initialize LLM clients - prefer manager for multi-model setup
        if llm_manager:
            self.llm_manager = llm_manager
        elif any([config.tutor_model, config.student_model, config.analyzer_model, config.planner_model]):
            # Build manager from config
            self.llm_manager = LLMClientManager(config.to_llm_config())
        else:
            # Single client fallback
            self.llm_manager = LLMClientManager(LLMConfig.all_same(
                llm_client.model if llm_client else "gpt-4o-mini"
            ))
        
        # Get component-specific clients
        self.llm = self.llm_manager.get_default()  # Backward compat
        self.tutor_llm = self.llm_manager.get_policy()
        self.student_llm = self.llm_manager.get_student()
        self.analyzer_llm = self.llm_manager.get_analyzer()
        self.planner_llm = self.llm_manager.get_planner()
        
        logger.info(f"LLM models configured", extra={
            "tutor": self.tutor_llm.model,
            "student": self.student_llm.model,
            "analyzer": self.analyzer_llm.model,
            "planner": self.planner_llm.model,
        })
        
        # Initialize components with appropriate LLM clients
        self.session_planner = SessionPlanner()
        self.concept_policy = LLMConceptPolicy(self.planner_llm)
        self.tutor_policy = LLMTutorPolicy(self.tutor_llm)  # Use LLMTutorPolicy for proper response generation
        self.input_analyzer = InputAnalyzer(self.analyzer_llm)  # For proper student message analysis
        self.rag = RAGTools()
        
        # Student population uses student model
        self.population = StudentPopulation(
            self.student_llm if config.use_llm_students else None
        )
        
        # Results tracking
        self.session_results: List[SessionResult] = []
        self.all_transitions: List[TutorTransition] = []
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CurriculumRunner initialized", extra={
            "resources": len(config.resource_ids),
            "concepts": len(config.concept_ids),
            "output_dir": str(self.output_dir),
        })
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full curriculum, generating training data.
        
        Returns:
            Summary statistics and file paths
        """
        start_time = time.time()
        
        # Step 1: Get concepts to cover
        concepts = self._get_concepts()
        if not concepts:
            logger.error("No concepts found to cover")
            return {"status": "error", "error": "No concepts found"}
        
        logger.info(f"Curriculum will cover {len(concepts)} concepts")
        
        # Step 2: Generate session plan
        session_plan = self._generate_session_plan(concepts)
        
        # Step 3: Run sessions with different students
        for student_idx in range(self.config.students_per_concept):
            student = self._create_student(student_idx)
            
            logger.info(f"Running session {student_idx + 1}/{self.config.students_per_concept} "
                       f"with {student.profile.learner_type.value} student")
            
            result = self._run_session(student, session_plan)
            self.session_results.append(result)
        
        # Step 4: Save results
        output_files = self._save_results()
        
        # Step 5: Compute statistics
        stats = self._compute_statistics()
        
        duration = time.time() - start_time
        
        return {
            "status": "success",
            "concepts_covered": len(concepts),
            "sessions_run": len(self.session_results),
            "total_transitions": len(self.all_transitions),
            "duration_seconds": duration,
            "output_files": output_files,
            "statistics": stats,
        }
    
    def _get_concepts(self) -> List[Dict[str, Any]]:
        """Get concepts to cover from resources or explicit list"""
        if self.config.concept_ids:
            # Use explicit concept list
            concepts = self.rag.get_concept_details(self.config.concept_ids)
            if not concepts:
                # Create minimal concept dicts if not in graph
                concepts = [{"id": cid, "name": cid} for cid in self.config.concept_ids]
            return concepts
        
        if self.config.resource_ids:
            # Get concepts from resources
            return self.rag.get_concepts_for_resources(self.config.resource_ids)
        
        return []
    
    def _generate_session_plan(self, concepts: List[Dict[str, Any]]) -> SessionPlan:
        """Generate session plan from concepts"""
        # Create dummy student profile for planning
        profile = StudentProfile(student_id="planner", mastery=0.0)
        
        concept_ids = [c.get("id") or c.get("name") for c in concepts]
        
        plan = self.session_planner.generate_plan(
            student_profile=profile,
            concept_ids=concept_ids[:self.config.max_concepts_per_session],
        )
        
        logger.info(f"Generated session plan with {len(plan.steps)} steps")
        return plan
    
    def _create_student(self, idx: int) -> LLMStudentSimulator:
        """Create a student for this session"""
        if self.config.learner_types:
            # Use specified learner types in rotation
            learner_type = self.config.learner_types[idx % len(self.config.learner_types)]
            return create_student(learner_type, self.student_llm if self.config.use_llm_students else None)
        
        # Sample from population
        return self.population.sample_student()
    
    def _run_session(
        self,
        student: LLMStudentSimulator,
        session_plan: SessionPlan,
    ) -> SessionResult:
        """Run a complete tutoring session"""
        session_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        result = SessionResult(
            session_id=session_id,
            student_id=student.state.student_id,
            learner_type=student.profile.learner_type.value,
            concepts_covered=[],
        )
        
        # Initialize trajectory logger
        trajectory_logger = TrajectoryLogger(
            output_dir=str(self.output_dir),
            session_id=f"{self.config.output_prefix}_{session_id}",
        )
        
        # Track state
        current_state = TutorState(
            session_id=session_id,
            student_id=student.state.student_id,
            session_plan=session_plan,
        )
        
        # Process each concept in the session plan
        for step_idx, session_step in enumerate(session_plan.steps):
            concept_id = session_step.concept_id or session_step.concept_name
            concept_name = session_step.concept_name or concept_id
            
            logger.info(f"Starting concept {step_idx + 1}/{len(session_plan.steps)}: {concept_name}")
            
            # Record initial mastery
            initial_mastery = student.get_mastery(concept_id)
            result.initial_mastery[concept_id] = initial_mastery
            
            # Generate concept plan
            concept_plan = self._generate_concept_plan(concept_id, student)
            current_state.plan = concept_plan
            current_state.concept_id = concept_id
            current_state.current_step_index = 0
            current_state.mastery_current = initial_mastery
            
            # Reset student for new concept
            student.reset_for_concept(concept_id)
            
            # Run concept loop
            concept_transitions = self._run_concept_loop(
                student, current_state, trajectory_logger
            )
            
            self.all_transitions.extend(concept_transitions)
            result.transitions.extend([t.to_dict() for t in concept_transitions])
            result.total_turns += len(concept_transitions)
            
            # Record final mastery
            final_mastery = student.get_mastery(concept_id)
            result.final_mastery[concept_id] = final_mastery
            result.concepts_covered.append(concept_id)
            
            if final_mastery >= self.config.target_mastery:
                result.concepts_mastered += 1
            
            logger.info(f"Completed concept {concept_name}: "
                       f"mastery {initial_mastery:.2f} -> {final_mastery:.2f}")
        
        result.total_concepts = len(session_plan.steps)
        result.duration_seconds = time.time() - start_time
        
        # Close trajectory logger
        trajectory_logger.close()
        
        return result
    
    def _generate_concept_plan(
        self,
        concept_id: str,
        student: LLMStudentSimulator,
    ) -> Plan:
        """Generate concept-level plan"""
        state_c = {
            "concept_id": concept_id,
            "student_profile": {
                "student_id": student.state.student_id,
                "mastery": student.get_mastery(concept_id),
            },
            "constraints": {"max_steps": 5},
        }
        
        try:
            return self.concept_policy.generate_plan(state_c)
        except Exception as e:
            logger.warning(f"Concept plan generation failed: {e}, using fallback")
            return Plan(
                plan_id=str(uuid.uuid4())[:8],
                concept=concept_id,
                steps=[
                    PlanStep(1, concept_id, "explain", content=f"Introduce {concept_id}"),
                    PlanStep(2, concept_id, "example", content=f"Example of {concept_id}"),
                    PlanStep(3, concept_id, "question", content=f"Practice {concept_id}"),
                ],
            )
    
    def _run_concept_loop(
        self,
        student: LLMStudentSimulator,
        state: TutorState,
        trajectory_logger: TrajectoryLogger,
    ) -> List[TutorTransition]:
        """
        Run the MDP loop for a single concept until mastery or max turns.
        
        Uses proper MDP v2 flow:
        1. InputAnalyzer for student message analysis
        2. LLMTutorPolicy.select_action() for pedagogical decisions
        3. LLMTutorPolicy.generate_response() for action-specific prompts
        4. RAG context for grounded responses
        """
        transitions = []
        turn = 0
        
        # Get RAG context for this concept
        rag_context, rag_data = self.rag.get_planning_context(state.concept_id)
        if rag_data.get("fallback"):
            logger.debug(f"Using fallback RAG context for {state.concept_id}")
        
        # Initial tutor message for first turn
        last_tutor_message = self._generate_initial_message(state)
        
        while turn < self.config.max_turns_per_concept:
            # Check termination conditions
            if state.mastery_current >= self.config.target_mastery:
                logger.debug(f"Concept mastered at turn {turn}")
                break
            
            if state.plan and state.current_step_index >= len(state.plan.steps):
                logger.debug(f"Plan completed at turn {turn}")
                break
            
            # Get current step
            current_step = state.current_step
            if not current_step:
                break
            
            # Record state before
            mastery_before = state.mastery_current
            step_before = state.current_step_index
            
            # ===== STEP 1: GET STUDENT RESPONSE =====
            student_response, student_meta = student.respond(
                tutor_message=last_tutor_message,
                concept=state.concept_id,
                step_info={"step": current_step.step_id, "pedagogy": current_step.pedagogy, "turn": turn},
            )
            
            # ===== STEP 2: ANALYZE STUDENT MESSAGE (MDP v2) =====
            # Use InputAnalyzer for proper analysis, with fallback to metadata
            try:
                analysis_result = self.input_analyzer.analyze(student_response, current_step)
                analysis = self._create_analysis_from_analyzer(analysis_result)
            except Exception as e:
                logger.debug(f"InputAnalyzer fallback: {e}")
                analysis = self._create_analysis_from_meta(student_meta)
            
            # ===== STEP 3: CREATE OBSERVATION WITH RAG CONTEXT =====
            observation = TutorObservation.from_state(
                state, analysis, student_response, rag_context
            )
            
            # ===== STEP 4: POLICY SELECTS ACTION (MDP v2) =====
            action = self.tutor_policy.select_action(observation)
            
            logger.debug(f"Policy action: {action.action.value}, thinking: {action.thinking[:50] if action.thinking else ''}")
            
            # ===== STEP 5: GENERATE RESPONSE USING POLICY (MDP v2) =====
            # Use LLMTutorPolicy.generate_response() for action-specific prompts
            if action.action not in PedagogicalAction.flow_actions():
                try:
                    action.response_text = self.tutor_policy.generate_response(
                        observation, action.action, rag_context
                    )
                except Exception as e:
                    logger.warning(f"Response generation fallback: {e}")
                    action.response_text = self._generate_fallback_response(state, action)
            
            # ===== STEP 6: UPDATE STATE =====
            mastery_delta = student_meta.get("mastery_delta", 0.0)
            state.mastery_current = student.get_mastery(state.concept_id)
            state.turn_count += 1
            
            # Handle step advancement
            if action.action == PedagogicalAction.ADVANCE_STEP:
                state.current_step_index += 1
            
            # Track hints
            if action.action in [PedagogicalAction.GIVE_HINT, PedagogicalAction.WORKED_EXAMPLE]:
                state.hints_given += 1
            
            # ===== STEP 7: COMPUTE REWARD =====
            reward = self._compute_reward(action, observation, mastery_delta)
            
            # ===== STEP 8: LOG TRANSITION =====
            transition = TutorTransition(
                session_id=state.session_id,
                concept_id=state.concept_id,
                turn_number=turn,
                observation=observation,
                action=action,
                reward=reward,
                mastery_before=mastery_before,
                mastery_after=state.mastery_current,
                step_before=step_before,
                step_after=state.current_step_index,
                is_terminal=(
                    state.mastery_current >= self.config.target_mastery or
                    (state.plan and state.current_step_index >= len(state.plan.steps))
                ),
            )
            
            transitions.append(transition)
            trajectory_logger.log(transition)
            
            # Add to conversation history
            state.add_turn("student", student_response, analysis=analysis.to_dict())
            state.add_turn("tutor", action.response_text[:200] if action.response_text else "", action=action.action.value)
            
            # Update last tutor message for next iteration
            last_tutor_message = action.response_text or ""
            
            turn += 1
            
            # Check for student give up
            if student_meta.get("intent") == "give_up":
                logger.debug("Student gave up")
                break
        
        return transitions
    
    def _generate_initial_message(self, state: TutorState) -> str:
        """Generate initial tutor message for first turn"""
        step = state.current_step
        if not step:
            return f"Let's learn about {state.concept_id}."
        
        # Use policy to generate initial message
        initial_analysis = StudentAnalysis(
            intent=StudentIntent.CONTINUE,
            correctness=CorrectnessLevel.NOT_APPLICABLE,
            recommended_action=RecommendedAction.STAY,
        )
        observation = TutorObservation.from_state(state, initial_analysis, "[Session Start]", "")
        
        try:
            # Get RAG context
            rag_context, _ = self.rag.get_planning_context(state.concept_id)
            return self.tutor_policy.generate_response(observation, PedagogicalAction.EXPLAIN, rag_context)
        except Exception:
            return f"Let me explain {step.concept}. {step.content or ''}"
    
    def _create_analysis_from_analyzer(self, result: Dict[str, Any]) -> StudentAnalysis:
        """Create StudentAnalysis from InputAnalyzer result dict."""
        intent_map = {
            "answer": StudentIntent.ANSWER,
            "question": StudentIntent.QUESTION,
            "confusion": StudentIntent.CONFUSION,
            "acknowledge": StudentIntent.ACKNOWLEDGE,
            "give_up": StudentIntent.GIVE_UP,
            "continue": StudentIntent.CONTINUE,
        }
        
        correctness_map = {
            "correct": CorrectnessLevel.CORRECT,
            "partial": CorrectnessLevel.PARTIAL,
            "incorrect": CorrectnessLevel.INCORRECT,
        }
        
        rec_map = {
            "advance": RecommendedAction.ADVANCE,
            "reply": RecommendedAction.REPLY,
            "replan": RecommendedAction.REPLAN,
            "stay": RecommendedAction.STAY,
        }
        
        intent_str = str(result.get("intent", "acknowledge")).lower()
        correctness_str = str(result.get("correctness", "")).lower()
        rec_str = str(result.get("recommended_action", "stay")).lower()
        
        return StudentAnalysis(
            intent=intent_map.get(intent_str, StudentIntent.ACKNOWLEDGE),
            correctness=correctness_map.get(correctness_str, CorrectnessLevel.NOT_APPLICABLE),
            correctness_score=result.get("correctness_score", 0.0),
            recommended_action=rec_map.get(rec_str, RecommendedAction.STAY),
            feedback_hint=result.get("feedback", ""),
            retrieval_query=result.get("retrieval_query", ""),
        )
    
    def _generate_fallback_response(self, state: TutorState, action: TutorAction) -> str:
        """Generate fallback response when policy generation fails"""
        step = state.current_step
        concept = state.concept_id
        
        templates = {
            PedagogicalAction.EXPLAIN: f"Let me explain this more clearly. {concept} involves...",
            PedagogicalAction.GIVE_HINT: f"Here's a hint: think about how {concept} relates to...",
            PedagogicalAction.SOCRATIC_QUESTION: f"What do you think happens when we apply {concept}?",
            PedagogicalAction.WORKED_EXAMPLE: f"Let's work through an example of {concept} step by step...",
            PedagogicalAction.USE_ANALOGY: f"Think of {concept} like a familiar example...",
            PedagogicalAction.CORRECT_MISCONCEPTION: f"Not quite. The key thing about {concept} is...",
            PedagogicalAction.SUMMARIZE: f"Great progress! To recap what we learned about {concept}...",
            PedagogicalAction.CONCEPT_CHECK: f"Quick check: can you explain {concept} in your own words?",
        }
        
        return templates.get(action.action, f"Let's continue with {concept}.")
    
    def _create_analysis_from_meta(self, meta: Dict[str, Any]) -> StudentAnalysis:
        """Create StudentAnalysis from student response metadata"""
        intent_map = {
            "answer": StudentIntent.ANSWER,
            "question": StudentIntent.QUESTION,
            "confusion": StudentIntent.CONFUSION,
            "acknowledge": StudentIntent.ACKNOWLEDGE,
            "give_up": StudentIntent.GIVE_UP,
        }
        
        correctness_map = {
            "correct": CorrectnessLevel.CORRECT,
            "partial": CorrectnessLevel.PARTIAL,
            "incorrect": CorrectnessLevel.INCORRECT,
            "na": CorrectnessLevel.NOT_APPLICABLE,
        }
        
        intent_str = meta.get("intent", "acknowledge")
        correctness_str = meta.get("correctness", "na")
        
        # Determine recommended action
        if correctness_str == "correct":
            rec_action = RecommendedAction.ADVANCE
        elif intent_str in ["question", "confusion"]:
            rec_action = RecommendedAction.REPLY
        else:
            rec_action = RecommendedAction.STAY
        
        return StudentAnalysis(
            intent=intent_map.get(intent_str, StudentIntent.ACKNOWLEDGE),
            correctness=correctness_map.get(correctness_str, CorrectnessLevel.NOT_APPLICABLE),
            correctness_score=meta.get("correctness_score", 0.0),
            recommended_action=rec_action,
            feedback_hint=meta.get("internal_thought", ""),
        )
    
    def _compute_reward(
        self,
        action: TutorAction,
        observation: TutorObservation,
        mastery_delta: float,
    ) -> TutorReward:
        """Compute reward for the transition"""
        import re
        
        reward = TutorReward(mastery_delta=mastery_delta)
        
        # Scaffolding check
        reward.scaffolding_used = action.action in PedagogicalAction.scaffolding_actions()
        
        # Answer leakage check
        if action.response_text:
            LEAKAGE_PATTERNS = [
                r"the answer is", r"the correct answer", r"= \d+",
                r"therefore[,\s]+\w+ equals", r"it's simply",
            ]
            response_lower = action.response_text.lower()
            for pattern in LEAKAGE_PATTERNS:
                if re.search(pattern, response_lower):
                    reward.no_answer_leakage = False
                    break
        
        # Pedagogical quality based on scaffolding
        if reward.scaffolding_used:
            reward.pedagogical_quality = 0.8
        elif action.action in PedagogicalAction.instruction_actions():
            reward.pedagogical_quality = 0.6
        else:
            reward.pedagogical_quality = 0.5
        
        # Efficiency
        if mastery_delta > 0.05:
            reward.efficiency_score = 0.8
        elif mastery_delta > 0:
            reward.efficiency_score = 0.6
        else:
            reward.efficiency_score = 0.4
        
        # Action appropriateness
        if observation.student_correctness == CorrectnessLevel.CORRECT:
            if action.action == PedagogicalAction.ADVANCE_STEP:
                reward.action_appropriate = True
                reward.pedagogical_quality = min(1.0, reward.pedagogical_quality + 0.1)
            else:
                reward.action_appropriate = False
                reward.flags.append("SHOULD_ADVANCE")
        elif observation.student_correctness == CorrectnessLevel.INCORRECT:
            if action.action == PedagogicalAction.ADVANCE_STEP:
                reward.action_appropriate = False
                reward.flags.append("PREMATURE_ADVANCE")
        
        return reward
    
    def _save_results(self) -> Dict[str, str]:
        """Save all results to files"""
        timestamp = int(time.time())
        prefix = f"{self.config.output_prefix}_{timestamp}"
        
        files = {}
        
        # Save metadata with model information
        metadata_path = self.output_dir / f"{prefix}_metadata.json"
        metadata = {
            "timestamp": timestamp,
            "prefix": prefix,
            "models": {
                "tutor": self.tutor_llm.model,
                "student": self.student_llm.model,
                "analyzer": self.analyzer_llm.model,
                "planner": self.planner_llm.model,
            },
            "config": {
                "concepts": self.config.concept_ids,
                "resources": self.config.resource_ids,
                "students_per_concept": self.config.students_per_concept,
                "learner_types": self.config.learner_types,
                "max_turns_per_concept": self.config.max_turns_per_concept,
                "target_mastery": self.config.target_mastery,
                "use_llm_students": self.config.use_llm_students,
            },
            "stats": {
                "sessions": len(self.session_results),
                "transitions": len(self.all_transitions),
            }
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        files["metadata"] = str(metadata_path)
        
        # Save session results
        results_path = self.output_dir / f"{prefix}_sessions.jsonl"
        with open(results_path, "w") as f:
            for result in self.session_results:
                f.write(json.dumps(result.to_dict()) + "\n")
        files["sessions"] = str(results_path)
        
        # Save all transitions in PPO format
        ppo_path = self.output_dir / f"{prefix}_ppo.jsonl"
        with open(ppo_path, "w") as f:
            for t in self.all_transitions:
                ppo_entry = t.to_ppo_format()
                ppo_entry["tutor_model"] = self.tutor_llm.model  # Add model info
                f.write(json.dumps(ppo_entry) + "\n")
        files["ppo"] = str(ppo_path)
        
        # Save SFT format (observation + response pairs)
        sft_path = self.output_dir / f"{prefix}_sft.jsonl"
        with open(sft_path, "w") as f:
            for t in self.all_transitions:
                if t.observation and t.action:
                    sft_entry = {
                        "input": t.observation.to_prompt_string(),
                        "output": t.action.to_training_format(),
                        "reward": t.reward.compute_total() if t.reward else 0.0,
                        "tutor_model": self.tutor_llm.model,
                    }
                    f.write(json.dumps(sft_entry) + "\n")
        files["sft"] = str(sft_path)
        
        # Save DPO pairs (high reward vs low reward for same observation type)
        dpo_path = self.output_dir / f"{prefix}_dpo.jsonl"
        dpo_pairs = self._generate_dpo_pairs()
        with open(dpo_path, "w") as f:
            for pair in dpo_pairs:
                pair["tutor_model"] = self.tutor_llm.model
                f.write(json.dumps(pair) + "\n")
        files["dpo"] = str(dpo_path)
        
        logger.info(f"Saved results to {self.output_dir}")
        return files
    
    def _generate_dpo_pairs(self) -> List[Dict[str, Any]]:
        """Generate preference pairs for DPO training"""
        pairs = []
        
        # Group transitions by observation type
        by_context = {}
        for t in self.all_transitions:
            if not t.observation or not t.action or not t.reward:
                continue
            
            # Create context key
            context = (
                t.observation.student_intent.value,
                t.observation.student_correctness.value,
            )
            
            if context not in by_context:
                by_context[context] = []
            by_context[context].append(t)
        
        # Create pairs from high/low reward samples
        for context, transitions in by_context.items():
            if len(transitions) < 2:
                continue
            
            # Sort by reward
            sorted_t = sorted(transitions, key=lambda x: x.reward.compute_total(), reverse=True)
            
            # Take best and worst
            for i, best in enumerate(sorted_t[:3]):
                for worst in sorted_t[-3:]:
                    if best.reward.compute_total() - worst.reward.compute_total() > 0.2:
                        pairs.append({
                            "prompt": best.observation.to_prompt_string(),
                            "chosen": best.action.to_training_format(),
                            "rejected": worst.action.to_training_format(),
                            "chosen_reward": best.reward.compute_total(),
                            "rejected_reward": worst.reward.compute_total(),
                        })
        
        logger.info(f"Generated {len(pairs)} DPO preference pairs")
        return pairs
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics"""
        if not self.session_results:
            return {}
        
        total_concepts = sum(r.total_concepts for r in self.session_results)
        concepts_mastered = sum(r.concepts_mastered for r in self.session_results)
        total_turns = sum(r.total_turns for r in self.session_results)
        
        # Mastery improvements
        mastery_gains = []
        for r in self.session_results:
            for concept in r.concepts_covered:
                gain = r.final_mastery.get(concept, 0) - r.initial_mastery.get(concept, 0)
                mastery_gains.append(gain)
        
        avg_gain = sum(mastery_gains) / len(mastery_gains) if mastery_gains else 0
        
        # Learner type distribution
        learner_dist = {}
        for r in self.session_results:
            lt = r.learner_type
            learner_dist[lt] = learner_dist.get(lt, 0) + 1
        
        # Reward statistics
        rewards = [t.reward.compute_total() for t in self.all_transitions if t.reward]
        
        return {
            "sessions": len(self.session_results),
            "total_concepts": total_concepts,
            "concepts_mastered": concepts_mastered,
            "mastery_rate": concepts_mastered / total_concepts if total_concepts else 0,
            "total_turns": total_turns,
            "avg_turns_per_concept": total_turns / total_concepts if total_concepts else 0,
            "avg_mastery_gain": avg_gain,
            "learner_distribution": learner_dist,
            "reward_stats": {
                "mean": sum(rewards) / len(rewards) if rewards else 0,
                "min": min(rewards) if rewards else 0,
                "max": max(rewards) if rewards else 0,
            },
            "total_transitions": len(self.all_transitions),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_curriculum(
    resource_ids: List[str] = None,
    concept_ids: List[str] = None,
    students_per_concept: int = 3,
    output_dir: str = "data/trajectories",
) -> Dict[str, Any]:
    """
    Convenience function to run curriculum and generate training data.
    
    Args:
        resource_ids: List of resource IDs to cover
        concept_ids: List of specific concept IDs (alternative to resources)
        students_per_concept: Number of different students per concept
        output_dir: Output directory for trajectory files
        
    Returns:
        Summary statistics and file paths
    """
    config = CurriculumConfig(
        resource_ids=resource_ids or [],
        concept_ids=concept_ids or [],
        students_per_concept=students_per_concept,
        output_dir=output_dir,
    )
    
    runner = CurriculumRunner(config)
    return runner.run()


def generate_training_data(
    concepts: List[str],
    num_students: int = 10,
    learner_types: List[str] = None,
    output_dir: str = "data/trajectories",
) -> Dict[str, Any]:
    """
    Generate training data for a list of concepts.
    
    Args:
        concepts: List of concept names/IDs
        num_students: Total number of student sessions
        learner_types: Specific learner types to use
        output_dir: Output directory
        
    Returns:
        Generation results and statistics
    """
    config = CurriculumConfig(
        concept_ids=concepts,
        students_per_concept=num_students,
        learner_types=learner_types,
        output_dir=output_dir,
        output_prefix="training",
    )
    
    runner = CurriculumRunner(config)
    return runner.run()
