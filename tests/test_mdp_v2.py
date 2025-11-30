"""
Tests for MDP v2 Components

Tests:
1. Schema definitions and serialization
2. Policy action selection
3. Observation creation
4. Reward computation
5. Trajectory logging
"""

import unittest
import sys
import os
import json
import tempfile
from pathlib import Path

# Add backend to path
backend_path = str(Path(__file__).parent.parent / "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Direct imports to avoid __init__.py dependency issues
from mdp.schemas_v2 import (
    PedagogicalAction,
    StudentIntent,
    CorrectnessLevel,
    TutorState,
    TutorObservation,
    TutorAction,
    TutorReward,
    TutorTransition,
    StudentAnalysis,
    StudentProfile,
    PlanStep,
    Plan,
)
from mdp.policies_v2 import (
    RuleBasedTutorPolicy,
)
from mdp.trajectory import (
    TrajectoryLogger,
    TrajectoryStats,
)


class TestPedagogicalAction(unittest.TestCase):
    """Test PedagogicalAction enum"""
    
    def test_from_string_exact_match(self):
        """Test exact string matching"""
        action = PedagogicalAction.from_string("explain")
        self.assertEqual(action, PedagogicalAction.EXPLAIN)
        
    def test_from_string_fuzzy_match(self):
        """Test fuzzy string matching"""
        action = PedagogicalAction.from_string("question")
        self.assertEqual(action, PedagogicalAction.SOCRATIC_QUESTION)
        
        action = PedagogicalAction.from_string("hint")
        self.assertEqual(action, PedagogicalAction.GIVE_HINT)
        
    def test_from_string_fallback(self):
        """Test fallback for unknown strings"""
        action = PedagogicalAction.from_string("unknown_action")
        self.assertEqual(action, PedagogicalAction.EXPLAIN)
    
    def test_scaffolding_actions(self):
        """Test scaffolding action categorization"""
        scaffolding = PedagogicalAction.scaffolding_actions()
        self.assertIn(PedagogicalAction.SOCRATIC_QUESTION, scaffolding)
        self.assertIn(PedagogicalAction.GIVE_HINT, scaffolding)
        self.assertNotIn(PedagogicalAction.EXPLAIN, scaffolding)
    
    def test_flow_actions(self):
        """Test flow control action categorization"""
        flow = PedagogicalAction.flow_actions()
        self.assertIn(PedagogicalAction.ADVANCE_STEP, flow)
        self.assertIn(PedagogicalAction.REPLAN, flow)
        self.assertNotIn(PedagogicalAction.EXPLAIN, flow)


class TestTutorState(unittest.TestCase):
    """Test TutorState data class"""
    
    def setUp(self):
        self.state = TutorState(
            session_id="test_session",
            student_id="student_1",
            concept_id="convection",
            mastery_current=0.3,
            mastery_target=0.8,
        )
        # Add a plan
        self.state.plan = Plan(
            plan_id="plan_1",
            steps=[
                PlanStep(1, "convection", "explain", "Introduce convection"),
                PlanStep(2, "convection", "example", "Show example"),
                PlanStep(3, "convection", "question", "Check understanding"),
            ],
        )
    
    def test_current_step(self):
        """Test current step property"""
        step = self.state.current_step
        self.assertEqual(step.step_id, 1)
        self.assertEqual(step.pedagogy, "explain")
    
    def test_steps_remaining(self):
        """Test steps remaining calculation"""
        self.assertEqual(self.state.steps_remaining, 3)
        self.state.current_step_index = 2
        self.assertEqual(self.state.steps_remaining, 1)
    
    def test_mastery_gap(self):
        """Test mastery gap calculation"""
        self.assertAlmostEqual(self.state.mastery_gap, 0.5)
    
    def test_add_turn(self):
        """Test adding conversation turns"""
        self.state.add_turn("student", "I understand", mastery_delta=0.1)
        self.assertEqual(len(self.state.conversation_history), 1)
        self.assertEqual(self.state.turn_count, 1)
        self.assertEqual(len(self.state.mastery_trajectory), 1)
    
    def test_mastery_trend(self):
        """Test mastery trend calculation"""
        self.assertEqual(self.state.mastery_trend, "unknown")
        
        # Add positive deltas
        for _ in range(3):
            self.state.mastery_trajectory.append(0.1)
        self.assertEqual(self.state.mastery_trend, "improving")
        
        # Add negative deltas
        self.state.mastery_trajectory = [-0.1, -0.1, -0.1]
        self.assertEqual(self.state.mastery_trend, "struggling")
    
    def test_to_dict(self):
        """Test serialization"""
        d = self.state.to_dict()
        self.assertEqual(d["session_id"], "test_session")
        self.assertEqual(d["concept_id"], "convection")
        self.assertIn("mastery_trend", d)


class TestTutorObservation(unittest.TestCase):
    """Test TutorObservation creation and formatting"""
    
    def test_to_prompt_string(self):
        """Test prompt string formatting"""
        obs = TutorObservation(
            concept_id="convection",
            step_pedagogy="explain",
            step_index=0,
            steps_total=3,
            student_message="I don't understand heat transfer",
            student_intent=StudentIntent.CONFUSION,
            student_correctness=CorrectnessLevel.NOT_APPLICABLE,
            mastery_current=0.3,
            mastery_target=0.8,
            mastery_trend="stable",
            turn_count=2,
            hints_given=1,
        )
        
        prompt = obs.to_prompt_string()
        
        self.assertIn("CONCEPT: convection", prompt)
        self.assertIn("CURRENT_STEP: 1/3", prompt)
        self.assertIn("STUDENT_INTENT: confusion", prompt)
        self.assertIn("MASTERY: 0.30 / 0.80", prompt)
        self.assertIn("<|observation|>", prompt)
        self.assertIn("<|/observation|>", prompt)
    
    def test_to_dict(self):
        """Test dictionary serialization"""
        obs = TutorObservation(
            concept_id="convection",
            student_intent=StudentIntent.ANSWER,
            student_correctness=CorrectnessLevel.CORRECT,
        )
        
        d = obs.to_dict()
        self.assertEqual(d["concept_id"], "convection")
        self.assertEqual(d["student_intent"], "answer")
        self.assertEqual(d["correctness"], "correct")


class TestTutorAction(unittest.TestCase):
    """Test TutorAction parsing and formatting"""
    
    def test_from_raw_output(self):
        """Test parsing model output"""
        raw = """<think>
The student is confused. I should use an analogy.
</think>
[Action: use_analogy]
Think of heat like water flowing downhill..."""
        
        action = TutorAction.from_raw_output(raw)
        
        self.assertEqual(action.action, PedagogicalAction.USE_ANALOGY)
        self.assertIn("confused", action.thinking)
        self.assertIn("heat like water", action.response_text)
    
    def test_to_training_format(self):
        """Test training format output"""
        action = TutorAction(
            action=PedagogicalAction.SOCRATIC_QUESTION,
            thinking="Student needs guidance",
            response_text="What do you think happens when heat rises?",
        )
        
        formatted = action.to_training_format()
        
        self.assertIn("<think>", formatted)
        self.assertIn("[Action: socratic_question]", formatted)
        self.assertIn("What do you think", formatted)


class TestTutorReward(unittest.TestCase):
    """Test reward computation"""
    
    def test_compute_total_positive(self):
        """Test positive reward computation"""
        reward = TutorReward(
            mastery_delta=0.1,
            pedagogical_quality=0.8,
            no_answer_leakage=True,
            grounding_score=0.7,
            efficiency_score=0.6,
            scaffolding_used=True,
        )
        
        total = reward.compute_total()
        self.assertGreater(total, 0)
        self.assertLessEqual(total, 1.0)
    
    def test_compute_total_with_leakage_penalty(self):
        """Test leakage penalty"""
        reward = TutorReward(
            mastery_delta=0.1,
            pedagogical_quality=0.8,
            no_answer_leakage=False,  # Leakage!
        )
        
        total = reward.compute_total()
        self.assertLess(total, 0)  # Should be negative due to penalty
        self.assertIn("ANSWER_LEAKAGE", reward.flags)
    
    def test_to_dict(self):
        """Test serialization"""
        reward = TutorReward(mastery_delta=0.05, scaffolding_used=True)
        d = reward.to_dict()
        
        self.assertIn("total", d)
        self.assertIn("mastery_delta", d)
        self.assertIn("scaffolding_used", d)


class TestRuleBasedPolicy(unittest.TestCase):
    """Test rule-based policy decisions"""
    
    def setUp(self):
        self.policy = RuleBasedTutorPolicy()
    
    def test_correct_answer_advances(self):
        """Test that correct answers advance the step"""
        obs = TutorObservation(
            student_intent=StudentIntent.ANSWER,
            student_correctness=CorrectnessLevel.CORRECT,
            step_index=0,
            steps_total=3,
        )
        
        action = self.policy.select_action(obs)
        self.assertEqual(action.action, PedagogicalAction.ADVANCE_STEP)
    
    def test_incorrect_answer_gives_hint(self):
        """Test that incorrect answers get hints"""
        obs = TutorObservation(
            student_intent=StudentIntent.ANSWER,
            student_correctness=CorrectnessLevel.INCORRECT,
            consecutive_incorrect=0,
            hints_given=0,
        )
        
        action = self.policy.select_action(obs)
        self.assertEqual(action.action, PedagogicalAction.GIVE_HINT)
    
    def test_confusion_with_low_mastery_uses_analogy(self):
        """Test that confusion with low mastery uses analogy"""
        obs = TutorObservation(
            student_intent=StudentIntent.CONFUSION,
            mastery_current=0.2,
        )
        
        action = self.policy.select_action(obs)
        self.assertEqual(action.action, PedagogicalAction.USE_ANALOGY)


class TestTutorTransition(unittest.TestCase):
    """Test transition logging"""
    
    def test_to_dict(self):
        """Test dictionary serialization"""
        obs = TutorObservation(concept_id="convection")
        action = TutorAction(action=PedagogicalAction.EXPLAIN)
        reward = TutorReward(mastery_delta=0.05)
        
        transition = TutorTransition(
            session_id="test",
            concept_id="convection",
            turn_number=1,
            observation=obs,
            action=action,
            reward=reward,
            mastery_before=0.3,
            mastery_after=0.35,
        )
        
        d = transition.to_dict()
        
        self.assertEqual(d["session_id"], "test")
        self.assertAlmostEqual(d["mastery_delta"], 0.05, places=5)
        self.assertIsNotNone(d["observation"])
        self.assertIsNotNone(d["action"])
    
    def test_to_ppo_format(self):
        """Test PPO training format"""
        obs = TutorObservation(concept_id="convection")
        action = TutorAction(
            action=PedagogicalAction.EXPLAIN,
            thinking="Need to explain",
            response_text="Let me explain...",
        )
        reward = TutorReward(mastery_delta=0.05)
        
        transition = TutorTransition(
            observation=obs,
            action=action,
            reward=reward,
        )
        
        ppo = transition.to_ppo_format()
        
        self.assertIn("query", ppo)
        self.assertIn("response", ppo)
        self.assertIn("reward", ppo)
        self.assertIsInstance(ppo["reward"], float)


class TestTrajectoryLogger(unittest.TestCase):
    """Test trajectory logging"""
    
    def test_log_and_flush(self):
        """Test logging and flushing transitions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrajectoryLogger(output_dir=tmpdir, buffer_size=10)
            
            # Log some transitions
            for i in range(5):
                obs = TutorObservation(concept_id="convection")
                action = TutorAction(action=PedagogicalAction.EXPLAIN)
                reward = TutorReward(mastery_delta=0.01 * i)
                
                transition = TutorTransition(
                    session_id="test",
                    turn_number=i,
                    observation=obs,
                    action=action,
                    reward=reward,
                )
                logger.log(transition)
            
            # Check stats
            stats = logger.get_stats()
            self.assertEqual(stats.total_transitions, 5)
            
            # Flush
            logger.close()
            
            # Check file exists
            files = list(Path(tmpdir).glob("*.jsonl"))
            self.assertGreater(len(files), 0)
    
    def test_validate(self):
        """Test validation logic"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrajectoryLogger(output_dir=tmpdir)
            
            # Log with leakage
            for i in range(10):
                reward = TutorReward(
                    no_answer_leakage=(i < 8),  # 20% leakage
                    scaffolding_used=(i < 4),   # 40% scaffolding
                )
                transition = TutorTransition(
                    reward=reward,
                    observation=TutorObservation(),
                    action=TutorAction(),
                )
                logger.log(transition)
            
            validation = logger.validate()
            
            # Should have warnings about leakage and scaffolding
            self.assertIn("warnings", validation)
            logger.close()


if __name__ == "__main__":
    unittest.main()
