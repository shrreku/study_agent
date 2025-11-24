import sys
from unittest.mock import MagicMock, ANY

# Mock dependencies before imports
sys.modules["psycopg2"] = MagicMock()
sys.modules["psycopg2.extras"] = MagicMock()
sys.modules["psycopg2.pool"] = MagicMock()
sys.modules["ingestion"] = MagicMock()
sys.modules["ingestion.embed"] = MagicMock()

import unittest
from dataclasses import dataclass, field

from backend.agents.tutor.orchestrator import TutorOrchestrator
from backend.agents.tutor.context_model import TutorContext
from backend.agents.tutor.state import TutorSessionPolicy
from backend.agents.tutor.mdp.policy import (
    DefaultSessionPolicy, 
    SRLConceptPolicy, 
    DefaultPedagogicalTutorPolicy
)
from backend.agents.tutor.mdp.actions import ConceptMDPAction
from backend.agents.tutor.mdp.session import SessionMDPAction
from backend.agents.tutor.mdp.pedagogical_tutor import PedagogicalTutorAction
from backend.agents.tutor.mdp.plans import ConceptPlan, ConceptPlanStep, SessionPlan, SessionPlanEntry

class TestOrchestratorSignals(unittest.TestCase):

    def setUp(self):
        # Mocks
        self.session_policy = MagicMock()
        self.concept_policy = MagicMock()
        self.pedagogical_policy = MagicMock()
        self.session_planner = MagicMock()
        self.concept_planner = MagicMock()
        self.response_generator = MagicMock()
        
        self.orchestrator = TutorOrchestrator(
            session_policy=self.session_policy,
            concept_policy=self.concept_policy,
            pedagogical_policy=self.pedagogical_policy,
            session_planner=self.session_planner,
            concept_planner=self.concept_planner,
            response_generator=self.response_generator
        )
        
        # Default Context
        self.policy_state = TutorSessionPolicy(
            concept_episode_id="ce-123",
            session_plan_index=0,
            concept_episode_step_count=5,
            concept_plan={"concept_id": "C1", "steps": [{"step_type": "introduction"}]}
        )
        self.context = TutorContext(
            session_id="s1",
            user_id="u1",
            policy_state=self.policy_state
        )
        
        # Setup Session Policy to Follow Plan
        self.session_policy.decide.return_value = SessionMDPAction.FOLLOW_PLAN_CONCEPT
        # Setup Session State behavior
        self.policy_state.session_plan = {
            "concept_plan": ["C1"]
        }
        
        self.response_generator.return_value = {"messages": [], "ui_mode": "free_text"}

    def test_approval_gating_waits(self):
        # Setup: No control signal
        self.context.step_control_type = None
        self.context.confirmed_action = None
        
        # Concept Policy says Follow Plan
        self.concept_policy.decide.return_value = ConceptMDPAction.FOLLOW_PLAN_STEP
        
        # Use REAL DefaultPedagogicalTutorPolicy
        self.orchestrator.pedagogical_policy = DefaultPedagogicalTutorPolicy()
        
        response = self.orchestrator.tick(self.context, {})
        
        # Verify Ped Policy returned WAIT_FOR_CONFIRMATION
        # response_generator is called
        self.response_generator.assert_called()
        args, kwargs = self.response_generator.call_args
        self.assertEqual(kwargs['pedagogical_action'], PedagogicalTutorAction.WAIT_FOR_CONFIRMATION)

    def test_continue_advances(self):
        # Setup: Continue signal
        self.context.step_control_type = "continue"
        
        self.concept_policy.decide.return_value = ConceptMDPAction.FOLLOW_PLAN_STEP
        self.orchestrator.pedagogical_policy = DefaultPedagogicalTutorPolicy()
        
        self.orchestrator.tick(self.context, {})
        
        # Verify Ped Policy proceeds
        self.response_generator.assert_called()
        args, kwargs = self.response_generator.call_args
        self.assertIn(kwargs['pedagogical_action'], [PedagogicalTutorAction.EXPLAIN, PedagogicalTutorAction.ASK_QUESTION])
        self.assertNotEqual(kwargs['pedagogical_action'], PedagogicalTutorAction.WAIT_FOR_CONFIRMATION)

    def test_replan_auto_executes(self):
        # Setup: Replan signal
        self.context.step_control_type = "replan_concept"
        
        # 1. Concept Policy is called first.
        self.concept_policy.decide.side_effect = [
            ConceptMDPAction.REPLAN_CONCEPT, 
            ConceptMDPAction.FOLLOW_PLAN_STEP
        ]
        
        # Mock Concept Planner to return a new plan
        new_plan = ConceptPlan(
            plan_id="new_plan",
            concept_id="C1",
            target_mastery=0.9,
            steps=[ConceptPlanStep(step_id="s1", step_type="explanation", instruction="New Step 1")]
        )
        self.concept_planner.return_value = new_plan
        
        self.orchestrator.pedagogical_policy = DefaultPedagogicalTutorPolicy()
        
        self.orchestrator.tick(self.context, {})
        
        # Verify Replan happened
        self.concept_planner.assert_called()
        self.assertEqual(self.concept_policy.decide.call_count, 2)
        
        # Verify Ped Policy proceeds (should NOT wait)
        self.response_generator.assert_called()
        args, kwargs = self.response_generator.call_args
        self.assertNotEqual(kwargs['pedagogical_action'], PedagogicalTutorAction.WAIT_FOR_CONFIRMATION)
        # Should execute the first step of NEW plan
        self.assertEqual(kwargs['ped_state'].plan_id, "new_plan")

if __name__ == '__main__':
    unittest.main()
