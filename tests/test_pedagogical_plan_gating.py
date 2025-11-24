import sys
import os
import unittest

# Mock external dependencies that backend imports may expect
sys.modules["psycopg2"] = type("Mock", (), {})()
mock_psycopg2_extras = type("MockPsycopg2Extras", (), {"Json": object})()
sys.modules["psycopg2.extras"] = mock_psycopg2_extras

# Stub core and core.db with a dummy get_db_conn so imports in agents.* succeed
mock_core = type("MockCore", (), {})()
mock_core_db = type("MockDB", (), {"get_db_conn": lambda *args, **kwargs: None})()
sys.modules.setdefault("core", mock_core)
sys.modules.setdefault("core.db", mock_core_db)

# Add backend to path so we can import agents.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from agents.tutor.mdp.pedagogical_tutor import (
    PedagogicalTutorState,
    PedagogicalTutorObservation,
    PedagogicalTutorAction,
    build_pedagogical_tutor_state,
    build_pedagogical_tutor_observation,
    apply_pedagogical_transition,
)
from agents.tutor.mdp.plans import ConceptPlan, ConceptPlanStep
from agents.tutor.mdp.policy import DefaultPedagogicalTutorPolicy


class TestPedagogicalPlanGating(unittest.TestCase):
    def _make_plan(self) -> ConceptPlan:
        steps = [
            ConceptPlanStep(
                step_id="s-intro",
                step_type="introduction",
                subgoal=None,
                instruction="Introduce concept",
                tool_call=None,
                params={},
                evaluation_type=None,
                expected_duration_steps=1,
            ),
            ConceptPlanStep(
                step_id="s-practice",
                step_type="practice",
                subgoal=None,
                instruction="Practice question",
                tool_call=None,
                params={},
                evaluation_type=None,
                expected_duration_steps=1,
            ),
            ConceptPlanStep(
                step_id="s-reflect",
                step_type="reflection",
                subgoal=None,
                instruction="Reflect on learning",
                tool_call=None,
                params={},
                evaluation_type=None,
                expected_duration_steps=1,
            ),
        ]
        return ConceptPlan(
            plan_id="cp-test",
            concept_id="c1",
            steps=steps,
            created_at_step=0,
            source="test",
            initial_mastery=None,
            target_mastery=0.8,
        )

    def test_first_step_auto_exec_then_gated_by_continue(self):
        plan = self._make_plan()
        policy = DefaultPedagogicalTutorPolicy()

        # Initial pedagogical state at plan_index=0 with no history.
        state0: PedagogicalTutorState = build_pedagogical_tutor_state(
            episode_id="pt-e1",
            session_id="s1",
            user_id="u1",
            concept_id="c1",
            plan_id=plan.plan_id,
            plan_index=0,
            plan_length=len(plan.steps),
            phase="instruction",
            mastery=0.2,
            target_mastery=0.8,
            last_intent="ask_question",
            last_affect="neutral",
            last_control_type=None,
            awaiting_mcq_answer=False,
            last_mcq_answered=False,
            last_quiz_correct=None,
            previous_state=None,
            last_pedagogical_action=None,
        )

        obs0: PedagogicalTutorObservation = build_pedagogical_tutor_observation(state=state0)

        # First plan step should auto-execute even without an explicit control signal.
        action0 = policy.decide(observation=obs0, concept_plan=plan)
        self.assertEqual(action0, PedagogicalTutorAction.EXPLAIN)

        # Apply transition to advance plan_index and record history.
        outcome0 = apply_pedagogical_transition(
            prev_state=state0,
            mdp_action=action0,
            mastery_delta=None,
            quiz_delta=None,
            last_quiz_correct=None,
            awaiting_mcq_answer=False,
            last_mcq_answered=False,
            last_control_type=None,
        )
        state1 = outcome0.state

        # With no explicit continue, subsequent step should be gated.
        obs1 = build_pedagogical_tutor_observation(state=state1)
        self.assertEqual(obs1.plan_index, 1)
        self.assertIsNone(obs1.last_control_type)

        action1 = policy.decide(observation=obs1, concept_plan=plan)
        self.assertEqual(action1, PedagogicalTutorAction.WAIT_FOR_CONFIRMATION)

        # Now simulate the user pressing "continue" for the next turn.
        state1_with_control: PedagogicalTutorState = build_pedagogical_tutor_state(
            episode_id=state1.episode_id,
            session_id=state1.session_id,
            user_id=state1.user_id,
            concept_id=state1.concept_id,
            plan_id=state1.plan_id,
            plan_index=state1.plan_index,
            plan_length=state1.plan_length,
            phase=state1.phase,
            mastery=state1.mastery,
            target_mastery=state1.target_mastery,
            last_intent=state1.last_intent,
            last_affect=state1.last_affect,
            last_control_type="continue",
            awaiting_mcq_answer=state1.awaiting_mcq_answer,
            last_mcq_answered=state1.last_mcq_answered,
            last_quiz_correct=state1.last_quiz_correct,
            previous_state=state1,
            last_pedagogical_action=state1.last_pedagogical_action,
        )

        obs2 = build_pedagogical_tutor_observation(state=state1_with_control)
        self.assertEqual(obs2.plan_index, 1)
        self.assertEqual(obs2.last_control_type, "continue")

        # Now the second plan step (practice) should execute as ASK_QUESTION.
        action2 = policy.decide(observation=PedagogicalTutorObservation(**obs2.__dict__), concept_plan=plan)
        self.assertEqual(action2, PedagogicalTutorAction.ASK_QUESTION)

        # Advance again.
        outcome1 = apply_pedagogical_transition(
            prev_state=state1_with_control,
            mdp_action=action2,
            mastery_delta=None,
            quiz_delta=None,
            last_quiz_correct=None,
            awaiting_mcq_answer=False,
            last_mcq_answered=False,
            last_control_type="continue",
        )
        state2 = outcome1.state

        # Simulate replan: new plan with a fresh first step and explicit replan_concept control.
        new_plan = self._make_plan()
        state_replan: PedagogicalTutorState = build_pedagogical_tutor_state(
            episode_id=state2.episode_id,
            session_id=state2.session_id,
            user_id=state2.user_id,
            concept_id=state2.concept_id,
            plan_id=new_plan.plan_id,
            plan_index=0,
            plan_length=len(new_plan.steps),
            phase=state2.phase,
            mastery=state2.mastery,
            target_mastery=state2.target_mastery,
            last_intent=state2.last_intent,
            last_affect=state2.last_affect,
            last_control_type="replan_concept",
            awaiting_mcq_answer=False,
            last_mcq_answered=False,
            last_quiz_correct=None,
            previous_state=None,
            last_pedagogical_action=None,
        )

        obs_replan = build_pedagogical_tutor_observation(state=state_replan)
        self.assertEqual(obs_replan.plan_index, 0)
        self.assertEqual(obs_replan.last_control_type, "replan_concept")

        # After replan, the first step of the new plan should auto-execute.
        action_replan = policy.decide(
            observation=PedagogicalTutorObservation(**obs_replan.__dict__),
            concept_plan=new_plan,
        )
        self.assertEqual(action_replan, PedagogicalTutorAction.EXPLAIN)


if __name__ == "__main__":
    unittest.main()
