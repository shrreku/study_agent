import sys
import os
import unittest
from unittest import mock

# Mock external dependencies expected by backend imports
mock_psycopg2 = type("MockPsycopg2", (), {})()
mock_psycopg2_extras = type("MockPsycopg2Extras", (), {"Json": object})()
sys.modules.setdefault("psycopg2", mock_psycopg2)
sys.modules.setdefault("psycopg2.extras", mock_psycopg2_extras)

# Stub core and core.db with a dummy get_db_conn so imports in agents.* succeed
mock_core = type("MockCore", (), {})()
mock_core_db = type("MockDB", (), {"get_db_conn": lambda *args, **kwargs: None})()
sys.modules.setdefault("core", mock_core)
sys.modules.setdefault("core.db", mock_core_db)

# Add backend to path so we can import agents.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from agents.tutor.tools import (  # type: ignore
    retrieval_tools,
    history_tools,
    session_plan_io,
    session_planner,
    concept_plan_io,
    concept_planner,
    step_executor,
    mastery_estimator,
    quiz_evaluator,
    turn_controls,
    turn_classification,
)
from agents.tutor.mdp.plans import SessionPlan, SessionPlanEntry, ConceptPlan, ConceptPlanStep  # type: ignore
from agents.tutor.state import TutorSessionPolicy  # type: ignore
from agents.tutor.mdp import tools_factory  # type: ignore


class TestRetrievalTools(unittest.TestCase):
    def test_retrieve_chunks_for_tutor_uses_base_module(self):
        with mock.patch.object(
            retrieval_tools._base_retrieval, "retrieve_chunks", return_value=[{"id": "c1"}]
        ) as mocked:
            out = retrieval_tools.retrieve_chunks_for_tutor("q", "r1", ["teacher"], k=5)
            self.assertEqual(out, [{"id": "c1"}])
            mocked.assert_called_once()

    def test_rehydrate_chunks_uses_base_module(self):
        with mock.patch.object(
            retrieval_tools._base_retrieval, "rehydrate_chunks_from_ids", return_value=[{"id": "c1"}]
        ) as mocked:
            out = retrieval_tools.rehydrate_chunks(["c1"])
            self.assertEqual(out, [{"id": "c1"}])
            mocked.assert_called_once_with(["c1"])


class TestHistoryTools(unittest.TestCase):
    def test_build_short_context_truncates(self):
        turns = [
            {"role": "user", "content": "hello"},
            {"role": "tutor", "content": "world"},
        ]
        ctx = history_tools.build_short_context(turns, max_chars=10)
        self.assertTrue(len(ctx) <= 10)

    def test_summarize_history_uses_summarizer(self):
        turns = [{"role": "user", "content": "hello"}]
        with mock.patch.object(history_tools, "HistorySummarizer") as HS:
            inst = HS.return_value
            inst.summarize.return_value = "summary"
            out = history_tools.summarize_history("s1", turns, current_summary="")
            self.assertEqual(out, "summary")
            inst.summarize.assert_called_once()


class TestSessionPlanner(unittest.TestCase):
    def test_session_planner_llm_basic(self):
        policy = TutorSessionPolicy.from_dict({})
        with mock.patch.object(session_planner, "call_llm_json") as call_mock, mock.patch.object(
            session_planner, "prompt_get"
        ) as get_mock, mock.patch.object(session_planner, "prompt_render") as render_mock:
            get_mock.return_value = "template"
            render_mock.return_value = "prompt"
            call_mock.return_value = {
                "plan_id": "sp-test",
                "strategy": "chronology",
                "entries": [
                    {"concept_id": "c1", "target_mastery": 0.8},
                ],
            }

            planner = session_planner.SessionPlannerLLM()
            plan = planner(
                user_id="u1",
                session_id="s1",
                strategy="chronology",
                target_concepts=["c1"],
                mastery_map={},
            )
            self.assertIsInstance(plan, SessionPlan)
            self.assertEqual(plan.plan_id, "sp-test")
            self.assertEqual(len(plan.entries), 1)

            session_plan_io.save_session_plan(plan, policy)
            loaded = session_plan_io.load_session_plan(policy)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.entries), 1)


class TestConceptPlanner(unittest.TestCase):
    def test_concept_planner_llm_basic(self):
        from agents.tutor.context_model import TutorContext  # type: ignore
        from agents.tutor.state_machine import TutorState  # type: ignore

        tutor_context = TutorContext(
            session_id="s1",
            user_id="u1",
            turn_index=0,
            message="hi",
            intent="question",
            affect="neutral",
            inferred_concept=None,
            current_state=TutorState.ORIENTATION,
            focus_concept="c1",
            concept_level="introductory",
        )

        context_obs = concept_planner.make_concept_planning_observation(
            tutor_context,
            retrieval_chunks=[{"id": "chunk1"}],
            mastery_map={"c1": {"mastery": 0.2}},
        )

        with mock.patch.object(concept_planner, "call_llm_json") as call_mock, mock.patch.object(
            concept_planner, "prompt_get"
        ) as get_mock, mock.patch.object(concept_planner, "prompt_render") as render_mock:
            get_mock.return_value = "template"
            render_mock.return_value = "prompt"
            call_mock.return_value = {
                "plan_id": "cp-s1-c1",
                "concept_id": "c1",
                "steps": [
                    {"step_id": "s1", "step_type": "EXPLAIN", "instruction": "do it"},
                ],
            }

            planner = concept_planner.ConceptPlannerLLM()
            plan = planner(
                user_id="u1",
                session_id="s1",
                concept_id="c1",
                target_mastery=0.8,
                context_obs=context_obs,
            )
            self.assertIsInstance(plan, ConceptPlan)
            self.assertEqual(plan.concept_id, "c1")
            self.assertEqual(len(plan.steps), 1)


class TestStepExecutor(unittest.TestCase):
    def test_execute_explain_step(self):
        step = ConceptPlanStep(
            step_id="s1",
            step_type="EXPLAIN",
            subgoal=None,
            instruction="explain",
            tool_call=None,
            params={},
            evaluation_type=None,
            expected_duration_steps=1,
        )
        context_obs = {
            "student_level": "introductory",
            "retrieval_chunks": [],
            "student_message": "hi",
        }

        with mock.patch.object(
            step_executor.responses, "generate_explain_response", return_value=("text", 0.5, [], None)
        ) as mocked:
            out = step_executor.execute_concept_step(
                session_id="s1",
                user_id="u1",
                concept_id="c1",
                step=step,
                context_obs=context_obs,
            )
            self.assertIn("messages", out)
            self.assertEqual(out["ui_mode"], "free_text")
            self.assertEqual(out["messages"][0]["content"], "text")
            mocked.assert_called_once()


class TestMasteryEstimator(unittest.TestCase):
    def test_heuristic_mastery_estimator_positive_delta(self):
        estimator = mastery_estimator.HeuristicMasteryEstimator()
        post, delta = estimator(
            concept_id="c1",
            mastery_before=0.5,
            recent_interactions=[
                {"intent": "answer", "affect": "engaged", "answer_correct": True},
            ],
        )
        self.assertIsNotNone(post)
        self.assertIsNotNone(delta)
        self.assertGreater(post, 0.5)


class TestQuizEvaluator(unittest.TestCase):
    def test_basic_quiz_evaluator_correct(self):
        question = {
            "question_id": "q1",
            "concept": "c1",
            "difficulty": "easy",
            "options": [
                {"id": "a", "difficulty": "easy"},
                {"id": "b", "difficulty": "easy"},
            ],
            "correct_option_id": "a",
            "explain_option_id": "explain",
        }
        user_answer = {"question_id": "q1", "option_id": "a"}
        evaluator = quiz_evaluator.BasicQuizEvaluator()
        outcome, delta = evaluator(
            question=question,
            user_answer=user_answer,
            correct_answer="a",
        )
        self.assertTrue(outcome["answer_correct"])
        self.assertIsNotNone(delta)


class TestToolsFactory(unittest.TestCase):
    class DummyConfig:
        pass

    def test_factories_return_callable_tools(self):
        cfg = self.DummyConfig()

        session_planner_tool = tools_factory.make_session_planner_tool(cfg)
        concept_planner_tool = tools_factory.make_concept_planner_tool(cfg)
        step_executor_tool = tools_factory.make_step_executor_tool(cfg)
        mastery_tool = tools_factory.make_mastery_estimator_tool(cfg)
        quiz_tool = tools_factory.make_quiz_evaluator_tool(cfg)

        self.assertTrue(callable(session_planner_tool))
        self.assertTrue(callable(concept_planner_tool))
        self.assertTrue(callable(step_executor_tool))
        self.assertTrue(callable(mastery_tool))
        self.assertTrue(callable(quiz_tool))

    def test_step_executor_adapter_delegates(self):
        cfg = self.DummyConfig()
        tool = tools_factory.make_step_executor_tool(cfg)

        step = ConceptPlanStep(
            step_id="s1",
            step_type="EXPLAIN",
            subgoal=None,
            instruction="explain",
            tool_call=None,
            params={},
            evaluation_type=None,
            expected_duration_steps=1,
        )

        context_obs = {
            "student_level": "introductory",
            "retrieval_chunks": [],
            "student_message": "hi",
        }

        with mock.patch.object(
            step_executor.responses,
            "generate_explain_response",
            return_value=("text", 0.5, [], None),
        ) as mocked:
            out = tool(
                session_id="s1",
                user_id="u1",
                concept_id="c1",
                step=step,
                context_obs=context_obs,
            )
            self.assertIn("messages", out)
            self.assertEqual(out["messages"][0]["content"], "text")
            mocked.assert_called_once()


class TestTurnControlsAndClassification(unittest.TestCase):
    def test_parse_turn_controls_mcq(self):
        payload = {"agent_action_mode": "step_by_step", "mcq_answer": {"question_id": "q1"}}
        parsed = turn_controls.parse_turn_controls(payload, message="")
        self.assertTrue(parsed.is_control_turn)
        self.assertEqual(parsed.canonical_control_label, "mcq_answer")

    def test_turn_classifier_content_turn(self):
        class DummyCtx:
            def __init__(self) -> None:
                self.target_concepts = ["c1"]
                self.message = "hi"

        session_state = {"last_concept": "c1", "target_concepts": ["c1"]}
        payload = {}
        parsed = turn_controls.parse_turn_controls(payload, message="hi")

        with mock.patch.object(
            turn_classification, "classify_message", return_value={"intent": "question", "affect": "neutral", "concept": "c1", "confidence": 0.9}
        ):
            classifier = turn_classification.TurnClassifier()
            classification = classifier.classify(DummyCtx(), session_state, parsed)
            self.assertEqual(classification.intent, "question")
            self.assertEqual(classification.concept, "c1")


if __name__ == "__main__":
    unittest.main()
