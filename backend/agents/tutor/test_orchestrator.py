import sys
import os
from unittest.mock import MagicMock

# Mock psycopg2 before importing backend modules
sys.modules["psycopg2"] = MagicMock()
sys.modules["psycopg2.extras"] = MagicMock()
sys.modules["psycopg2.pool"] = MagicMock()
sys.modules["ingestion"] = MagicMock()
sys.modules["ingestion.embed"] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from backend.agents.tutor.factory import make_orchestrator
from backend.agents.tutor.context_model import TutorContext
from backend.agents.tutor.state import TutorSessionPolicy
import llm

def test_orchestrator():
    print("Initializing Orchestrator...")
    
    # Configure LLM Mock
    llm.call_llm_json.return_value = {
        "plan_id": "mock_plan",
        "strategy": "learning_path",
        "entries": [
            {"concept_id": "concept_a"},
            {"concept_id": "concept_b"}
        ]
    }

    orchestrator = make_orchestrator()
    
    print("Creating Mock Context...")
    # Mock Context with a plan
    policy_state = TutorSessionPolicy()
    policy_state.session_plan = {"concept_plan": ["concept_a", "concept_b"]}
    policy_state.session_plan_index = 0
    
    context = TutorContext(
        session_id="test_session",
        user_id="test_user",
        policy_state=policy_state,
        mastery_map={"concept_a": {"mastery": 0.1}}
    )
    
    print("Ticking Orchestrator...")
    # Tick 1: Should trigger session planning (REPLAN_SESSION) -> Concept Planning -> Pedagogy
    # Since we are using real LLM tools (or their stubs), this might fail if keys aren't set or mocks aren't perfect.
    # However, the default tools handle exceptions gracefully.
    
    try:
        response = orchestrator.tick(context, mastery_map={"concept_a": {"mastery": 0.1}})
        print("Response:", response)
    except Exception as e:
        print(f"Error during tick: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_orchestrator()
