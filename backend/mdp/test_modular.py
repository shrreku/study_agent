import sys
import os
# Add backend to sys.path so we can resolve 'mdp' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

try:
    from mdp.engine import Orchestrator
    from mdp.planning import LLMConceptPolicy
    from mdp.policies import SimpleTutorPolicy
    from mdp.schemas import StudentProfile
    from mdp.llm_client import LLMClient
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test():
    print("Testing modular imports...")
    try:
        client = LLMClient()
        # Mock for testing without API key
        if not os.getenv("OPENAI_API_KEY"):
             print("Mocking LLM client...")
             def mock_call_json(s, u):
                 return {"steps": [{"concept": "Test", "pedagogy": "explain", "content": "Test Content"}]}
             client.call_json = mock_call_json
             
             def mock_chat(m):
                 return {"choices": [{"message": {"content": "Test Response"}}]}
             client.chat_completion = mock_chat

        c_policy = LLMConceptPolicy(client)
        t_policy = SimpleTutorPolicy()
        orch = Orchestrator(concept_policy=c_policy, tutor_policy=t_policy, llm_client=client)
        
        profile = StudentProfile(student_id="test_user", mastery=0.0)
        print("Starting session...")
        res = orch.start_session(profile, "TestConcept")
        print("Start Session Result:", res)
        
        if res.get("status") == "rendered":
            print("Handling button continue...")
            res2 = orch.handle_button("continue", profile)
            print("Continue Result:", res2)
        
        print("Modular test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test()
