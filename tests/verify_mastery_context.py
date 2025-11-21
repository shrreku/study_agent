
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies
sys.modules["psycopg2"] = MagicMock()
sys.modules["psycopg2.extras"] = MagicMock()
sys.modules["core"] = MagicMock()
sys.modules["core.db"] = MagicMock()

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from agents.tutor.validators.assessment import assess_student_response

class TestMasteryContext(unittest.TestCase):

    def test_assessment_prompt_rendering(self):
        """Verify assessment prompt includes question context."""
        print("\nTesting Assessment Prompt Rendering...")
        
        # Mock call_json_chat
        # Since it's imported inside the function, we need to mock it where it comes from
        sys.modules["llm.common"] = MagicMock()
        from llm.common import call_json_chat
        
        with patch("llm.common.call_json_chat") as mock_call:
            mock_call.return_value = {"correct": True, "quality": 0.9, "reasoning": "Good"}
            
            question = "What happens if thickness doubles?"
            message = "It becomes half."
            
            assess_student_response(
                student_message=message,
                expected_concept="Heat Transfer",
                reference_chunks=[],
                question_context=question
            )
            
            # Get the prompt passed to call_json_chat
            args, _ = mock_call.call_args
            prompt = args[0]
            
            print(f"  Prompt Preview: {prompt[:200]}...")
            
            self.assertIn("Tutor asked: \"What happens if thickness doubles?\"", prompt)
            self.assertIn("Student said: \"It becomes half.\"", prompt)
            
        print("✅ Assessment prompt rendering verified.")

if __name__ == "__main__":
    unittest.main()
