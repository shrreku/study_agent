
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

from agents.tutor.responses import build_reflect_response
from prompts import get as prompt_get, render as prompt_render

class TestReflectContext(unittest.TestCase):

    def test_reflect_prompt_rendering(self):
        """Verify reflect prompt includes history."""
        print("\nTesting Reflect Prompt Rendering...")
        
        # Mock call_json_chat to capture the prompt
        with patch("agents.tutor.responses.call_json_chat") as mock_call:
            mock_call.return_value = {"response": "Good job!", "confidence": 0.9}
            
            history = "Tutor: What is 2+2?"
            message = "It is 4."
            
            build_reflect_response(
                concept="Math",
                level="beginner",
                chunks=[],
                message=message,
                history=history
            )
            
            # Get the prompt passed to call_json_chat
            args, _ = mock_call.call_args
            prompt = args[0]
            
            print(f"  Prompt Preview: {prompt[:200]}...")
            
            self.assertIn("Recent history:", prompt)
            self.assertIn("Tutor: What is 2+2?", prompt)
            self.assertIn("Student message: It is 4.", prompt)
            
        print("✅ Reflect prompt rendering verified.")

if __name__ == "__main__":
    unittest.main()
