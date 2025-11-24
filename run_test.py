import sys
import os
from unittest.mock import MagicMock

# Mock dependencies BEFORE any imports
sys.modules["psycopg2"] = MagicMock()
sys.modules["psycopg2.extras"] = MagicMock()
sys.modules["ingestion"] = MagicMock()
sys.modules["ingestion.embed"] = MagicMock()
sys.modules["metrics"] = MagicMock()
sys.modules["llm"] = MagicMock()
sys.modules["prompts"] = MagicMock()
sys.modules["core"] = MagicMock()
sys.modules["core.db"] = MagicMock()
sys.modules["backend.agents.tutor.agent"] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Now import the test
from backend.agents.tutor.test_orchestrator import test_orchestrator

if __name__ == "__main__":
    test_orchestrator()
