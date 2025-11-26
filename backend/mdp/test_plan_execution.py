import sys
import os
import logging
import json

# Add backend to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load .env manually if dotenv not found
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
if os.path.exists(env_path):
    logger.info(f"Loading .env from {env_path}")
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

from mdp.engine import Orchestrator
from mdp.planning import LLMConceptPolicy
from mdp.policies import SimpleTutorPolicy
from mdp.schemas import StudentProfile
from mdp.llm_client import LLMClient
import prompts

def main():
    logger.info("Starting MDP Plan & Execution Test")
    
    # 1. Check Prompt Set
    active_set = prompts.active_set()
    logger.info(f"Active Prompt Set: {active_set}")
    
    # 2. Setup Components
    try:
        llm_client = LLMClient()
        planner = LLMConceptPolicy(llm_client)
        tutor_policy = SimpleTutorPolicy()
        orchestrator = Orchestrator(planner, tutor_policy, llm_client)
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)
        
    # 3. Setup Student & Concept
    student = StudentProfile(
        student_id="test_user",
        mastery=0.1
    )
    concept = "convection"
    
    # 4. Generate Plan
    logger.info(f"Generating plan for concept: {concept}")
    try:
        step_1_result = orchestrator.start_session(student, concept)
        plan = orchestrator.tutor_state.get("plan")
        
        if not plan:
            logger.error("Failed to generate plan")
            sys.exit(1)
            
        logger.info(f"Plan Generated with {len(plan.steps)} steps")
        for step in plan.steps:
            logger.info(f"Step {step.step_id}: {step.pedagogy} - {step.content[:50]}...")
            
        logger.info(f"Tutor Response (Step 1): {step_1_result.get('rendered_content')}")

    except Exception as e:
        logger.error(f"Planning failed: {e}")
        sys.exit(1)
        
    # 5. Execute Step 2 (Simulate 'continue')
    logger.info("\n--- Executing Step 2 (Continue) ---")
    try:
        step_2_result = orchestrator.handle_button("continue", student)
        logger.info(f"Tutor Response (Step 2): {step_2_result.get('rendered_content')}")
    except Exception as e:
        logger.error(f"Execution failed at Step 2: {e}")

if __name__ == "__main__":
    main()
