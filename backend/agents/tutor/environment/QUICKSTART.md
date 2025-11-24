# Quick Start Guide - 3-Layer MDP Environment

This guide will help you get started with the new environment architecture in 5 minutes.

## Basic Usage

### 1. Import the Orchestrator

```python
from backend.agents.tutor.environment import run_environment_turn
```

### 2. Create a Turn Context

```python
from backend.agents.tutor.runtime.context import TurnContext

ctx = TurnContext(
    session_id="sess_001",
    user_id="user_123",
    message="I want to learn",
    target_concepts=["Concept_A", "Concept_B"],  # For new sessions
    turn_index=0,
    payload={}  # Or {"button_clicked": True} for button clicks
)
```

### 3. Run a Turn

```python
# Get database cursor (however your app does this)
cur = db.cursor()

# Run one turn through all 3 layers
response = run_environment_turn(ctx, cur)

# Response structure:
{
    "messages": [
        {"role": "assistant", "content": "Let me explain..."}
    ],
    "ui_mode": "buttons_only",
    "button_options": ["Continue"],
    "mcq_payload": None,
    "agent_action_mode": "environment_v1",
    "debug": {
        "session": {...},
        "concept": {...},
        "tutor": {...}
    }
}
```

## Complete Example

```python
# Initialize session
ctx = TurnContext(
    session_id="sess_001",
    user_id="user_123",
    message="Start learning",
    target_concepts=["Heat_Transfer", "Convection"],
    turn_index=0,
    payload={}
)

# First turn - creates session plan and concept plan
response1 = run_environment_turn(ctx, cur)
print(response1["messages"][0]["content"])
# → "Let me explain Heat Transfer..."

# User clicks Continue button
ctx.turn_index = 1
ctx.payload = {"button_clicked": True}

# Second turn - advances to next step
response2 = run_environment_turn(ctx, cur)
print(response2["messages"][0]["content"])
# → "Here's an example..."

# Continue until concept complete
# Then moves to next concept automatically
```

## Understanding the Flow

Each turn goes through 3 layers:

```
User → [Session Layer] → [Concept Layer] → [Tutor Layer] → Response
         ↓                  ↓                 ↓
    Which concept?    What step?        What to say?
```

### Layer 1: Session
- Decides which concept to study
- Tracks progress through concepts
- Ends session when all done

### Layer 2: Concept
- Generates learning plan (via LLM)
- Tracks which step to execute
- Monitors mastery

### Layer 3: Tutor
- Executes the plan step
- Generates messages
- Creates UI controls

## Direct Environment Usage

You can also use environments directly:

```python
from backend.agents.tutor.environment import (
    SessionEnvironment,
    ConceptEnvironment,
    TutorEnvironment,
    SessionAction,
    ConceptAction,
    TutorAction,
)

# Create session environment
session_env = SessionEnvironment(
    session_id="sess_001",
    user_id="user_123",
)

# Set a plan
from backend.agents.tutor.mdp.plans import SessionPlan, SessionPlanEntry
plan = SessionPlan(
    strategy="sequential",
    entries=[
        SessionPlanEntry(concept_id="Concept_A"),
        SessionPlanEntry(concept_id="Concept_B"),
    ]
)
session_env.set_session_plan(plan)

# Execute action
transition = session_env.step(SessionAction.START_CONCEPT)
print(transition.outputs)
# → {"focus_concept": "Concept_A"}
```

## Concept Plan Generation

The environment automatically generates concept plans using LLM:

```python
# Happens automatically in orchestrator, but you can do it manually:
from backend.agents.tutor.mdp.tools_factory import make_concept_planner_tool

planner = make_concept_planner_tool()

plan = planner(
    user_id="user_123",
    session_id="sess_001",
    concept_id="Heat_Transfer",
    target_mastery=0.8,
    context_obs={
        "concept_id": "Heat_Transfer",
        "current_mastery": 0.0,
        "target_mastery": 0.8,
    }
)

# Plan will have steps like:
# - step_type: "explain", instruction: "Explain the concept..."
# - step_type: "example", instruction: "Show an example..."
# - step_type: "practice", instruction: "Practice problem..."
```

## Handling User Interactions

### Button Clicks

```python
# User clicks "Continue" button
ctx.payload = {"button_clicked": True}
response = run_environment_turn(ctx, cur)
# → Environment advances to next step
```

### Different Button Types

```python
# Future support for other buttons
ctx.payload = {
    "button_clicked": True,
    "button_label": "Skip to Quiz"  # Will be handled in policy
}
```

## Accessing State Information

The debug field in responses contains full state:

```python
response = run_environment_turn(ctx, cur)

# Session state
session_state = response["debug"]["session"]
print(f"Current concept index: {session_state['current_concept_index']}")
print(f"Concepts completed: {session_state['concepts_completed']}")

# Concept state
concept_state = response["debug"]["concept"]
print(f"Current step: {concept_state['current_step_index']}")
print(f"Mastery: {concept_state['current_mastery']:.2f}")

# Tutor state
tutor_state = response["debug"]["tutor"]
print(f"Last action: {tutor_state['last_action']}")
```

## Customizing Policies

Replace default policies with your own:

```python
from backend.agents.tutor.environment import EnvironmentOrchestrator

orchestrator = EnvironmentOrchestrator(cur)

# Replace with custom policy
class MyConceptPolicy:
    def decide(self, state, step_complete=False, **kwargs):
        # Your logic here
        if state.current_mastery > 0.9:
            return ConceptAction.COMPLETE_CONCEPT
        return ConceptAction.EXECUTE_STEP

orchestrator.concept_policy = MyConceptPolicy()

# Now all turns use your custom policy
```

## Testing

### Unit Test Example

```python
def test_session_environment():
    from backend.agents.tutor.environment import SessionEnvironment
    from backend.agents.tutor.mdp.plans import SessionPlan, SessionPlanEntry
    
    # Create environment
    env = SessionEnvironment("sess_test", "user_test")
    
    # Set plan
    plan = SessionPlan(
        strategy="sequential",
        entries=[SessionPlanEntry(concept_id="C1")]
    )
    env.set_session_plan(plan)
    
    # Test action
    from backend.agents.tutor.environment import SessionAction
    transition = env.step(SessionAction.START_CONCEPT)
    
    assert not transition.terminated
    assert env.get_current_concept() == "C1"
```

### Integration Test Example

```python
def test_full_flow():
    # Mock database cursor
    cur = MockCursor()
    
    # Create context
    ctx = TurnContext(
        session_id="test_001",
        user_id="test_user",
        message="start",
        target_concepts=["C1", "C2"],
        turn_index=0,
        payload={}
    )
    
    # Run multiple turns
    responses = []
    for i in range(5):
        ctx.turn_index = i
        if i > 0:
            ctx.payload = {"button_clicked": True}
        
        response = run_environment_turn(ctx, cur)
        responses.append(response)
        
        if response.get("debug", {}).get("session_complete"):
            break
    
    # Verify we got responses
    assert len(responses) > 0
    assert all("messages" in r for r in responses)
```

## Troubleshooting

### Issue: "No plan available"

```python
# Make sure target_concepts is provided for new sessions
ctx = TurnContext(
    ...,
    target_concepts=["Concept_A"]  # Required!
)
```

### Issue: "Session immediately terminates"

```python
# Check that session plan has entries
env = SessionEnvironment(...)
env.set_session_plan(plan)
assert len(plan.entries) > 0
```

### Issue: "Concept plan is empty"

```python
# The LLM planner might have failed
# Check logs for "environment_plan_generated"
# Verify LLM configuration is correct
```

## Common Patterns

### Pattern 1: New Session Setup

```python
# First turn of a new session
def start_new_session(user_id, concepts):
    ctx = TurnContext(
        session_id=f"sess_{generate_id()}",
        user_id=user_id,
        message="",
        target_concepts=concepts,
        turn_index=0,
        payload={}
    )
    return run_environment_turn(ctx, cur)
```

### Pattern 2: Continue Existing Session

```python
# Subsequent turns
def continue_session(session_id, user_id, turn_index):
    ctx = TurnContext(
        session_id=session_id,
        user_id=user_id,
        message="",
        target_concepts=None,  # Will load from state
        turn_index=turn_index,
        payload={"button_clicked": True}
    )
    return run_environment_turn(ctx, cur)
```

### Pattern 3: Check Session Status

```python
response = run_environment_turn(ctx, cur)

if response["debug"].get("session_complete"):
    print("Session finished!")
elif response["debug"]["concept"].get("concept_complete"):
    print("Concept completed, moving to next")
else:
    print("Learning in progress...")
```

## Next Steps

1. **Try it out**: Run the basic example above
2. **Inspect state**: Look at the debug field in responses
3. **Add persistence**: Implement save/load in `EnvironmentStateManager`
4. **Customize policies**: Replace simple policies with your logic
5. **Add features**: Extend states, actions, or add new tools

## Need Help?

- Read `environment/README.md` for detailed architecture
- Check `ENVIRONMENT_IMPLEMENTATION.md` for full overview
- Look at existing policy code in `environment/policies.py`
- Review tool implementations in `environment/tools.py`
