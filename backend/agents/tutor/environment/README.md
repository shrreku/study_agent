# 3-Layer MDP Environment

This directory contains a clean, efficient implementation of a hierarchical 3-layer MDP architecture for the tutor agent.

## Architecture Overview

The environment is organized into three distinct layers, each responsible for a different level of decision-making:

### Layer 1: Session Environment
**File**: `session_env.py`

**Responsibility**: Manages the overall study session and concept sequencing.

**Key Components**:
- `SessionState`: Tracks session plan, current concept, mastery map
- `SessionAction`: Action space (START_CONCEPT, ADVANCE_CONCEPT, END_SESSION)
- `SessionEnvironment`: Environment for session-level decisions

**Decision Logic**: 
- Decides which concept to study next
- Tracks progress through the session plan
- Determines when the session should end

### Layer 2: Concept Environment
**File**: `concept_env.py`

**Responsibility**: Handles learning for a single concept.

**Key Components**:
- `ConceptState`: Tracks concept plan, mastery, current step
- `ConceptAction`: Action space (GENERATE_PLAN, EXECUTE_STEP, COMPLETE_CONCEPT)
- `ConceptEnvironment`: Environment for concept-level decisions

**Decision Logic**:
- Generates or requests concept learning plans (via LLM)
- Tracks progress through plan steps
- Monitors mastery and decides when concept is complete

### Layer 3: Tutor Environment
**File**: `tutor_env.py`

**Responsibility**: Executes individual pedagogical actions.

**Key Components**:
- `TutorState`: Tracks current step, pedagogical action, user interaction
- `TutorAction`: Action space (EXPLAIN, ASK_QUESTION, WORKED_EXAMPLE, etc.)
- `TutorEnvironment`: Environment for tutor-level decisions

**Decision Logic**:
- Maps plan steps to concrete pedagogical actions
- Generates tutor messages and UI controls
- Handles user interactions (button clicks)

## Core Components

### Base Classes (`base.py`)
- `EnvironmentState`: Base state protocol
- `EnvironmentTransition`: Transition result container
- `BaseEnvironment`: Abstract environment interface

### Context Models (`context.py`)
Lightweight context objects for each layer:
- `SessionContext`: Session-level context
- `ConceptContext`: Concept-level context
- `TutorContext`: Tutor-level context

### Policies (`policies.py`)
Simple rule-based policies for MVP:
- `SimpleSessionPolicy`: Sequential concept progression
- `SimpleConceptPolicy`: Plan-following with replanning
- `SimpleTutorPolicy`: Step-type to action mapping

### Tools (`tools.py`)
Environment management utilities:
- `EnvironmentStateManager`: State persistence
- `PlanCoordinator`: Plan generation coordination
- `ResponseBuilder`: Frontend response formatting

### Orchestrator (`orchestrator.py`)
Main coordination layer:
- `EnvironmentOrchestrator`: Coordinates all 3 layers
- `run_environment_turn()`: Entry point for one turn

## Control Flow

```
User Input → Orchestrator
              ↓
      Layer 1: Session
      - Get current concept
      - Check session completion
              ↓
      Layer 2: Concept
      - Ensure plan exists (call LLM if needed)
      - Get current step
      - Check concept completion
              ↓
      Layer 3: Tutor
      - Execute pedagogical action
      - Generate response
              ↓
      Response → User
```

## Key Features

### 1. Clean Separation
Each layer has clear responsibilities with no overlap or tight coupling.

### 2. Extensibility
- Policies can be easily swapped with learned policies
- New actions can be added to any layer
- State can be extended without breaking existing code

### 3. Tool Integration
Uses existing tools from `mdp/` and `tools/`:
- `ConceptPlannerLLM` for plan generation
- `SessionPlannerLLM` for session planning
- Existing plan dataclasses (`SessionPlan`, `ConceptPlan`)

### 4. Persistence Ready
- State managers handle save/load
- All states are serializable
- Easy to integrate with existing database

### 5. Frontend Compatible
- Generates standard response format
- Supports button-based interaction
- Compatible with existing UI modes

## Usage Example

```python
from backend.agents.tutor.environment import run_environment_turn
from backend.agents.tutor.runtime.context import TurnContext

# Create turn context
ctx = TurnContext(
    session_id="session_123",
    user_id="user_456",
    message="I want to learn about heat transfer",
    target_concepts=["Concept_A", "Concept_B", "Concept_C"],
    payload={"button_clicked": True},
)

# Run one turn through all 3 layers
response = run_environment_turn(ctx, db_cursor)

# Response contains:
# - messages: List of tutor messages
# - ui_mode: "buttons_only"
# - button_options: ["Continue"]
# - debug: State info from all layers
```

## MVP Implementation

For the initial version, the environment:
- ✅ Uses simple sequential session strategy
- ✅ Generates concept plans via LLM
- ✅ Executes plans step-by-step
- ✅ Uses button-based interaction
- ✅ Tracks mastery heuristically
- ❌ Does not analyze user text input (future)
- ❌ Does not use complex quiz evaluation (future)
- ❌ Does not use learned policies (future)

## Extending the Environment

### Adding New Actions

1. **Add to enum** in respective `*_env.py` file
2. **Update policy** in `policies.py` to decide when to use it
3. **Handle in step()** method of the environment

### Adding New State Fields

1. **Extend state dataclass** in respective `*_env.py` file
2. **Update serialization** in `tools.py` if needed
3. **Use in policy/step logic** as required

### Replacing Policies

```python
# Custom policy implementing the protocol
class LearnedConceptPolicy:
    def decide(self, state, **kwargs) -> ConceptAction:
        # Your learned policy logic
        pass

# Use in orchestrator
orchestrator.concept_policy = LearnedConceptPolicy()
```

## Integration Points

### With Existing Code
- Uses `mdp/plans.py` for plan dataclasses
- Uses `mdp/tools_factory.py` for tool creation
- Uses `tools/concept_planner.py` for LLM planning
- Compatible with existing persistence layer

### With Frontend
- Same response format as existing orchestrators
- Same UI control vocabulary
- Same button interaction pattern

## Testing

Each layer can be tested independently:

```python
# Test session environment
session_env = SessionEnvironment(session_id, user_id, plan)
transition = session_env.step(SessionAction.START_CONCEPT)
assert not transition.terminated

# Test concept environment  
concept_env = ConceptEnvironment(session_id, user_id, concept_id)
concept_env.set_plan(test_plan)
transition = concept_env.step(ConceptAction.EXECUTE_STEP)
assert transition.outputs["current_step"] is not None

# Test tutor environment
tutor_env = TutorEnvironment(session_id, user_id, concept_id)
transition = tutor_env.step(TutorAction.EXPLAIN, step=test_step)
assert len(transition.outputs["messages"]) > 0
```

## Next Steps

1. **Wire up orchestrator** to API endpoint
2. **Implement state persistence** in `EnvironmentStateManager`
3. **Add mastery estimation** integration
4. **Test full flow** end-to-end
5. **Add logging** for RL data collection
6. **Iterate on policies** based on usage
