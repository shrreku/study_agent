# 3-Layer MDP Architecture - Visual Guide

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                                   │
│                    (Buttons, Messages, UI)                               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ↓
                    ┌────────────────────────┐
                    │   ORCHESTRATOR         │
                    │  (coordinator)         │
                    └────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ↓                       ↓                       ↓
┌────────────────┐      ┌────────────────┐     ┌────────────────┐
│  LAYER 1       │      │  LAYER 2       │     │  LAYER 3       │
│  SESSION       │  →   │  CONCEPT       │  →  │  TUTOR         │
│  ENVIRONMENT   │      │  ENVIRONMENT   │     │  ENVIRONMENT   │
└────────────────┘      └────────────────┘     └────────────────┘
         │                       │                       │
         ↓                       ↓                       ↓
    [Session State]         [Concept State]        [Tutor State]
    - Concept plan          - Learning plan        - Current step
    - Progress              - Mastery              - Action history
    - Mastery map           - Step index           - UI state
         │                       │                       │
         ↓                       ↓                       ↓
    [Session Policy]        [Concept Policy]       [Tutor Policy]
    - Sequential            - Plan executor        - Step mapper
    - Advance logic         - Mastery checker      - Pedagogical
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                                 ↓
                        ┌────────────────┐
                        │  TOOLS LAYER   │
                        ├────────────────┤
                        │ - Planners     │
                        │ - Estimators   │
                        │ - Builders     │
                        └────────────────┘
```

## Layer Responsibilities

### 🎯 Layer 1: Session Environment

**Manages**: Overall study session

```
┌─────────────────────────────────────┐
│      SESSION ENVIRONMENT            │
├─────────────────────────────────────┤
│ State:                              │
│  • session_plan: [C1, C2, C3]      │
│  • current_index: 1                 │
│  • mastery_map: {C1: 0.7, C2: 0.3} │
├─────────────────────────────────────┤
│ Actions:                            │
│  • START_CONCEPT                    │
│  • ADVANCE_CONCEPT                  │
│  • END_SESSION                      │
├─────────────────────────────────────┤
│ Decides:                            │
│  → Which concept to study?          │
│  → Move to next concept?            │
│  → End session?                     │
└─────────────────────────────────────┘
```

**Example Decision Flow**:
```
If concept_complete:
    → ADVANCE_CONCEPT
If all_concepts_done:
    → END_SESSION
Else:
    → START_CONCEPT
```

### 🎯 Layer 2: Concept Environment

**Manages**: Learning for one concept

```
┌─────────────────────────────────────┐
│      CONCEPT ENVIRONMENT            │
├─────────────────────────────────────┤
│ State:                              │
│  • concept_id: "Heat_Transfer"     │
│  • concept_plan: [S1, S2, S3, S4]  │
│  • current_step_index: 2           │
│  • current_mastery: 0.5            │
│  • target_mastery: 0.8             │
├─────────────────────────────────────┤
│ Actions:                            │
│  • GENERATE_PLAN (LLM call)        │
│  • EXECUTE_STEP                    │
│  • ADVANCE_STEP                    │
│  • COMPLETE_CONCEPT                │
├─────────────────────────────────────┤
│ Decides:                            │
│  → Need a plan?                     │
│  → Execute current step?            │
│  → Mastery sufficient?              │
│  → Need replanning?                 │
└─────────────────────────────────────┘
```

**Example Decision Flow**:
```
If no_plan:
    → GENERATE_PLAN (calls LLM)
If step_complete:
    → ADVANCE_STEP
If plan_exhausted AND mastery_ok:
    → COMPLETE_CONCEPT
Else:
    → EXECUTE_STEP
```

### 🎯 Layer 3: Tutor Environment

**Manages**: Pedagogical actions

```
┌─────────────────────────────────────┐
│      TUTOR ENVIRONMENT              │
├─────────────────────────────────────┤
│ State:                              │
│  • current_step: Step{              │
│      type: "explain",               │
│      instruction: "Explain..."      │
│    }                                │
│  • last_action: EXPLAIN             │
│  • awaiting_input: true             │
├─────────────────────────────────────┤
│ Actions:                            │
│  • EXPLAIN                          │
│  • ASK_QUESTION                     │
│  • WORKED_EXAMPLE                   │
│  • GUIDED_PRACTICE                  │
│  • SUMMARY                          │
├─────────────────────────────────────┤
│ Decides:                            │
│  → What pedagogical action?         │
│  → What message to show?            │
│  → What button to display?          │
└─────────────────────────────────────┘
```

**Example Decision Flow**:
```
Based on step.step_type:
    "explain" → EXPLAIN
    "example" → WORKED_EXAMPLE
    "practice" → GUIDED_PRACTICE
    "summary" → SUMMARY
```

## Data Flow Through Layers

### Turn Execution Sequence

```
USER CLICKS BUTTON
        │
        ↓
┌───────────────────┐
│  1. Parse Input   │
└─────────┬─────────┘
          │
          ↓
┌───────────────────────────────────────┐
│  2. SESSION LAYER                     │
│  ┌─────────────────────────────────┐ │
│  │ State: current_index = 1        │ │
│  │ Action: CONTINUE_CONCEPT        │ │
│  │ Result: focus_concept = "C2"    │ │
│  └─────────────────────────────────┘ │
└───────────────┬───────────────────────┘
                │ Pass: concept_id="C2"
                ↓
┌───────────────────────────────────────┐
│  3. CONCEPT LAYER                     │
│  ┌─────────────────────────────────┐ │
│  │ Ensure plan exists (LLM?)       │ │
│  │ State: step_index = 3           │ │
│  │ Action: EXECUTE_STEP            │ │
│  │ Result: current_step = S3       │ │
│  └─────────────────────────────────┘ │
└───────────────┬───────────────────────┘
                │ Pass: step = S3
                ↓
┌───────────────────────────────────────┐
│  4. TUTOR LAYER                       │
│  ┌─────────────────────────────────┐ │
│  │ Map step type: "example"        │ │
│  │ Action: WORKED_EXAMPLE          │ │
│  │ Generate message: "Here's..."   │ │
│  │ Create button: "Continue"       │ │
│  └─────────────────────────────────┘ │
└───────────────┬───────────────────────┘
                │
                ↓
┌───────────────────────────────────────┐
│  5. BUILD RESPONSE                    │
│  {                                    │
│    messages: [...],                   │
│    button_options: ["Continue"],      │
│    debug: {session, concept, tutor}   │
│  }                                    │
└───────────────┬───────────────────────┘
                │
                ↓
        RETURN TO USER
```

## State Transitions

### Session State Machine

```
     START
       │
       ↓
   [INIT] ─── target_concepts → [CREATE_PLAN]
       │                              │
       │                              ↓
       └──────────────────────→  [READY]
                                     │
                        ┌────────────┴────────────┐
                        │                         │
                        ↓                         ↓
                  [STUDYING_C1]            [STUDYING_C2]
                        │                         │
                  concept_complete          concept_complete
                        │                         │
                        └────────────┬────────────┘
                                     ↓
                              [ALL_COMPLETE]
                                     │
                                     ↓
                                  [END]
```

### Concept State Machine

```
       START
         │
         ↓
     [NO_PLAN] ──── generate ───→ [HAS_PLAN]
         ↑                             │
         │                             ↓
         │                        [EXECUTING]
         │                             │
         │                    ┌────────┼────────┐
         │                    │        │        │
         │              step_done   mastery  more_steps
         │                    │      ok?       needed
         │                    ↓        │        │
         └───── replan ── [COMPLETE] ←┘        │
                              │                  │
                              ↓                  │
                            [END] ←──────────────┘
```

### Tutor State Machine

```
       START
         │
         ↓
   [IDLE] ──── get_step ───→ [READY]
                                 │
                                 ↓
                           [MAP_ACTION]
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ↓                       ↓                       ↓
    [EXPLAIN]             [EXAMPLE]              [PRACTICE]
         │                       │                       │
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                                 ↓
                          [AWAIT_INPUT]
                                 │
                          user_clicks
                                 │
                                 ↓
                          [STEP_COMPLETE]
```

## Component Interactions

### Orchestrator Coordination

```
┌──────────────────────────────────────────────────────┐
│                   ORCHESTRATOR                       │
│                                                      │
│  run_turn(session_id, user_id, message, ...):       │
│                                                      │
│    1. Initialize Session Environment                │
│       ├─→ Load state from DB                        │
│       └─→ Ensure session plan exists                │
│                                                      │
│    2. Session decides focus concept                 │
│       ├─→ Policy: decide(session_state)            │
│       └─→ Action: START/ADVANCE/END                 │
│                                                      │
│    3. Initialize Concept Environment                │
│       ├─→ Load concept state                        │
│       └─→ Ensure concept plan exists (LLM)          │
│                                                      │
│    4. Concept decides current step                  │
│       ├─→ Policy: decide(concept_state)            │
│       └─→ Action: EXECUTE/ADVANCE/COMPLETE          │
│                                                      │
│    5. Initialize Tutor Environment                  │
│       └─→ Set current step                          │
│                                                      │
│    6. Tutor generates response                      │
│       ├─→ Policy: decide(tutor_state, step)        │
│       ├─→ Action: EXPLAIN/EXAMPLE/etc              │
│       └─→ Generate message + button                 │
│                                                      │
│    7. Build response                                │
│       ├─→ Format messages                           │
│       ├─→ Add debug info                            │
│       └─→ Return to frontend                        │
│                                                      │
│    8. Persist states                                │
│       ├─→ Save session state                        │
│       └─→ Save concept state                        │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### Tool Integration

```
┌──────────────────────────────────────────┐
│            PLANNING TOOLS                 │
├──────────────────────────────────────────┤
│                                          │
│  SessionPlannerLLM                       │
│    ↓                                     │
│  generate_session_plan()                 │
│    • Input: target_concepts              │
│    • Output: SessionPlan                 │
│                                          │
│  ConceptPlannerLLM                       │
│    ↓                                     │
│  generate_concept_plan()                 │
│    • Input: concept_id, mastery          │
│    • Output: ConceptPlan (LLM call)      │
│                                          │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│          ESTIMATION TOOLS                 │
├──────────────────────────────────────────┤
│                                          │
│  SimpleMasteryEstimator                  │
│    ↓                                     │
│  estimate_from_step_completion()         │
│    • Input: step_type, current_mastery   │
│    • Output: mastery_delta               │
│                                          │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│          INTERACTION TOOLS                │
├──────────────────────────────────────────┤
│                                          │
│  ButtonController                        │
│    ↓                                     │
│  parse_from_payload()                    │
│    • Input: payload dict                 │
│    • Output: ButtonClick                 │
│                                          │
└──────────────────────────────────────────┘
```

## Key Interfaces

### Environment Protocol

```python
class BaseEnvironment[S]:
    state: S
    
    def reset() -> S:
        """Reset to initial state"""
        
    def step(action, **kwargs) -> EnvironmentTransition:
        """Execute one step"""
        
    def get_state() -> S:
        """Get current state"""
        
    def is_terminated() -> bool:
        """Check if episode done"""
```

### Policy Protocol

```python
class Policy:
    def decide(state, **kwargs) -> Action:
        """Decide next action based on state"""
```

### Tool Protocol

```python
class PlannerTool:
    def __call__(**context) -> Plan:
        """Generate a plan"""
```

## Extension Points

### 1. Adding New Actions

```python
# In session_env.py
class SessionAction(str, Enum):
    START_CONCEPT = "START_CONCEPT"
    ADVANCE_CONCEPT = "ADVANCE_CONCEPT"
    END_SESSION = "END_SESSION"
    SKIP_TO_CONCEPT = "SKIP_TO_CONCEPT"  # ← NEW
```

### 2. Adding New State Fields

```python
# In concept_env.py
@dataclass
class ConceptState(EnvironmentState):
    concept_id: str
    concept_plan: Optional[ConceptPlan]
    current_step_index: int
    current_mastery: float
    difficulty_level: str = "medium"  # ← NEW
```

### 3. Replacing Policies

```python
# Custom learned policy
class LearnedConceptPolicy:
    def __init__(self, model):
        self.model = model
    
    def decide(self, state, **kwargs):
        features = self.extract_features(state)
        action_probs = self.model.predict(features)
        return self.sample_action(action_probs)

# Use in orchestrator
orchestrator.concept_policy = LearnedConceptPolicy(trained_model)
```

## Summary

The 3-layer architecture provides:

✅ **Clear separation**: Each layer has one responsibility
✅ **Extensible design**: Easy to add actions, states, policies
✅ **Tool integration**: Uses existing LLM planners and utilities
✅ **Type safety**: Protocols and dataclasses throughout
✅ **Testable**: Each component can be tested independently
✅ **Production ready**: Logging, persistence, error handling

Each layer communicates through well-defined interfaces, making it easy to understand, test, and extend the system.
