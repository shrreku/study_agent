Tutor Agent MDP Redesign Plan
Goal Description
Redesign the Tutor Agent to use a 3-layered MDP (State, Action, Tools) architecture. The goal is to create a modular, extensible, and "MDP-friendly" system that handles session planning, concept planning, and pedagogical actions.

User Review Required
 Architecture Design: Review the proposed State, Action, and Tool definitions.
 Orchestration Flow: Confirm the interaction between the layers (Session -> Concept -> Pedagogy).
Proposed Changes
Architecture Overview
The system will be divided into three main layers:

Session Layer: Manages the overall session, including the sequence of concepts to be taught.
Concept Layer: Manages the teaching of a specific concept, including the pedagogical strategy (plan).
Pedagogy/Interaction Layer: Handles the immediate interaction with the user (Tutor Action -> User Feedback).
Components
State Space
Session State: Current concept index, overall progress, user profile summary.
Concept State: Current concept details, pedagogical plan status, user understanding of current concept.
Interaction State: Recent conversation history, last user feedback, current pedagogical step.
Action Space
Session Actions: PlanSession, NextConcept, EndSession.
Concept Actions: PlanConcept, ExecuteStep, AssessUnderstanding.
Tutor Actions: Explain, AskQuestion, ProvideExample, Hint, Feedback.
Orchestrator (
backend/agents/tutor/orchestrator.py
)
Class: 
TutorOrchestrator
Responsibilities:
Manage the lifecycle of the 3 MDPs.
Maintain the 
TutorContext
.
Execute the main loop:
Update State: Build/Update Session, Concept, and Pedagogical states from context.
Session Layer: Check 
SessionPolicy
. If action is REPLAN, call 
SessionPlannerTool
. If TERMINATE, end.
Concept Layer: Check 
ConceptPolicy
. If action is REPLAN, call 
ConceptPlannerTool
. If ADVANCE, move to next concept.
Pedagogy Layer: Check 
PedagogicalTutorPolicy
. Decide 
PedagogicalTutorAction
.
Execution: Call 
PedagogicalResponseGeneratorTool
 to generate response.
Return: Return response to user.
Integration Plan
Persistence Layer: Create backend/agents/tutor/persistence.py containing TutorStateManager.
Responsibility: Load/Save 
TutorContext
 and 
TutorSessionPolicy
 from/to 
tutor_session
 table and user_concept_mastery.
Logic: Adapt SQL queries from 
backend/agents/tutor/environment/tools.py
.
Factory Update: Update 
backend/agents/tutor/factory.py
 to:
Accept db_cursor or db_connection.
Initialize TutorStateManager.
Initialize real 
SessionPlannerLLM
, 
ConceptPlannerLLM
, 
DefaultPedagogicalResponseGeneratorTool
.
Entry Point Update: Modify 
backend/agents/tutor/environment/orchestrator.py
:
Rewrite 
run_environment_turn
 to:
Initialize TutorStateManager.
Load 
TutorContext
 from DB.
Call TutorOrchestrator.tick().
Save updated state to DB.
Convert output to frontend response format.
Tools (Existing & New)
SessionPlannerTool: Re-use 
SessionPlannerLLM
.
ConceptPlannerTool: Re-use 
ConceptPlannerLLM
.
PedagogicalResponseGeneratorTool: Re-use 
DefaultPedagogicalResponseGeneratorTool
.
State Builders: Re-use builders in 
mdp/session.py
, 
mdp/concept.py
, 
mdp/pedagogical_tutor.py
.
Directory Structure
backend/agents/tutor/

mdp/: Existing MDP definitions (Session, Concept, Pedagogy).
tools/: Existing tool implementations.
orchestrator.py
: [NEW] Main orchestration logic.
factory.py
: [NEW] Factory to assemble the orchestrator with policies and tools.
Verification Plan
Unit Tests: Test individual state transitions and policy decisions.
Integration Tests: Simulate a full session flow (Session Start -> Plan -> Concept -> Action -> Feedback).