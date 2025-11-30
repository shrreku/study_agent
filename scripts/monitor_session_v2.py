#!/usr/bin/env python3
"""
Monitor Session v2 - Using proper MDP v2 Architecture with Configurable LLMs

This script runs a tutoring session using the full MDP v2 pipeline:
1. TutorPolicy.select_action() for pedagogical decisions
2. TutorPolicy.generate_response() for response generation with action-specific prompts
3. LLM-based student model for realistic responses

Features:
- Configurable LLM models for each component (planner, policy, response, analyzer, student)
- Complete all plan steps (not just reach mastery threshold)
- Uses InputAnalyzer for student message classification

Usage:
    python scripts/monitor_session_v2.py --concept "convection" --learner "struggling"
    
    # With custom models
    python scripts/monitor_session_v2.py --concept "heat transfer" \\
        --planner-model "openai/gpt-4o" \\
        --analyzer-model "google/gemini-2.5-flash-lite" \\
        --student-model "google/gemini-2.0-flash-lite-001"
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


def print_header(text: str):
    console.print()
    console.print(Panel(text, style="bold cyan", box=box.DOUBLE))


def print_turn(turn_num: int, role: str, message: str, meta: Optional[dict] = None):
    style = "green" if role == "tutor" else "yellow"
    icon = "🎓" if role == "tutor" else "🎒"
    display_msg = message[:600] + "..." if len(message) > 600 else message
    
    console.print(f"\n[bold]{icon} Turn {turn_num} - {role.upper()}[/bold]", style=style)
    console.print(Panel(display_msg, border_style=style, box=box.ROUNDED))
    
    if meta:
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("Value")
        for key in ["action", "intent", "correctness", "mastery_after", "engagement", "thinking"]:
            if key in meta:
                val = meta[key]
                if isinstance(val, float):
                    val = f"{val:.2f}"
                elif hasattr(val, 'value'):
                    val = val.value
                if key == "thinking" and len(str(val)) > 80:
                    val = str(val)[:80] + "..."
                table.add_row(key, str(val))
        console.print(table)


def print_step_progress(current: int, total: int, step_content: str):
    """Show current step progress"""
    bar_width = 20
    filled = int((current / total) * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    console.print(f"[dim cyan]Step {current}/{total} [{bar}] {step_content[:40]}...[/dim cyan]")


DEFAULT_MODEL = "google/gemini-2.5-flash"


def run_mdp_session(
    concept: str,
    learner_type: str = "average",
    max_turns_per_step: int = 3,
    planner_model: Optional[str] = None,
    policy_model: Optional[str] = None,
    response_model: Optional[str] = None,
    analyzer_model: Optional[str] = None,
    student_model: Optional[str] = None,
    complete_all_steps: bool = True,
):
    """
    Run session using proper MDP v2 architecture.
    
    Args:
        concept: Concept to teach
        learner_type: Student learner type
        max_turns_per_step: Max turns before forcing step advancement
        planner_model: LLM for plan generation
        policy_model: LLM for action selection
        response_model: LLM for response generation
        analyzer_model: LLM for input analysis
        student_model: LLM for student simulation
        complete_all_steps: If True, complete all plan steps regardless of mastery
    """
    
    print_header(f"📚 MDP v2 Session: {concept}")
    console.print(f"[dim]Learner: {learner_type} | Complete All Steps: {complete_all_steps}[/dim]")
    
    # Import MDP v2 components
    from mdp.llm_config import LLMConfig, LLMClientManager
    from mdp.policies_v2 import LLMTutorPolicy
    from mdp.planning import LLMConceptPolicy
    from mdp.input_analysis import InputAnalyzer
    from mdp.rag import RAGTools
    from mdp.schemas_v2 import (
        TutorState, TutorObservation, Plan, PlanStep,
        StudentAnalysis, StudentIntent, CorrectnessLevel, RecommendedAction,
        PedagogicalAction,
    )
    from mdp.student_models import LLMStudentSimulator, LearningProfile, LearnerType
    import uuid
    
    # Initialize LLM Configuration
    # Default: use google/gemini-2.5-flash for all components unless overridden
    config = LLMConfig(
        planner_model=planner_model or DEFAULT_MODEL,
        policy_model=policy_model or DEFAULT_MODEL,
        response_model=response_model or DEFAULT_MODEL,
        analyzer_model=analyzer_model or DEFAULT_MODEL,
        student_model=student_model or DEFAULT_MODEL,
    )
    
    clients = LLMClientManager(config)
    
    # Display model configuration
    console.print(f"\n[bold]LLM Configuration:[/bold]")
    for name, model in config.to_dict().items():
        console.print(f"  [green]✓[/green] {name}: {model}")
    
    # Initialize RAG
    rag = RAGTools()
    
    # Initialize components with specific LLMs
    planner_llm = clients.get_planner()
    policy_llm = clients.get_policy()
    analyzer_llm = clients.get_analyzer()
    student_llm = clients.get_student()
    
    # Create student
    try:
        lt = LearnerType(learner_type.lower())
    except ValueError:
        lt = LearnerType.AVERAGE
    
    profile = LearningProfile.from_learner_type(lt)
    student = LLMStudentSimulator(profile, student_llm)
    
    console.print(f"\n[green]✓ Student: {lt.value}[/green]")
    console.print(f"  Learning rate: {profile.learning_rate:.2f}, Prior: {profile.prior_knowledge:.2f}")
    
    # Create components
    tutor_policy = LLMTutorPolicy(policy_llm)
    input_analyzer = InputAnalyzer(analyzer_llm)
    concept_policy = LLMConceptPolicy(planner_llm)
    
    console.print(f"[green]✓ Tutor Policy: LLMTutorPolicy[/green]")
    console.print(f"[green]✓ Input Analyzer: Using {config.analyzer_model}[/green]")
    
    # Generate concept plan
    console.print(f"\n[cyan]Generating plan for: {concept}[/cyan]")
    
    try:
        plan_state = {
            "concept_id": concept,
            "student_profile": {"mastery": profile.prior_knowledge},
            "constraints": {"max_steps": 6},
        }
        plan = concept_policy.generate_plan(plan_state)
    except Exception as e:
        console.print(f"[yellow]Plan generation failed: {e}[/yellow]")
        plan = Plan(
            plan_id=str(uuid.uuid4())[:8],
            concept=concept,
            steps=[
                PlanStep(1, concept, "explain", f"Introduction to {concept}"),
                PlanStep(2, concept, "example", f"Example of {concept}"),
                PlanStep(3, concept, "question", f"Check understanding of {concept}"),
                PlanStep(4, concept, "explain", f"Deeper explanation of {concept}"),
                PlanStep(5, concept, "question", f"Practice problem for {concept}"),
                PlanStep(6, concept, "summary", f"Summary of {concept}"),
            ],
        )
    
    console.print(f"[green]✓ Plan: {len(plan.steps)} steps[/green]")
    for step in plan.steps:
        content_preview = (step.content[:50] + "...") if step.content and len(step.content) > 50 else (step.content or "")
        console.print(f"  Step {step.step_id} [{step.pedagogy}]: {content_preview}")
    
    # Get RAG context
    rag_status = rag.get_status()
    console.print(f"\n[bold]RAG Status:[/bold]")
    console.print(f"  Vector Search: {'✓' if rag_status['vector_search'] else '✗'} {rag_status.get('vector_error', '') if not rag_status['vector_search'] else ''}")
    console.print(f"  Graph Search: {'✓' if rag_status['graph_search'] else '✗'}")
    
    rag_context, rag_data = rag.get_planning_context(concept)
    if rag_data.get("fallback"):
        console.print(f"[yellow]⚠ Using fallback context (RAG unavailable)[/yellow]")
    elif rag_context:
        console.print(f"[green]✓ RAG context: {len(rag_context)} chars[/green]")
    else:
        console.print(f"[yellow]⚠ No RAG context[/yellow]")
    
    # Initialize TutorState
    session_id = str(uuid.uuid4())[:8]
    state = TutorState(
        session_id=session_id,
        student_id=student.state.student_id,
        concept_id=concept,
        plan=plan,
        current_step_index=0,
        mastery_current=profile.prior_knowledge,
    )
    
    print_header("🎭 Starting MDP Session")
    
    # Conversation loop - iterate through ALL steps
    conversation = []
    total_turns = 0
    
    while state.current_step_index < len(plan.steps):
        current_step = plan.steps[state.current_step_index]
        step_turns = 0
        
        print_step_progress(state.current_step_index + 1, len(plan.steps), current_step.content or current_step.pedagogy)
        
        # Process current step
        while step_turns < max_turns_per_step:
            total_turns += 1
            step_turns += 1
            
            # === TUTOR TURN ===
            
            # Create observation
            if total_turns == 1:
                # First turn - no student message yet
                analysis = StudentAnalysis(
                    intent=StudentIntent.CONTINUE,
                    correctness=CorrectnessLevel.NOT_APPLICABLE,
                    recommended_action=RecommendedAction.STAY,
                )
                last_student_msg = "[Session Start]"
            else:
                last_student_msg = conversation[-1]["content"] if conversation else ""
                # Use InputAnalyzer to classify student message
                try:
                    analysis_result = input_analyzer.analyze(last_student_msg, current_step)
                    analysis = _create_analysis_from_dict(analysis_result)
                except Exception as e:
                    console.print(f"[dim red]Analyzer error: {e}[/dim red]")
                    analysis = _create_analysis(student_meta)
            
            observation = TutorObservation.from_state(
                state, 
                analysis,
                last_student_msg,
                rag_context,
            )
            
            # Policy selects action
            try:
                action = tutor_policy.select_action(observation)
            except Exception as e:
                console.print(f"[red]Policy error: {e}[/red]")
                from mdp.schemas_v2 import TutorAction
                action = TutorAction(action=PedagogicalAction.EXPLAIN, thinking=f"Error: {e}")
            
            # Handle flow control actions
            if action.action == PedagogicalAction.ADVANCE_STEP:
                console.print(f"[cyan]→ Policy requested: ADVANCE_STEP[/cyan]")
                break  # Exit step loop, move to next step
            
            # Generate response
            try:
                tutor_response = tutor_policy.generate_response(observation, action.action, rag_context)
            except Exception as e:
                console.print(f"[red]Response gen error: {e}[/red]")
                tutor_response = f"Let me explain {concept}. {current_step.content or ''}"
            
            conversation.append({"role": "tutor", "content": tutor_response})
            print_turn(total_turns, "tutor", tutor_response, {
                "action": action.action.value,
                "thinking": action.thinking,
            })
            
            # === STUDENT TURN ===
            
            student_response, student_meta = student.respond(
                tutor_message=tutor_response,
                concept=concept,
                step_info={"step": current_step.step_id, "pedagogy": current_step.pedagogy},
            )
            
            conversation.append({"role": "student", "content": student_response})
            print_turn(total_turns, "student", student_response, student_meta)
            
            # Update state
            state.mastery_current = student.get_mastery(concept)
            state.turn_count += 1
            
            # Check if student answered correctly (for question steps)
            if student_meta.get("correctness") == "correct" and current_step.pedagogy in ["question"]:
                console.print(f"[green]✓ Correct answer![/green]")
                break  # Move to next step
        
        # Advance to next step
        state.current_step_index += 1
        if state.current_step_index < len(plan.steps):
            console.print(f"[cyan]→ Advanced to step {state.current_step_index + 1}/{len(plan.steps)}[/cyan]")
    
    # Final summary
    print_header("📊 Session Summary")
    
    final_mastery = student.get_mastery(concept)
    mastery_gain = final_mastery - profile.prior_knowledge
    
    table = Table(title="Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Turns", str(total_turns))
    table.add_row("Steps Completed", f"{state.current_step_index}/{len(plan.steps)}")
    table.add_row("All Steps Done", "✓ Yes" if state.current_step_index >= len(plan.steps) else "✗ No")
    table.add_row("Initial Mastery", f"{profile.prior_knowledge:.2f}")
    table.add_row("Final Mastery", f"{final_mastery:.2f}")
    table.add_row("Mastery Gain", f"{mastery_gain:+.2f}")
    table.add_row("Correct Answers", str(student.state.total_correct))
    table.add_row("Incorrect Answers", str(student.state.total_incorrect))
    table.add_row("Student Engagement", f"{student.state.engagement:.2f}")
    
    console.print(table)
    
    # Model usage summary
    print_header("🤖 Models Used")
    for name, model in config.to_dict().items():
        console.print(f"  {name}: {model}")
    
    # Conversation log
    print_header("💬 Full Conversation")
    for i, turn_data in enumerate(conversation):
        role = turn_data["role"]
        msg = turn_data["content"][:120] + "..." if len(turn_data["content"]) > 120 else turn_data["content"]
        icon = "🎓" if role == "tutor" else "🎒"
        console.print(f"{icon} {msg}")
    
    return {
        "turns": total_turns,
        "mastery_gain": mastery_gain,
        "conversation": conversation,
        "steps_completed": state.current_step_index,
        "all_steps_done": state.current_step_index >= len(plan.steps),
    }


def _create_analysis_from_dict(result: Dict[str, Any]) -> "StudentAnalysis":
    """Create StudentAnalysis from InputAnalyzer result dict."""
    from mdp.schemas_v2 import StudentAnalysis, StudentIntent, CorrectnessLevel, RecommendedAction
    
    intent_map = {
        "answer": StudentIntent.ANSWER,
        "question": StudentIntent.QUESTION,
        "confusion": StudentIntent.CONFUSION,
        "acknowledge": StudentIntent.ACKNOWLEDGE,
        "give_up": StudentIntent.GIVE_UP,
        "continue": StudentIntent.CONTINUE,
    }
    
    correctness_map = {
        "correct": CorrectnessLevel.CORRECT,
        "partial": CorrectnessLevel.PARTIAL,
        "incorrect": CorrectnessLevel.INCORRECT,
    }
    
    rec_map = {
        "advance": RecommendedAction.ADVANCE,
        "reply": RecommendedAction.REPLY,
        "replan": RecommendedAction.REPLAN,
        "stay": RecommendedAction.STAY,
    }
    
    intent_str = str(result.get("intent", "acknowledge")).lower()
    correctness_str = str(result.get("correctness", "")).lower()
    rec_str = str(result.get("recommended_action", "stay")).lower()
    
    return StudentAnalysis(
        intent=intent_map.get(intent_str, StudentIntent.ACKNOWLEDGE),
        correctness=correctness_map.get(correctness_str, CorrectnessLevel.NOT_APPLICABLE),
        correctness_score=result.get("correctness_score", 0.0),
        recommended_action=rec_map.get(rec_str, RecommendedAction.STAY),
        feedback_hint=result.get("feedback", ""),
    )


def _create_analysis(meta: Dict[str, Any]) -> "StudentAnalysis":
    """Create StudentAnalysis from student response metadata."""
    from mdp.schemas_v2 import StudentAnalysis, StudentIntent, CorrectnessLevel, RecommendedAction
    
    intent_map = {
        "answer": StudentIntent.ANSWER,
        "question": StudentIntent.QUESTION,
        "confusion": StudentIntent.CONFUSION,
        "acknowledge": StudentIntent.ACKNOWLEDGE,
        "give_up": StudentIntent.GIVE_UP,
    }
    
    correctness_map = {
        "correct": CorrectnessLevel.CORRECT,
        "partial": CorrectnessLevel.PARTIAL,
        "incorrect": CorrectnessLevel.INCORRECT,
        "na": CorrectnessLevel.NOT_APPLICABLE,
    }
    
    intent_str = meta.get("intent", "acknowledge")
    correctness_str = meta.get("correctness", "na")
    
    if correctness_str == "correct":
        rec_action = RecommendedAction.ADVANCE
    elif intent_str in ["question", "confusion"]:
        rec_action = RecommendedAction.REPLY
    else:
        rec_action = RecommendedAction.STAY
    
    return StudentAnalysis(
        intent=intent_map.get(intent_str, StudentIntent.ACKNOWLEDGE),
        correctness=correctness_map.get(correctness_str, CorrectnessLevel.NOT_APPLICABLE),
        correctness_score=meta.get("correctness_score", 0.0),
        recommended_action=rec_action,
        feedback_hint=meta.get("internal_thought", ""),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Monitor MDP v2 Session with Configurable LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/monitor_session_v2.py --concept "convection"
  
  # With custom models
  python scripts/monitor_session_v2.py --concept "heat transfer" \\
      --planner-model "openai/gpt-4o" \\
      --analyzer-model "google/gemini-2.5-flash-lite" \\
      --student-model "google/gemini-2.0-flash-lite-001"
  
  # Different learner types
  python scripts/monitor_session_v2.py --concept "conduction" --learner "struggling"
        """
    )
    
    # Basic options
    parser.add_argument("--concept", type=str, default="convection heat transfer",
                       help="Concept to teach")
    parser.add_argument("--learner", type=str, default="average",
                       choices=["fast_learner", "average", "struggling", "anxious", "rusher", "deep_thinker", "passive"],
                       help="Learner type")
    parser.add_argument("--max-turns-per-step", type=int, default=3,
                       help="Max turns before advancing to next step")
    
    # Model configuration
    parser.add_argument("--planner-model", type=str, default=None,
                       help="LLM model for plan generation (e.g., 'openai/gpt-4o')")
    parser.add_argument("--policy-model", type=str, default=None,
                       help="LLM model for action selection")
    parser.add_argument("--response-model", type=str, default=None,
                       help="LLM model for response generation")
    parser.add_argument("--analyzer-model", type=str, default=None,
                       help="LLM model for input analysis (e.g., 'google/gemini-2.5-flash-lite')")
    parser.add_argument("--student-model", type=str, default=None,
                       help="LLM model for student simulation (default: google/gemini-2.5-flash)")
    
    args = parser.parse_args()
    
    try:
        result = run_mdp_session(
            concept=args.concept,
            learner_type=args.learner,
            max_turns_per_step=args.max_turns_per_step,
            planner_model=args.planner_model,
            policy_model=args.policy_model,
            response_model=args.response_model,
            analyzer_model=args.analyzer_model,
            student_model=args.student_model,
        )
        
        console.print(f"\n[bold green]✓ Session completed![/bold green]")
        console.print(f"  All steps done: {'Yes ✓' if result['all_steps_done'] else 'No ✗'}")
        console.print(f"  Mastery gain: {result['mastery_gain']:+.3f}")
        console.print(f"  Steps: {result['steps_completed']}")
        
    except Exception as e:
        console.print(f"[bold red]✗ Session failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
