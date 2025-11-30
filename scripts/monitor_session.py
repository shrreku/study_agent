#!/usr/bin/env python3
"""
Monitor a tutoring session with student and tutor models.

This script runs a small session and displays detailed interaction logs
showing how the student and tutor models interact.

Usage:
    python scripts/monitor_session.py --concept "convection heat transfer"
    python scripts/monitor_session.py --resource <resource_id> --turns 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv
load_dotenv()

# Setup rich logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich import box

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress most logs
    format="%(message)s",
)


def print_header(text: str):
    """Print a section header"""
    console.print()
    console.print(Panel(text, style="bold cyan", box=box.DOUBLE))


def print_turn(turn_num: int, role: str, message: str, meta: Optional[dict] = None):
    """Print a conversation turn with formatting"""
    if role == "tutor":
        style = "green"
        icon = "🎓"
    else:
        style = "yellow"
        icon = "🎒"
    
    # Truncate long messages for display
    display_msg = message[:500] + "..." if len(message) > 500 else message
    
    console.print(f"\n[bold]{icon} Turn {turn_num} - {role.upper()}[/bold]", style=style)
    console.print(Panel(display_msg, border_style=style, box=box.ROUNDED))
    
    if meta:
        # Show key metadata
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("Value")
        
        for key in ["intent", "correctness", "mastery_after", "engagement", "action"]:
            if key in meta:
                val = meta[key]
                if isinstance(val, float):
                    val = f"{val:.2f}"
                elif hasattr(val, 'value'):
                    val = val.value
                table.add_row(key, str(val))
        
        console.print(table)


def print_state_summary(state: dict):
    """Print a summary of the current state"""
    table = Table(title="Session State", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, val in state.items():
        if isinstance(val, float):
            val = f"{val:.3f}"
        table.add_row(key, str(val))
    
    console.print(table)


def run_monitored_session(
    concept: str,
    learner_type: str = "average",
    max_turns: int = 8,
    use_llm_tutor: bool = True,
    use_llm_student: bool = True,
):
    """Run a monitored tutoring session"""
    
    print_header(f"📚 Monitored Session: {concept}")
    console.print(f"[dim]Learner Type: {learner_type} | Max Turns: {max_turns}[/dim]")
    console.print(f"[dim]LLM Tutor: {use_llm_tutor} | LLM Student: {use_llm_student}[/dim]")
    
    # Import components
    from mdp.student_models import create_student, LLMStudentSimulator
    from mdp.llm_client import LLMClient
    from mdp.policies_v2 import HybridTutorPolicy, RuleBasedTutorPolicy
    from mdp.planning import LLMConceptPolicy
    from mdp.rag import RAGTools
    from mdp.schemas_v2 import (
        TutorState, TutorObservation, StudentProfile, Plan, PlanStep,
        StudentAnalysis, StudentIntent, CorrectnessLevel, RecommendedAction,
        PedagogicalAction,
    )
    import uuid
    
    # Initialize LLM
    llm_client = None
    if use_llm_tutor or use_llm_student:
        try:
            llm_client = LLMClient()
            console.print(f"[green]✓ LLM initialized: {llm_client.model}[/green]")
        except Exception as e:
            console.print(f"[red]✗ LLM init failed: {e}[/red]")
    
    # Initialize components
    rag = RAGTools()
    
    # Create student
    student = create_student(learner_type, llm_client if use_llm_student else None)
    console.print(f"[green]✓ Student created: {student.profile.learner_type.value}[/green]")
    console.print(f"  Learning rate: {student.profile.learning_rate:.2f}")
    console.print(f"  Prior knowledge: {student.profile.prior_knowledge:.2f}")
    
    # Create tutor policy
    if use_llm_tutor and llm_client:
        tutor_policy = HybridTutorPolicy(llm_client)
    else:
        tutor_policy = RuleBasedTutorPolicy()
    console.print(f"[green]✓ Tutor policy: {type(tutor_policy).__name__}[/green]")
    
    # Generate concept plan
    console.print(f"\n[cyan]Generating plan for concept: {concept}[/cyan]")
    
    try:
        concept_policy = LLMConceptPolicy(llm_client) if llm_client else None
        if concept_policy:
            plan_state = {
                "concept_id": concept,
                "student_profile": {"mastery": student.profile.prior_knowledge},
                "constraints": {"max_steps": 4},
            }
            plan = concept_policy.generate_plan(plan_state)
        else:
            # Fallback plan
            plan = Plan(
                plan_id=str(uuid.uuid4())[:8],
                concept=concept,
                steps=[
                    PlanStep(1, concept, "explain", f"Introduction to {concept}"),
                    PlanStep(2, concept, "example", f"Example of {concept}"),
                    PlanStep(3, concept, "question", f"Practice question about {concept}"),
                ],
            )
    except Exception as e:
        console.print(f"[yellow]Plan generation failed: {e}, using fallback[/yellow]")
        plan = Plan(
            plan_id=str(uuid.uuid4())[:8],
            concept=concept,
            steps=[
                PlanStep(1, concept, "explain", f"Introduction to {concept}"),
                PlanStep(2, concept, "question", f"Question about {concept}"),
            ],
        )
    
    console.print(f"[green]✓ Plan generated: {len(plan.steps)} steps[/green]")
    for step in plan.steps:
        console.print(f"  Step {step.step_id}: [{step.pedagogy}] {step.content[:60] if step.content else 'N/A'}...")
    
    # Initialize state
    session_id = str(uuid.uuid4())[:8]
    state = TutorState(
        session_id=session_id,
        student_id=student.state.student_id,
        concept_id=concept,
        plan=plan,
        current_step_index=0,
        mastery_current=student.profile.prior_knowledge,
    )
    
    # Get RAG context
    rag_context, _ = rag.get_planning_context(concept)
    if rag_context:
        console.print(f"[green]✓ RAG context retrieved ({len(rag_context)} chars)[/green]")
    else:
        console.print(f"[yellow]⚠ No RAG context available[/yellow]")
    
    print_header("🎭 Starting Session")
    
    # Conversation loop
    conversation = []
    turn = 0
    
    while turn < max_turns and state.current_step_index < len(plan.steps):
        current_step = plan.steps[state.current_step_index]
        
        # Generate tutor message for current step
        tutor_message = generate_tutor_message(current_step, concept, rag_context, llm_client, turn, conversation)
        
        conversation.append({"role": "tutor", "content": tutor_message})
        print_turn(turn + 1, "tutor", tutor_message, {"action": current_step.pedagogy})
        
        # Get student response
        student_response, student_meta = student.respond(
            tutor_message=tutor_message,
            concept=concept,
            step_info={"step": current_step.step_id, "turn": turn},
        )
        
        conversation.append({"role": "student", "content": student_response})
        print_turn(turn + 1, "student", student_response, student_meta)
        
        # Create observation
        analysis = create_analysis_from_meta(student_meta)
        observation = TutorObservation.from_state(state, analysis, student_response, rag_context)
        
        # Get tutor policy action
        try:
            action = tutor_policy.select_action(observation)
            console.print(f"[dim]Policy selected: {action.action.value}[/dim]")
        except Exception as e:
            console.print(f"[red]Policy error: {e}[/red]")
            action = None
        
        # Update state
        state.mastery_current = student.get_mastery(concept)
        state.turn_count += 1
        
        # Decide advancement - advance on correct answer OR after 2 turns on same step
        turns_on_step = turn - (state.current_step_index * 2)  # Rough estimate
        should_advance = (
            student_meta.get("correctness") == "correct" or
            (action and action.action == PedagogicalAction.ADVANCE_STEP) or
            (current_step.pedagogy in ["explain", "example"] and turns_on_step >= 1)  # Auto-advance explanations
        )
        
        if should_advance:
            state.current_step_index += 1
            console.print(f"[cyan]→ Advanced to step {state.current_step_index + 1}[/cyan]")
        
        turn += 1
        
        # Check mastery threshold
        if state.mastery_current >= 0.8:
            console.print(f"[green]🎉 Mastery threshold reached![/green]")
            break
    
    # Final summary
    print_header("📊 Session Summary")
    
    final_mastery = student.get_mastery(concept)
    mastery_gain = final_mastery - student.profile.prior_knowledge
    
    summary = {
        "Total Turns": turn,
        "Steps Completed": f"{state.current_step_index}/{len(plan.steps)}",
        "Initial Mastery": f"{student.profile.prior_knowledge:.2f}",
        "Final Mastery": f"{final_mastery:.2f}",
        "Mastery Gain": f"{mastery_gain:+.2f}",
        "Student Engagement": f"{student.state.engagement:.2f}",
        "Student Frustration": f"{student.state.frustration:.2f}",
        "Correct Answers": student.state.total_correct,
        "Incorrect Answers": student.state.total_incorrect,
    }
    
    print_state_summary(summary)
    
    # Print conversation log
    print_header("💬 Conversation Log")
    for i, turn in enumerate(conversation):
        role = turn["role"]
        msg = turn["content"][:100] + "..." if len(turn["content"]) > 100 else turn["content"]
        icon = "🎓" if role == "tutor" else "🎒"
        console.print(f"{icon} {msg}")
    
    return {
        "turns": turn,
        "mastery_gain": mastery_gain,
        "conversation": conversation,
    }


def generate_tutor_message(step, concept: str, rag_context: str, llm_client, turn: int, conversation: list = None) -> str:
    """Generate tutor message for a step"""
    
    # Build conversation context
    conv_context = ""
    if conversation:
        for t in conversation[-4:]:
            role = "Tutor" if t["role"] == "tutor" else "Student"
            conv_context += f"{role}: {t['content'][:150]}\n"
    
    if llm_client:
        # Use LLM for response generation
        try:
            pedagogy_instructions = {
                "explain": "Explain the concept clearly and engagingly. Start with the key idea.",
                "example": "Give a concrete, relatable example that illustrates the concept.",
                "question": "Ask the student a question to check their understanding. Make it answerable.",
                "hint": "Provide a helpful hint that guides without giving the answer away.",
                "summary": "Summarize what was learned so far.",
            }
            
            instruction = pedagogy_instructions.get(step.pedagogy, pedagogy_instructions["explain"])
            
            prompt = f"""You are teaching about: {concept}

Step goal: {step.content or f'Help student understand {concept}'}
Pedagogy: {step.pedagogy}

{f'Previous conversation:{chr(10)}{conv_context}' if conv_context else ''}

{f'Knowledge base context:{chr(10)}{rag_context[:600]}' if rag_context else ''}

Instruction: {instruction}

Generate a tutor message (2-4 sentences). Be encouraging and clear."""

            messages = [
                {"role": "system", "content": "You are an expert tutor. Be concise, clear, and pedagogically effective."},
                {"role": "user", "content": prompt}
            ]
            result = llm_client.chat_completion(messages, max_tokens=250, temperature=0.7)
            if result and "choices" in result:
                response = result["choices"][0]["message"]["content"].strip()
                if response:
                    return response
        except Exception as e:
            console.print(f"[dim red]LLM error: {e}[/dim red]")
    
    # Template fallback
    templates = {
        "explain": f"Let me explain {concept}. {step.content or 'This is a fundamental concept where heat energy moves from hotter to cooler regions.'}",
        "example": f"Here's an example of {concept}: {step.content or 'Think about how a hot cup of coffee cools down when left on a table - heat moves from the hot coffee to the cooler surrounding air.'}",
        "question": f"Let me check your understanding: {step.content or f'Can you explain in your own words how {concept} works?'}",
        "hint": f"Here's a hint about {concept}: {step.content or 'Consider what happens to heat energy when two objects at different temperatures are in contact.'}",
        "summary": f"To summarize {concept}: {step.content or 'We learned about how heat transfers between objects at different temperatures.'}",
    }
    
    return templates.get(step.pedagogy, templates["explain"])


def create_analysis_from_meta(meta: dict):
    """Create StudentAnalysis from student metadata"""
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
    parser = argparse.ArgumentParser(description="Monitor a tutoring session")
    parser.add_argument("--concept", type=str, default="convection heat transfer",
                       help="Concept to teach")
    parser.add_argument("--resource", type=str, default=None,
                       help="Resource ID (will pick first concept)")
    parser.add_argument("--learner", type=str, default="average",
                       choices=["fast_learner", "average", "struggling", "anxious", "rusher", "deep_thinker", "passive"],
                       help="Learner type")
    parser.add_argument("--turns", type=int, default=8,
                       help="Maximum turns")
    parser.add_argument("--no-llm-tutor", action="store_true",
                       help="Use rule-based tutor (faster)")
    parser.add_argument("--no-llm-student", action="store_true",
                       help="Use template-based student (faster)")
    
    args = parser.parse_args()
    
    # If resource specified, get first concept from it
    concept = args.concept
    if args.resource:
        from mdp.rag import RAGTools
        rag = RAGTools()
        concepts = rag.get_concepts_for_resources([args.resource])
        if concepts:
            concept = concepts[0].get("name") or concepts[0].get("id")
            console.print(f"Using concept from resource: {concept}")
    
    try:
        result = run_monitored_session(
            concept=concept,
            learner_type=args.learner,
            max_turns=args.turns,
            use_llm_tutor=not args.no_llm_tutor,
            use_llm_student=not args.no_llm_student,
        )
        
        console.print(f"\n[bold green]✓ Session completed successfully![/bold green]")
        console.print(f"  Mastery gain: {result['mastery_gain']:+.3f}")
        console.print(f"  Total turns: {result['turns']}")
        
    except Exception as e:
        console.print(f"[bold red]✗ Session failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
