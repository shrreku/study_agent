#!/usr/bin/env python3
"""
Session Simulation Script for Tutor RL Training Data Generation

This script generates expert trajectories by:
1. Using a SOTA model (GPT-4o/Claude) as the "teacher" tutor
2. Using a weaker model or scripted personas as the "student"
3. Logging full state-action-reward tuples for SFT/GRPO training

Usage:
    python scripts/simulate_session.py --concept "convection" --persona "confused" --turns 10
    python scripts/simulate_session.py --concept "thermodynamics" --batch 100 --output data/sessions.jsonl
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ===== STUDENT PERSONAS =====

STUDENT_PERSONAS = {
    "diligent": {
        "description": "Engaged student who tries hard and usually gets it right",
        "p_correct": 0.7,
        "p_partial": 0.2,
        "p_question": 0.3,
        "p_give_up": 0.02,
        "response_templates": {
            "correct": [
                "I think it's {answer}. Is that right?",
                "So the answer would be {answer}?",
                "Based on what you said, {answer}.",
            ],
            "incorrect": [
                "Hmm, is it {wrong_answer}?",
                "I'm not sure, maybe {wrong_answer}?",
            ],
            "partial": [
                "I think I understand part of it - {partial_answer}",
                "So {partial_answer}, but I'm not sure about the rest",
            ],
            "question": [
                "Can you explain why {concept_aspect}?",
                "I don't understand the part about {concept_aspect}",
                "What does {concept_aspect} mean exactly?",
            ],
            "acknowledge": [
                "Okay, I think I get it now.",
                "That makes sense, thanks!",
                "Got it, let me try to apply that.",
            ],
        }
    },
    "confused": {
        "description": "Student who struggles and needs extra help",
        "p_correct": 0.25,
        "p_partial": 0.35,
        "p_question": 0.5,
        "p_give_up": 0.08,
        "response_templates": {
            "correct": [
                "Wait, so is it {answer}?",
                "I think... {answer}?",
            ],
            "incorrect": [
                "I'm confused, isn't it {wrong_answer}?",
                "But I thought {misconception}, so {wrong_answer}?",
                "{misconception}, right?",
            ],
            "partial": [
                "I only understood {partial_answer}...",
                "The {partial_answer} part makes sense but...",
            ],
            "question": [
                "I'm lost. What do you mean by {concept_aspect}?",
                "Can you explain {concept_aspect} more simply?",
                "I don't get it at all. Why is {concept_aspect}?",
                "This is confusing. {concept_aspect}???",
            ],
            "acknowledge": [
                "Okay... I think?",
                "I'll try, but I'm still not 100% sure.",
            ],
        }
    },
    "rusher": {
        "description": "Impatient student who wants to move fast",
        "p_correct": 0.5,
        "p_partial": 0.2,
        "p_question": 0.1,
        "p_give_up": 0.15,
        "response_templates": {
            "correct": [
                "Yeah {answer}, got it. Next?",
                "{answer}. What's next?",
            ],
            "incorrect": [
                "{wrong_answer}. Can we move on?",
                "Whatever, {wrong_answer}. Next topic?",
            ],
            "partial": [
                "Something like {partial_answer}. Moving on?",
            ],
            "question": [
                "Why does this matter?",
                "Can we skip this part?",
            ],
            "acknowledge": [
                "Got it got it. Next?",
                "Yeah okay. What else?",
                "Sure sure. Continue.",
            ],
        }
    },
    "challenger": {
        "description": "Advanced student who asks deep questions",
        "p_correct": 0.65,
        "p_partial": 0.2,
        "p_question": 0.45,
        "p_give_up": 0.02,
        "response_templates": {
            "correct": [
                "So {answer}, but what about edge case {edge_case}?",
                "{answer}. Does this also apply to {related_concept}?",
            ],
            "incorrect": [
                "Actually, I read that {misconception}. Isn't it {wrong_answer}?",
                "But {edge_case} would suggest {wrong_answer}?",
            ],
            "partial": [
                "{partial_answer}, but I'm wondering about {edge_case}",
            ],
            "question": [
                "What's the relationship between {concept_aspect} and {related_concept}?",
                "How does this connect to {related_concept}?",
                "Why doesn't {edge_case} contradict this?",
                "Can you derive {concept_aspect} from first principles?",
            ],
            "acknowledge": [
                "Interesting, that clarifies the {concept_aspect} part.",
                "Okay, so it's more nuanced than I thought.",
            ],
        }
    },
}


@dataclass
class TurnLog:
    """Single turn log for training"""
    turn_id: int
    role: str  # "student" | "tutor"
    content: str
    
    # State at time of turn
    mastery_before: float = 0.0
    mastery_after: float = 0.0
    mastery_delta: float = 0.0
    
    # For tutor turns
    thinking: Optional[str] = None
    strategy: Optional[str] = None
    
    # Analysis (for student turns)
    intent: Optional[str] = None
    correctness: Optional[str] = None
    
    timestamp: float = field(default_factory=time.time)


@dataclass  
class SessionLog:
    """Full session log for training"""
    session_id: str
    concept: str
    student_persona: str
    
    turns: List[TurnLog] = field(default_factory=list)
    
    # Session-level metrics
    initial_mastery: float = 0.0
    final_mastery: float = 0.0
    total_mastery_gain: float = 0.0
    turn_count: int = 0
    success: bool = False  # Reached target mastery
    
    # Metadata
    teacher_model: str = ""
    student_model: str = ""
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "concept": self.concept,
            "student_persona": self.student_persona,
            "turns": [asdict(t) for t in self.turns],
            "initial_mastery": self.initial_mastery,
            "final_mastery": self.final_mastery,
            "total_mastery_gain": self.total_mastery_gain,
            "turn_count": self.turn_count,
            "success": self.success,
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "created_at": self.created_at,
        }


class StudentSimulator:
    """Simulates student responses based on persona"""
    
    def __init__(self, persona_name: str = "diligent", llm_client=None):
        self.persona = STUDENT_PERSONAS.get(persona_name, STUDENT_PERSONAS["diligent"])
        self.persona_name = persona_name
        self.llm_client = llm_client
        
        # Track student state
        self.understanding = 0.3  # Internal understanding level
        self.patience = 1.0
        self.consecutive_failures = 0
        
    def respond(
        self, 
        tutor_message: str, 
        concept: str,
        turn_number: int,
    ) -> tuple[str, str, str]:
        """
        Generate student response based on persona and tutor message.
        
        Returns: (response_text, intent, correctness)
        """
        # Check for give up
        if random.random() < self.persona["p_give_up"] * (1 - self.patience):
            return "I give up, this is too hard.", "give_up", "null"
        
        # Decide response type based on probabilities
        roll = random.random()
        
        # Update understanding based on tutor message (simple heuristic)
        if "?" in tutor_message:
            # Tutor asked a question - need to answer
            return self._generate_answer(concept, tutor_message)
        
        # Check if student has a question
        if roll < self.persona["p_question"]:
            return self._generate_question(concept)
        
        # Otherwise acknowledge
        return self._generate_acknowledgment()
    
    def _generate_answer(self, concept: str, question: str) -> tuple[str, str, str]:
        """Generate an answer to tutor's question"""
        roll = random.random()
        
        # Determine correctness
        if roll < self.persona["p_correct"]:
            correctness = "correct"
            self.understanding = min(1.0, self.understanding + 0.15)
            self.consecutive_failures = 0
        elif roll < self.persona["p_correct"] + self.persona["p_partial"]:
            correctness = "partial"
            self.understanding = min(1.0, self.understanding + 0.05)
            self.consecutive_failures = 0
        else:
            correctness = "incorrect"
            self.understanding = max(0.0, self.understanding - 0.05)
            self.consecutive_failures += 1
            self.patience = max(0.3, self.patience - 0.1)
        
        # Pick template
        templates = self.persona["response_templates"].get(correctness, ["I'm not sure."])
        template = random.choice(templates)
        
        # Fill placeholders (simplified)
        response = template.format(
            answer=f"related to {concept}",
            wrong_answer=f"something different about {concept}",
            partial_answer=f"part of {concept}",
            misconception=f"a common misconception about {concept}",
            concept_aspect=f"the {concept} mechanism",
            edge_case=f"extreme cases",
            related_concept=f"related topic",
        )
        
        return response, "answer", correctness
    
    def _generate_question(self, concept: str) -> tuple[str, str, str]:
        """Generate a question about the concept"""
        templates = self.persona["response_templates"].get("question", ["Can you explain more?"])
        template = random.choice(templates)
        
        response = template.format(
            concept_aspect=f"the {concept} process",
            related_concept="related phenomena",
            edge_case="boundary conditions",
        )
        
        return response, "question", "null"
    
    def _generate_acknowledgment(self) -> tuple[str, str, str]:
        """Generate acknowledgment"""
        templates = self.persona["response_templates"].get("acknowledge", ["Okay."])
        response = random.choice(templates)
        return response, "acknowledge", "null"
    
    def get_simulated_mastery(self) -> float:
        """Return current simulated mastery level"""
        return self.understanding


class TeacherTutor:
    """Uses SOTA model to generate expert tutor responses with reasoning traces"""
    
    TEACHER_PROMPT = """You are an expert tutor demonstrating pedagogical best practices.

CRITICAL RULES:
1. NEVER give the answer directly - use scaffolding (questions, hints, analogies)
2. ALWAYS output your thinking in <think>...</think> tags FIRST
3. ALWAYS declare your strategy in [Strategy: X] format
4. Keep responses under 120 words
5. Be warm, encouraging, and patient

AVAILABLE STRATEGIES:
- SOCRATIC_QUESTION: Guide via questioning
- HINT: Give partial information
- WORKED_EXAMPLE: Walk through similar problem step-by-step
- ANALOGY: Connect to something familiar
- EXPLAIN: Direct explanation (use sparingly)
- CORRECT_MISCONCEPTION: Address specific error
- CONCEPT_CHECK: Quick verification question
- REFLECT: Metacognitive prompt

Student state:
- Concept: {concept}
- Current mastery: {mastery:.2f}
- Turn: {turn_number}
- Student persona: {persona} 
- Recent student message: "{student_message}"

Conversation so far:
{conversation_history}

Respond in this exact format:
<think>
[Your reasoning about what the student needs and why you're choosing this strategy]
</think>
[Strategy: STRATEGY_NAME]
[Your response to the student - warm, scaffolding, under 120 words]"""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    def respond(
        self,
        concept: str,
        mastery: float,
        turn_number: int,
        student_message: str,
        conversation_history: str,
        persona: str,
    ) -> tuple[str, str, str]:
        """
        Generate expert tutor response with reasoning trace.
        
        Returns: (full_response, thinking, strategy)
        """
        prompt = self.TEACHER_PROMPT.format(
            concept=concept,
            mastery=mastery,
            turn_number=turn_number,
            student_message=student_message,
            conversation_history=conversation_history[-2000:],  # Truncate
            persona=persona,
        )
        
        try:
            # Use raw completion to preserve formatting
            messages = [
                {"role": "system", "content": "You are an expert pedagogical tutor."},
                {"role": "user", "content": prompt}
            ]
            
            result = self.llm_client.chat_completion(
                messages, 
                max_tokens=500, 
                temperature=0.7
            )
            
            if result and "choices" in result:
                full_response = result["choices"][0]["message"]["content"]
                thinking, strategy, _ = self._parse_response(full_response)
                return full_response, thinking, strategy
            
        except Exception as e:
            logger.error(f"Teacher generation failed: {e}")
        
        # Fallback
        return (
            "<think>Fallback response needed</think>\n[Strategy: EXPLAIN]\nLet me explain this concept step by step.",
            "Fallback response needed",
            "EXPLAIN"
        )
    
    def _parse_response(self, response: str) -> tuple[str, str, str]:
        """Parse thinking and strategy from response"""
        import re
        
        thinking = ""
        strategy = "EXPLAIN"
        
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
        
        strat_match = re.search(r"\[Strategy:\s*(\w+)\]", response, re.IGNORECASE)
        if strat_match:
            strategy = strat_match.group(1).upper()
        
        return thinking, strategy, response


def simulate_session(
    concept: str,
    persona: str = "diligent",
    max_turns: int = 15,
    target_mastery: float = 0.8,
    llm_client=None,
) -> SessionLog:
    """
    Simulate a full tutoring session.
    
    Args:
        concept: The concept to teach
        persona: Student persona name
        max_turns: Maximum number of turns
        target_mastery: Mastery level to reach for success
        llm_client: LLM client for teacher model
        
    Returns:
        SessionLog with full trajectory
    """
    session = SessionLog(
        session_id=str(uuid.uuid4()),
        concept=concept,
        student_persona=persona,
        teacher_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        student_model=f"persona:{persona}",
    )
    
    student = StudentSimulator(persona)
    teacher = TeacherTutor(llm_client) if llm_client else None
    
    session.initial_mastery = student.get_simulated_mastery()
    current_mastery = session.initial_mastery
    
    conversation_history = ""
    
    # Initial tutor message
    if teacher:
        full_resp, thinking, strategy = teacher.respond(
            concept=concept,
            mastery=current_mastery,
            turn_number=0,
            student_message="(Session start)",
            conversation_history="",
            persona=persona,
        )
    else:
        full_resp = f"Let's learn about {concept} today! What do you already know about it?"
        thinking = "Starting session with orientation question"
        strategy = "SOCRATIC_QUESTION"
    
    # Extract visible response (after strategy tag)
    visible_resp = full_resp
    import re
    strat_match = re.search(r"\[Strategy:\s*\w+\]", full_resp)
    if strat_match:
        visible_resp = full_resp[strat_match.end():].strip()
    
    session.turns.append(TurnLog(
        turn_id=0,
        role="tutor",
        content=visible_resp,
        thinking=thinking,
        strategy=strategy,
        mastery_before=current_mastery,
        mastery_after=current_mastery,
    ))
    conversation_history += f"Tutor: {visible_resp}\n"
    
    for turn in range(1, max_turns):
        # Student responds
        student_resp, intent, correctness = student.respond(
            tutor_message=visible_resp,
            concept=concept,
            turn_number=turn,
        )
        
        prev_mastery = current_mastery
        current_mastery = student.get_simulated_mastery()
        mastery_delta = current_mastery - prev_mastery
        
        session.turns.append(TurnLog(
            turn_id=turn,
            role="student",
            content=student_resp,
            intent=intent,
            correctness=correctness,
            mastery_before=prev_mastery,
            mastery_after=current_mastery,
            mastery_delta=mastery_delta,
        ))
        conversation_history += f"Student: {student_resp}\n"
        
        # Check termination
        if intent == "give_up":
            logger.info(f"Session ended: student gave up at turn {turn}")
            break
        
        if current_mastery >= target_mastery:
            session.success = True
            logger.info(f"Session success: mastery {current_mastery:.2f} at turn {turn}")
            break
        
        # Tutor responds
        if teacher:
            full_resp, thinking, strategy = teacher.respond(
                concept=concept,
                mastery=current_mastery,
                turn_number=turn,
                student_message=student_resp,
                conversation_history=conversation_history,
                persona=persona,
            )
        else:
            # Fallback without LLM
            if correctness == "correct":
                full_resp = "Great job! Let's move to the next aspect."
                thinking = "Student correct, advancing"
                strategy = "ADVANCE"
            elif correctness == "incorrect":
                full_resp = f"Not quite. Let me give you a hint about {concept}..."
                thinking = "Student incorrect, providing hint"
                strategy = "HINT"
            else:
                full_resp = f"Good question! Think about how {concept} relates to everyday examples."
                thinking = "Student asked question, using analogy"
                strategy = "ANALOGY"
        
        # Extract visible response
        visible_resp = full_resp
        strat_match = re.search(r"\[Strategy:\s*\w+\]", full_resp)
        if strat_match:
            visible_resp = full_resp[strat_match.end():].strip()
        # Also remove thinking tags from visible
        visible_resp = re.sub(r"<think>.*?</think>\s*", "", visible_resp, flags=re.DOTALL)
        
        session.turns.append(TurnLog(
            turn_id=turn + 1,
            role="tutor",
            content=visible_resp,
            thinking=thinking,
            strategy=strategy,
            mastery_before=current_mastery,
            mastery_after=current_mastery,
        ))
        conversation_history += f"Tutor: {visible_resp}\n"
    
    session.final_mastery = current_mastery
    session.total_mastery_gain = session.final_mastery - session.initial_mastery
    session.turn_count = len(session.turns)
    
    return session


def main():
    parser = argparse.ArgumentParser(description="Simulate tutoring sessions for RL training")
    parser.add_argument("--concept", type=str, default="convection", help="Concept to teach")
    parser.add_argument("--persona", type=str, default="diligent", 
                       choices=list(STUDENT_PERSONAS.keys()), help="Student persona")
    parser.add_argument("--turns", type=int, default=15, help="Max turns per session")
    parser.add_argument("--batch", type=int, default=1, help="Number of sessions to generate")
    parser.add_argument("--output", type=str, default="data/sessions.jsonl", help="Output file")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls (use fallback)")
    
    args = parser.parse_args()
    
    # Initialize LLM client
    llm_client = None
    if not args.no_llm:
        try:
            from mdp.llm_client import LLMClient
            llm_client = LLMClient()
            logger.info(f"Using LLM: {llm_client.model}")
        except Exception as e:
            logger.warning(f"Could not initialize LLM client: {e}. Using fallback.")
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate sessions
    sessions = []
    personas = list(STUDENT_PERSONAS.keys()) if args.batch > 1 else [args.persona]
    
    for i in range(args.batch):
        persona = random.choice(personas) if args.batch > 1 else args.persona
        logger.info(f"Generating session {i+1}/{args.batch} with persona={persona}")
        
        session = simulate_session(
            concept=args.concept,
            persona=persona,
            max_turns=args.turns,
            llm_client=llm_client,
        )
        sessions.append(session)
        
        # Log summary
        logger.info(
            f"  -> Turns: {session.turn_count}, "
            f"Mastery: {session.initial_mastery:.2f} → {session.final_mastery:.2f}, "
            f"Success: {session.success}"
        )
    
    # Write to file
    with open(output_path, "w") as f:
        for session in sessions:
            f.write(json.dumps(session.to_dict()) + "\n")
    
    logger.info(f"Wrote {len(sessions)} sessions to {output_path}")
    
    # Summary stats
    successes = sum(1 for s in sessions if s.success)
    avg_gain = sum(s.total_mastery_gain for s in sessions) / len(sessions)
    avg_turns = sum(s.turn_count for s in sessions) / len(sessions)
    
    print(f"\n=== Summary ===")
    print(f"Sessions: {len(sessions)}")
    print(f"Success rate: {successes}/{len(sessions)} ({100*successes/len(sessions):.1f}%)")
    print(f"Avg mastery gain: {avg_gain:.3f}")
    print(f"Avg turns: {avg_turns:.1f}")


if __name__ == "__main__":
    main()
