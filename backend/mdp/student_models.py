"""
LLM-Based Student Models for RL Training Data Generation

This module provides sophisticated student simulators that:
1. Use LLM to generate realistic student responses
2. Model different learning profiles (fast learner, struggling, etc.)
3. Track internal knowledge state and mastery progression
4. Generate diverse training data for tutor RL

Student profiles are designed based on educational psychology research:
- Zone of Proximal Development (ZPD)
- Knowledge components and prerequisite relationships
- Affective states (confusion, frustration, engagement)
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.llm_client import LLMClient

logger = logging.getLogger(__name__)


# =============================================================================
# STUDENT PROFILE DEFINITIONS
# =============================================================================

class LearnerType(Enum):
    """Learner archetype based on educational research"""
    FAST_LEARNER = "fast_learner"        # High aptitude, quick to understand
    AVERAGE = "average"                   # Typical student
    STRUGGLING = "struggling"             # Needs more scaffolding
    ANXIOUS = "anxious"                   # Capable but lacks confidence
    RUSHER = "rusher"                     # Impatient, wants to move fast
    DEEP_THINKER = "deep_thinker"        # Asks probing questions
    PASSIVE = "passive"                   # Minimal engagement, needs prompting


@dataclass
class LearningProfile:
    """
    Configurable learning profile for student simulation.
    
    Based on cognitive and affective modeling research.
    """
    learner_type: LearnerType = LearnerType.AVERAGE
    
    # Cognitive parameters (0.0 to 1.0)
    learning_rate: float = 0.15           # How fast mastery increases per correct
    prior_knowledge: float = 0.2          # Starting knowledge level
    retention_rate: float = 0.95          # Knowledge retention between turns
    transfer_ability: float = 0.3         # Ability to apply to related concepts
    
    # Response probabilities
    p_correct_at_mastery: float = 0.85    # P(correct) when mastery = 1.0
    p_correct_at_zero: float = 0.1        # P(correct) when mastery = 0.0
    p_ask_question: float = 0.2           # P(asking a clarifying question)
    p_express_confusion: float = 0.15     # P(expressing confusion when stuck)
    p_give_up: float = 0.02               # P(giving up when frustrated)
    
    # Affective parameters
    initial_engagement: float = 0.7       # Starting engagement level
    frustration_threshold: int = 3        # Consecutive errors before frustration
    patience: float = 0.8                 # How long student persists
    
    # Response style
    verbosity: str = "medium"             # brief, medium, detailed
    formality: str = "casual"             # formal, casual, very_casual
    
    @classmethod
    def from_learner_type(cls, learner_type: LearnerType) -> "LearningProfile":
        """Create profile from learner type archetype"""
        profiles = {
            LearnerType.FAST_LEARNER: cls(
                learner_type=LearnerType.FAST_LEARNER,
                learning_rate=0.25,
                prior_knowledge=0.35,
                p_correct_at_mastery=0.92,
                p_correct_at_zero=0.2,
                p_ask_question=0.15,
                p_express_confusion=0.05,
                patience=0.95,
            ),
            LearnerType.AVERAGE: cls(
                learner_type=LearnerType.AVERAGE,
                learning_rate=0.15,
                prior_knowledge=0.2,
                p_correct_at_mastery=0.85,
                p_correct_at_zero=0.1,
                p_ask_question=0.2,
                p_express_confusion=0.15,
            ),
            LearnerType.STRUGGLING: cls(
                learner_type=LearnerType.STRUGGLING,
                learning_rate=0.08,
                prior_knowledge=0.1,
                p_correct_at_mastery=0.75,
                p_correct_at_zero=0.05,
                p_ask_question=0.35,
                p_express_confusion=0.3,
                p_give_up=0.06,
                patience=0.6,
                frustration_threshold=2,
            ),
            LearnerType.ANXIOUS: cls(
                learner_type=LearnerType.ANXIOUS,
                learning_rate=0.12,
                prior_knowledge=0.25,
                p_correct_at_mastery=0.7,  # Underperforms due to anxiety
                p_correct_at_zero=0.08,
                p_ask_question=0.4,
                p_express_confusion=0.35,
                initial_engagement=0.5,
                verbosity="brief",
            ),
            LearnerType.RUSHER: cls(
                learner_type=LearnerType.RUSHER,
                learning_rate=0.1,
                prior_knowledge=0.15,
                p_correct_at_mastery=0.75,
                p_correct_at_zero=0.15,
                p_ask_question=0.05,
                p_express_confusion=0.05,
                p_give_up=0.08,
                patience=0.4,
                verbosity="brief",
            ),
            LearnerType.DEEP_THINKER: cls(
                learner_type=LearnerType.DEEP_THINKER,
                learning_rate=0.2,
                prior_knowledge=0.3,
                p_correct_at_mastery=0.9,
                p_correct_at_zero=0.15,
                p_ask_question=0.45,
                p_express_confusion=0.1,
                patience=0.95,
                verbosity="detailed",
            ),
            LearnerType.PASSIVE: cls(
                learner_type=LearnerType.PASSIVE,
                learning_rate=0.1,
                prior_knowledge=0.15,
                p_correct_at_mastery=0.7,
                p_correct_at_zero=0.1,
                p_ask_question=0.05,
                p_express_confusion=0.05,
                initial_engagement=0.4,
                verbosity="brief",
            ),
        }
        return profiles.get(learner_type, profiles[LearnerType.AVERAGE])


@dataclass
class StudentState:
    """Internal state of the simulated student"""
    student_id: str = ""
    
    # Knowledge state per concept
    concept_mastery: Dict[str, float] = field(default_factory=dict)
    
    # Current session state
    current_concept: str = ""
    current_step: int = 0
    consecutive_correct: int = 0
    consecutive_incorrect: int = 0
    hints_received: int = 0
    
    # Affective state
    engagement: float = 0.7
    frustration: float = 0.0
    confidence: float = 0.5
    
    # History
    conversation_turns: int = 0
    total_correct: int = 0
    total_incorrect: int = 0
    
    def get_mastery(self, concept: str) -> float:
        """Get mastery for a concept"""
        return self.concept_mastery.get(concept, 0.0)
    
    def update_mastery(self, concept: str, delta: float):
        """Update mastery for a concept"""
        current = self.concept_mastery.get(concept, 0.0)
        self.concept_mastery[concept] = max(0.0, min(1.0, current + delta))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "concept_mastery": self.concept_mastery,
            "current_concept": self.current_concept,
            "current_step": self.current_step,
            "engagement": self.engagement,
            "frustration": self.frustration,
            "confidence": self.confidence,
            "consecutive_correct": self.consecutive_correct,
            "consecutive_incorrect": self.consecutive_incorrect,
        }


# =============================================================================
# LLM-BASED STUDENT SIMULATOR
# =============================================================================

class LLMStudentSimulator:
    """
    Sophisticated student simulator using LLM for realistic responses.
    
    Features:
    - LLM-generated responses based on learning profile
    - Internal mastery tracking with probabilistic correctness
    - Affective state modeling (frustration, engagement)
    - Memory of conversation context
    """
    
    STUDENT_SYSTEM_PROMPT = """You are simulating a {learner_type} student learning about {concept}.

Your learning profile:
- Learning style: {learning_style}
- Current understanding: {mastery_level} ({mastery_pct}% mastery)
- Engagement: {engagement}
- Frustration: {frustration}

Your task: Generate a realistic student response to the tutor's message.

RULES:
1. Stay in character as this type of student
2. Your response should reflect your understanding level
3. {correctness_instruction}
4. Be natural - use appropriate language for a student
5. Keep response under 80 words
6. Do NOT break character or mention you are an AI

Response format (JSON):
{{
    "response": "your student message",
    "internal_thought": "what the student is thinking (hidden from tutor)"
}}"""

    CORRECTNESS_INSTRUCTIONS = {
        "correct": "You DO understand this - give a CORRECT answer showing understanding",
        "partial": "You partially understand - give a PARTIALLY correct answer with some gaps",
        "incorrect": "You DON'T quite understand - give an INCORRECT answer showing misconception",
        "question": "You're confused - ASK a clarifying question instead of answering",
        "confusion": "You're lost - EXPRESS confusion and ask for help",
        "acknowledge": "Just acknowledge what the tutor said briefly",
        "give_up": "Express frustration and reluctance to continue",
    }
    
    def __init__(
        self,
        profile: LearningProfile,
        llm_client: Optional["LLMClient"] = None,
        student_id: Optional[str] = None,
    ):
        self.profile = profile
        self.llm = llm_client
        self.state = StudentState(
            student_id=student_id or f"sim_{uuid.uuid4().hex[:8]}",
            engagement=profile.initial_engagement,
            confidence=0.5,
        )
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
    
    def respond(
        self,
        tutor_message: str,
        concept: str,
        step_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate student response to tutor message.
        
        Returns:
            Tuple of (response_text, metadata_dict)
            metadata contains: intent, correctness, mastery_delta, internal_state
        """
        self.state.current_concept = concept
        self.state.conversation_turns += 1
        
        # Initialize concept mastery if new
        if concept not in self.state.concept_mastery:
            self.state.concept_mastery[concept] = self.profile.prior_knowledge
        
        current_mastery = self.state.get_mastery(concept)
        
        # Determine response type based on tutor message and state
        response_type, correctness = self._decide_response_type(tutor_message, current_mastery)
        
        # Calculate mastery change based on response
        mastery_delta = self._calculate_mastery_delta(response_type, correctness)
        
        # Generate response
        if self.llm:
            response_text, internal_thought = self._generate_llm_response(
                tutor_message, concept, response_type, current_mastery
            )
        else:
            response_text, internal_thought = self._generate_template_response(
                tutor_message, concept, response_type
            )
        
        # Update state
        self._update_state(response_type, correctness, mastery_delta)
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "tutor",
            "content": tutor_message[:200],
        })
        self.conversation_history.append({
            "role": "student",
            "content": response_text[:200],
        })
        
        # Keep history bounded
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        metadata = {
            "intent": response_type,
            "correctness": correctness,
            "correctness_score": self._correctness_to_score(correctness),
            "mastery_before": current_mastery,
            "mastery_after": self.state.get_mastery(concept),
            "mastery_delta": mastery_delta,
            "engagement": self.state.engagement,
            "frustration": self.state.frustration,
            "internal_thought": internal_thought,
            "student_state": self.state.to_dict(),
        }
        
        logger.debug(f"student_response type={response_type} correctness={correctness} mastery_delta={mastery_delta:.3f}")
        
        return response_text, metadata
    
    def _decide_response_type(self, tutor_message: str, mastery: float) -> Tuple[str, str]:
        """
        Decide what type of response to give based on:
        - Tutor message content (is it a question?)
        - Current mastery level
        - Learning profile probabilities
        - Affective state
        """
        # Check if tutor is asking a question
        is_question = "?" in tutor_message or any(
            q in tutor_message.lower() for q in 
            ["what do you think", "can you", "try to", "how would", "why"]
        )
        
        # Check for frustration/give up
        if self.state.frustration > 0.7 and random.random() < self.profile.p_give_up * 3:
            return "give_up", "na"
        
        # If not a question, usually acknowledge or ask question
        if not is_question:
            if random.random() < self.profile.p_ask_question:
                return "question", "na"
            if random.random() < self.profile.p_express_confusion and mastery < 0.3:
                return "confusion", "na"
            return "acknowledge", "na"
        
        # It's a question - decide correctness probabilistically
        # P(correct) = p_correct_at_zero + mastery * (p_correct_at_mastery - p_correct_at_zero)
        p_correct = (
            self.profile.p_correct_at_zero + 
            mastery * (self.profile.p_correct_at_mastery - self.profile.p_correct_at_zero)
        )
        
        # Adjust for frustration (frustrated students perform worse)
        p_correct *= (1 - self.state.frustration * 0.3)
        
        # Adjust for confidence
        p_correct *= (0.7 + self.state.confidence * 0.3)
        
        roll = random.random()
        
        if roll < p_correct:
            return "answer", "correct"
        elif roll < p_correct + 0.2:  # 20% partial band
            return "answer", "partial"
        else:
            # Incorrect - but might ask question instead
            if random.random() < self.profile.p_express_confusion:
                return "confusion", "na"
            return "answer", "incorrect"
    
    def _calculate_mastery_delta(self, response_type: str, correctness: str) -> float:
        """Calculate mastery change based on response"""
        if response_type != "answer":
            # Non-answer responses have minimal mastery impact
            return 0.0
        
        if correctness == "correct":
            delta = self.profile.learning_rate
            # Bonus for consecutive correct
            if self.state.consecutive_correct > 0:
                delta *= 1.1
            return delta
        elif correctness == "partial":
            return self.profile.learning_rate * 0.3
        else:  # incorrect
            # Small negative or no change
            return -self.profile.learning_rate * 0.1
    
    def _update_state(self, response_type: str, correctness: str, mastery_delta: float):
        """Update internal student state after response"""
        # Update mastery
        self.state.update_mastery(self.state.current_concept, mastery_delta)
        
        # Update consecutive counters
        if correctness == "correct":
            self.state.consecutive_correct += 1
            self.state.consecutive_incorrect = 0
            self.state.total_correct += 1
            self.state.confidence = min(1.0, self.state.confidence + 0.1)
            self.state.frustration = max(0.0, self.state.frustration - 0.1)
        elif correctness == "incorrect":
            self.state.consecutive_correct = 0
            self.state.consecutive_incorrect += 1
            self.state.total_incorrect += 1
            self.state.confidence = max(0.0, self.state.confidence - 0.1)
            
            # Increase frustration if consecutive errors
            if self.state.consecutive_incorrect >= self.profile.frustration_threshold:
                self.state.frustration = min(1.0, self.state.frustration + 0.2)
        
        # Track hints
        if "hint" in response_type.lower():
            self.state.hints_received += 1
        
        # Engagement decay
        self.state.engagement *= 0.98  # Slight decay each turn
    
    def _generate_llm_response(
        self,
        tutor_message: str,
        concept: str,
        response_type: str,
        mastery: float,
    ) -> Tuple[str, str]:
        """Generate response using LLM"""
        # Build conversation context
        history_str = ""
        for turn in self.conversation_history[-6:]:
            role = "Tutor" if turn["role"] == "tutor" else "Student"
            history_str += f"{role}: {turn['content']}\n"
        
        # Build system prompt
        learner_desc = {
            LearnerType.FAST_LEARNER: "quick, confident, rarely confused",
            LearnerType.AVERAGE: "typical, sometimes confused, asks questions when stuck",
            LearnerType.STRUGGLING: "slow to understand, often confused, needs extra help",
            LearnerType.ANXIOUS: "nervous, second-guesses self, hesitant to answer",
            LearnerType.RUSHER: "impatient, wants to move fast, brief responses",
            LearnerType.DEEP_THINKER: "analytical, asks deep questions, wants to understand why",
            LearnerType.PASSIVE: "minimal engagement, short responses, rarely asks questions",
        }
        
        mastery_desc = (
            "novice" if mastery < 0.3 else
            "developing" if mastery < 0.6 else
            "proficient"
        )
        
        correctness_key = response_type if response_type != "answer" else f"{response_type}_{self._decide_response_type(tutor_message, mastery)[1]}"
        correctness_instruction = self.CORRECTNESS_INSTRUCTIONS.get(
            response_type, self.CORRECTNESS_INSTRUCTIONS["acknowledge"]
        )
        
        system_prompt = self.STUDENT_SYSTEM_PROMPT.format(
            learner_type=self.profile.learner_type.value.replace("_", " "),
            concept=concept,
            learning_style=learner_desc.get(self.profile.learner_type, "typical"),
            mastery_level=mastery_desc,
            mastery_pct=int(mastery * 100),
            engagement="engaged" if self.state.engagement > 0.5 else "disengaged",
            frustration="frustrated" if self.state.frustration > 0.5 else "calm",
            correctness_instruction=correctness_instruction,
        )
        
        user_prompt = f"""Recent conversation:
{history_str}

Tutor's latest message: "{tutor_message}"

Generate your response as this student. Remember: {correctness_instruction}"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Don't use json_mode - Gemini and other models may not support it
            result = self.llm.chat_completion(messages, max_tokens=200, temperature=0.8, json_mode=False)
            
            if result and "choices" in result:
                content = result["choices"][0]["message"]["content"].strip()
                
                # Try to parse as JSON if it looks like JSON
                if content.startswith("{"):
                    try:
                        parsed = json.loads(content)
                        return (
                            parsed.get("response", content),
                            parsed.get("internal_thought", "")
                        )
                    except json.JSONDecodeError:
                        pass
                
                # Extract text from any JSON-like wrapper
                if '"response"' in content:
                    import re
                    match = re.search(r'"response"\s*:\s*"([^"]+)"', content)
                    if match:
                        return match.group(1), ""
                
                # Return raw content, cleaning any stray JSON artifacts
                clean = content.replace('{"response":', '').replace('"}', '').strip(' "{}\n')
                return clean, ""
        except Exception as e:
            logger.warning(f"LLM student generation failed: {e}")
        
        # Fallback to template
        return self._generate_template_response(tutor_message, concept, response_type)
    
    def _generate_template_response(
        self,
        tutor_message: str,
        concept: str,
        response_type: str,
    ) -> Tuple[str, str]:
        """Generate template-based response as fallback"""
        templates = {
            "correct": [
                f"I think it's related to {concept}. Is that right?",
                f"So the answer involves {concept}?",
                f"Based on what you explained, I'd say it's about {concept}.",
            ],
            "partial": [
                f"I think I understand part of it - the {concept} part makes sense.",
                f"So {concept} is involved, but I'm not sure about the details.",
            ],
            "incorrect": [
                f"Hmm, isn't it the opposite of what {concept} suggests?",
                f"I thought {concept} worked differently...",
                f"Wait, so it's NOT about {concept}?",
            ],
            "question": [
                f"Can you explain why {concept} works that way?",
                f"I don't understand the part about {concept}.",
                f"What's the connection between this and {concept}?",
            ],
            "confusion": [
                f"I'm lost. Can you explain {concept} again more simply?",
                f"This is confusing. I don't get how {concept} relates.",
                "I'm not following at all...",
            ],
            "acknowledge": [
                "Okay, I think I get it now.",
                "That makes sense, thanks!",
                "Got it, let me think about that.",
                "I see.",
            ],
            "give_up": [
                "This is too hard, I give up.",
                "I don't think I can understand this.",
                "Can we move on to something else?",
            ],
        }
        
        template_list = templates.get(response_type, templates["acknowledge"])
        response = random.choice(template_list)
        
        return response, f"Using template for {response_type}"
    
    def _correctness_to_score(self, correctness: str) -> float:
        """Convert correctness string to numeric score"""
        scores = {
            "correct": 1.0,
            "partial": 0.5,
            "incorrect": 0.0,
            "na": 0.0,
        }
        return scores.get(correctness, 0.0)
    
    def get_mastery(self, concept: Optional[str] = None) -> float:
        """Get current mastery for concept or current concept"""
        concept = concept or self.state.current_concept
        return self.state.get_mastery(concept)
    
    def reset_for_concept(self, concept: str):
        """Reset session state for a new concept (but keep overall mastery)"""
        self.state.current_concept = concept
        self.state.current_step = 0
        self.state.consecutive_correct = 0
        self.state.consecutive_incorrect = 0
        self.state.hints_received = 0
        # Slight engagement reset
        self.state.engagement = min(1.0, self.state.engagement + 0.1)
        self.state.frustration = max(0.0, self.state.frustration - 0.2)


# =============================================================================
# STUDENT POPULATION GENERATOR
# =============================================================================

class StudentPopulation:
    """
    Generate diverse student population for training data.
    
    Creates students with varied learning profiles to ensure
    the tutor sees diverse interaction patterns.
    """
    
    # Distribution of learner types in typical classroom
    DEFAULT_DISTRIBUTION = {
        LearnerType.FAST_LEARNER: 0.15,
        LearnerType.AVERAGE: 0.40,
        LearnerType.STRUGGLING: 0.20,
        LearnerType.ANXIOUS: 0.10,
        LearnerType.RUSHER: 0.05,
        LearnerType.DEEP_THINKER: 0.05,
        LearnerType.PASSIVE: 0.05,
    }
    
    def __init__(
        self,
        llm_client: Optional["LLMClient"] = None,
        distribution: Optional[Dict[LearnerType, float]] = None,
    ):
        self.llm = llm_client
        self.distribution = distribution or self.DEFAULT_DISTRIBUTION
    
    def sample_student(self) -> LLMStudentSimulator:
        """Sample a random student from the population distribution"""
        learner_types = list(self.distribution.keys())
        weights = list(self.distribution.values())
        
        learner_type = random.choices(learner_types, weights=weights, k=1)[0]
        profile = LearningProfile.from_learner_type(learner_type)
        
        # Add some random variation
        profile.learning_rate *= random.uniform(0.8, 1.2)
        profile.prior_knowledge *= random.uniform(0.8, 1.2)
        profile.prior_knowledge = max(0.0, min(1.0, profile.prior_knowledge))
        
        return LLMStudentSimulator(profile, self.llm)
    
    def generate_batch(self, count: int) -> List[LLMStudentSimulator]:
        """Generate a batch of students"""
        return [self.sample_student() for _ in range(count)]
    
    def generate_balanced(self) -> List[LLMStudentSimulator]:
        """Generate one student of each type"""
        students = []
        for learner_type in LearnerType:
            profile = LearningProfile.from_learner_type(learner_type)
            students.append(LLMStudentSimulator(profile, self.llm))
        return students


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_student(
    learner_type: str = "average",
    llm_client: Optional["LLMClient"] = None,
) -> LLMStudentSimulator:
    """
    Convenience function to create a student simulator.
    
    Args:
        learner_type: One of: fast_learner, average, struggling, anxious, rusher, deep_thinker, passive
        llm_client: Optional LLM client for realistic responses
    """
    try:
        lt = LearnerType(learner_type.lower())
    except ValueError:
        lt = LearnerType.AVERAGE
    
    profile = LearningProfile.from_learner_type(lt)
    return LLMStudentSimulator(profile, llm_client)
