"""
Pedagogical Reward Model for Tutor RL Training

This module implements the composite reward function based on:
- Mastery gain (verifiable)
- Pedagogical quality (LLM judge)
- Grounding/factuality (verifiable)
- Strategy alignment (verifiable)
- Efficiency metrics (verifiable)

Reference: "From Problem-Solving to Teaching Problem-Solving" (arXiv:2505.15607)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from mdp.llm_client import LLMClient

logger = logging.getLogger(__name__)


class PedagogicalStrategy(Enum):
    """Explicit pedagogical micro-actions for tutor policy"""
    
    # Scaffolding strategies (preferred - avoid answer leakage)
    SOCRATIC_QUESTION = "socratic_question"  # Guide via questioning
    HINT = "hint"                            # Partial reveal
    WORKED_EXAMPLE = "worked_example"        # Step-by-step demo
    ANALOGY = "analogy"                      # Connect to known concept
    NUDGE = "nudge"                          # Gentle redirection
    
    # Direct instruction (when scaffolding insufficient)
    EXPLAIN = "explain"                      # Clear explanation
    CORRECT_MISCONCEPTION = "correct"        # Address specific error
    SUMMARIZE = "summarize"                  # Consolidate learning
    ELABORATE = "elaborate"                  # Add depth to understanding
    
    # Assessment strategies
    CONCEPT_CHECK = "concept_check"          # Quick verification question
    CHALLENGE = "challenge"                  # Harder problem
    REFLECT = "reflect"                      # Metacognitive prompt
    
    # Flow control
    ADVANCE = "advance"                      # Move to next step
    REPLAN = "replan"                        # Regenerate plan
    CONCLUDE = "conclude"                    # End session
    
    @classmethod
    def scaffolding_strategies(cls) -> List["PedagogicalStrategy"]:
        """Strategies that avoid direct answer leakage"""
        return [
            cls.SOCRATIC_QUESTION, cls.HINT, cls.WORKED_EXAMPLE, 
            cls.ANALOGY, cls.NUDGE, cls.CONCEPT_CHECK, cls.REFLECT
        ]
    
    @classmethod
    def from_string(cls, s: str) -> "PedagogicalStrategy":
        """Parse strategy from string, with fuzzy matching"""
        s_lower = s.lower().strip()
        for strategy in cls:
            if strategy.value in s_lower or s_lower in strategy.value:
                return strategy
        # Default fallback
        return cls.EXPLAIN


@dataclass
class RewardWeights:
    """Configurable weights for reward components"""
    mastery: float = 0.35
    pedagogical: float = 0.30
    grounding: float = 0.15
    efficiency: float = 0.10
    strategy: float = 0.10
    
    def normalize(self) -> "RewardWeights":
        total = self.mastery + self.pedagogical + self.grounding + self.efficiency + self.strategy
        if total == 0:
            return self
        return RewardWeights(
            mastery=self.mastery / total,
            pedagogical=self.pedagogical / total,
            grounding=self.grounding / total,
            efficiency=self.efficiency / total,
            strategy=self.strategy / total,
        )


@dataclass
class PedagogicalReward:
    """
    Composite reward for a single tutor turn.
    
    Based on arXiv:2505.15607 reward design:
    r = r_mastery + r_ped * gate(no_leakage) - λ * penalty(violations)
    """
    
    # ===== MASTERY OUTCOME (Verifiable) =====
    mastery_delta: float = 0.0  # Post-turn mastery - Pre-turn mastery [-1, 1]
    
    # ===== PEDAGOGICAL QUALITY (LLM Judge) =====
    no_answer_leakage: bool = True      # Did NOT give away the answer
    scaffolding_quality: float = 0.5    # 0-1: Used hints/questions appropriately
    helpfulness: float = 0.5            # 0-1: Addressed student's actual need
    tone_quality: float = 0.5           # 0-1: Encouraging, patient, clear
    
    # ===== GROUNDING (Verifiable) =====
    factual_accuracy: float = 0.5       # 0-1: Claims match retrieved chunks
    no_hallucination: bool = True       # Hard constraint
    
    # ===== EFFICIENCY (Verifiable) =====
    response_length_penalty: float = 0.0  # Penalty for overly long responses
    hint_efficiency: float = 1.0          # Fewer hints to achieve mastery = better
    
    # ===== STRATEGY ALIGNMENT (Verifiable) =====
    strategy_followed: float = 0.5      # Did response match declared [Strategy]?
    declared_strategy: Optional[PedagogicalStrategy] = None
    
    # Weights
    weights: RewardWeights = field(default_factory=RewardWeights)
    
    # Metadata
    judge_confidence: float = 0.5
    flags: List[str] = field(default_factory=list)
    
    def compute_total(self, lambda_penalty: float = 0.5) -> float:
        """
        Compute weighted total reward with hard constraints.
        
        Based on paper formula:
        r = r_sol + r_ped * 1{all_judges_accept} - λ * 1{any_judge_rejects}
        """
        # Hard constraint: hallucination is unacceptable
        if not self.no_hallucination:
            self.flags.append("HALLUCINATION_PENALTY")
            return -1.0
        
        # Hard constraint: answer leakage severely reduces reward
        if not self.no_answer_leakage:
            self.flags.append("ANSWER_LEAKAGE_PENALTY")
            # Still give some credit for mastery gain, but penalized
            return max(-0.5, self.mastery_delta * 0.3 - lambda_penalty)
        
        # Normalize weights
        w = self.weights.normalize()
        
        # Component scores
        ped_score = (self.scaffolding_quality + self.helpfulness + self.tone_quality) / 3
        efficiency_score = (1.0 - self.response_length_penalty + self.hint_efficiency) / 2
        
        # Weighted sum
        r_mastery = self.mastery_delta * w.mastery
        r_ped = ped_score * w.pedagogical
        r_ground = self.factual_accuracy * w.grounding
        r_eff = efficiency_score * w.efficiency
        r_strat = self.strategy_followed * w.strategy
        
        total = r_mastery + r_ped + r_ground + r_eff + r_strat
        
        # Clip to [-1, 1]
        return max(-1.0, min(1.0, total))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/training"""
        return {
            "components": {
                "mastery": {"score": self.mastery_delta, "weight": self.weights.mastery},
                "pedagogical": {
                    "score": (self.scaffolding_quality + self.helpfulness + self.tone_quality) / 3,
                    "scaffolding": self.scaffolding_quality,
                    "helpfulness": self.helpfulness,
                    "tone": self.tone_quality,
                    "no_leakage": self.no_answer_leakage,
                    "weight": self.weights.pedagogical,
                },
                "grounding": {
                    "score": self.factual_accuracy,
                    "no_hallucination": self.no_hallucination,
                    "weight": self.weights.grounding,
                },
                "efficiency": {
                    "score": (1.0 - self.response_length_penalty + self.hint_efficiency) / 2,
                    "length_penalty": self.response_length_penalty,
                    "hint_efficiency": self.hint_efficiency,
                    "weight": self.weights.efficiency,
                },
                "strategy": {
                    "score": self.strategy_followed,
                    "declared": self.declared_strategy.value if self.declared_strategy else None,
                    "weight": self.weights.strategy,
                },
            },
            "total": self.compute_total(),
            "flags": self.flags,
            "judge_confidence": self.judge_confidence,
        }


class PedagogicalRewardModel:
    """
    Compute rewards for tutor turns using:
    1. Verifiable signals (mastery, grounding, strategy alignment)
    2. LLM judge for pedagogical quality
    """
    
    # Answer leakage detection patterns
    LEAKAGE_PATTERNS = [
        r"the answer is",
        r"the correct answer",
        r"the solution is",
        r"= \d+",  # Direct numerical answer
        r"therefore[,\s]+\w+ equals",
        r"so the result is",
    ]
    
    # Scaffolding indicators
    SCAFFOLDING_PATTERNS = [
        r"\?$",  # Ends with question
        r"what do you think",
        r"can you",
        r"try to",
        r"consider",
        r"think about",
        r"let's see if",
        r"what if",
        r"how would",
    ]
    
    def __init__(self, llm_client: Optional["LLMClient"] = None):
        self.llm_client = llm_client
        self.weights = RewardWeights()
    
    def compute_reward(
        self,
        response_text: str,
        declared_strategy: Optional[PedagogicalStrategy],
        mastery_before: float,
        mastery_after: float,
        retrieved_chunks: List[str],
        hints_used_total: int,
        max_response_length: int = 300,
        use_llm_judge: bool = True,
    ) -> PedagogicalReward:
        """
        Compute full reward for a tutor turn.
        
        Args:
            response_text: The tutor's response
            declared_strategy: The strategy the tutor declared (from [Strategy: X] tag)
            mastery_before: Student mastery before this turn
            mastery_after: Student mastery after this turn
            retrieved_chunks: RAG chunks used for grounding check
            hints_used_total: Total hints given in session so far
            max_response_length: Penalty threshold for response length
            use_llm_judge: Whether to call LLM for pedagogical quality
        """
        reward = PedagogicalReward(weights=self.weights)
        
        # 1. Mastery delta (verifiable)
        reward.mastery_delta = mastery_after - mastery_before
        
        # 2. Answer leakage check (verifiable via patterns)
        reward.no_answer_leakage = not self._check_answer_leakage(response_text)
        
        # 3. Scaffolding quality (heuristic + optional LLM)
        reward.scaffolding_quality = self._compute_scaffolding_score(
            response_text, declared_strategy
        )
        
        # 4. Grounding check (verifiable)
        reward.factual_accuracy, reward.no_hallucination = self._check_grounding(
            response_text, retrieved_chunks
        )
        
        # 5. Response length penalty (verifiable)
        word_count = len(response_text.split())
        if word_count > max_response_length:
            reward.response_length_penalty = min(0.5, (word_count - max_response_length) / 100)
            reward.flags.append("RESPONSE_TOO_LONG")
        
        # 6. Hint efficiency (verifiable)
        # Diminishing returns after 3 hints
        reward.hint_efficiency = max(0.0, 1.0 - (hints_used_total / 5))
        
        # 7. Strategy alignment (verifiable)
        reward.declared_strategy = declared_strategy
        reward.strategy_followed = self._check_strategy_alignment(
            response_text, declared_strategy
        )
        
        # 8. LLM judge for subjective quality (optional, expensive)
        if use_llm_judge and self.llm_client:
            judge_scores = self._call_pedagogical_judge(response_text, declared_strategy)
            reward.helpfulness = judge_scores.get("helpfulness", 0.5)
            reward.tone_quality = judge_scores.get("tone", 0.5)
            reward.judge_confidence = judge_scores.get("confidence", 0.5)
            
            # Override scaffolding if judge disagrees significantly
            judge_scaffolding = judge_scores.get("scaffolding", reward.scaffolding_quality)
            reward.scaffolding_quality = (reward.scaffolding_quality + judge_scaffolding) / 2
        
        logger.debug(f"Computed reward: {reward.compute_total():.3f}", extra=reward.to_dict())
        return reward
    
    def _check_answer_leakage(self, response: str) -> bool:
        """Check if response gives away the answer directly"""
        response_lower = response.lower()
        for pattern in self.LEAKAGE_PATTERNS:
            if re.search(pattern, response_lower):
                return True
        return False
    
    def _compute_scaffolding_score(
        self, 
        response: str, 
        strategy: Optional[PedagogicalStrategy]
    ) -> float:
        """
        Score how well the response uses scaffolding techniques.
        Higher = more scaffolding (questions, hints) vs direct answers.
        """
        response_lower = response.lower()
        
        # Count scaffolding indicators
        scaffolding_count = sum(
            1 for pattern in self.SCAFFOLDING_PATTERNS
            if re.search(pattern, response_lower)
        )
        
        # Base score from pattern matching
        base_score = min(1.0, scaffolding_count * 0.2)
        
        # Bonus if declared strategy is scaffolding type
        if strategy and strategy in PedagogicalStrategy.scaffolding_strategies():
            base_score = min(1.0, base_score + 0.2)
        
        return base_score
    
    def _check_grounding(
        self, 
        response: str, 
        chunks: List[str]
    ) -> tuple[float, bool]:
        """
        Check if response is grounded in retrieved chunks.
        Returns (accuracy_score, no_hallucination_flag)
        """
        if not chunks:
            # No chunks to ground against - be lenient
            return 0.7, True
        
        # Simple keyword overlap check (production should use embeddings)
        response_words = set(response.lower().split())
        chunk_words = set()
        for chunk in chunks:
            chunk_words.update(chunk.lower().split())
        
        # Filter out common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                       "being", "have", "has", "had", "do", "does", "did", "will",
                       "would", "could", "should", "may", "might", "must", "shall",
                       "can", "to", "of", "in", "for", "on", "with", "at", "by",
                       "from", "as", "into", "through", "during", "before", "after",
                       "above", "below", "between", "under", "again", "further",
                       "then", "once", "here", "there", "when", "where", "why",
                       "how", "all", "each", "few", "more", "most", "other", "some",
                       "such", "no", "nor", "not", "only", "own", "same", "so",
                       "than", "too", "very", "just", "and", "but", "if", "or",
                       "because", "until", "while", "this", "that", "these", "those"}
        
        response_content = response_words - common_words
        chunk_content = chunk_words - common_words
        
        if not response_content:
            return 0.8, True
        
        overlap = len(response_content & chunk_content)
        coverage = overlap / len(response_content) if response_content else 0
        
        # Hallucination flag: if less than 30% grounded, flag it
        no_hallucination = coverage > 0.3
        
        return min(1.0, coverage * 1.5), no_hallucination
    
    def _check_strategy_alignment(
        self, 
        response: str, 
        strategy: Optional[PedagogicalStrategy]
    ) -> float:
        """Check if response matches the declared strategy"""
        if not strategy:
            return 0.5  # No strategy declared, neutral
        
        response_lower = response.lower()
        
        # Strategy-specific patterns
        strategy_patterns = {
            PedagogicalStrategy.SOCRATIC_QUESTION: [r"\?", r"what", r"why", r"how"],
            PedagogicalStrategy.HINT: [r"hint", r"consider", r"think about", r"clue"],
            PedagogicalStrategy.WORKED_EXAMPLE: [r"for example", r"let's work through", r"step \d"],
            PedagogicalStrategy.ANALOGY: [r"like", r"similar to", r"think of it as", r"imagine"],
            PedagogicalStrategy.EXPLAIN: [r"means", r"is defined as", r"refers to", r"because"],
            PedagogicalStrategy.CORRECT_MISCONCEPTION: [r"actually", r"not quite", r"common mistake"],
            PedagogicalStrategy.CONCEPT_CHECK: [r"\?", r"can you", r"what is", r"tell me"],
            PedagogicalStrategy.REFLECT: [r"think about", r"how did you", r"what did you learn"],
        }
        
        patterns = strategy_patterns.get(strategy, [])
        if not patterns:
            return 0.5
        
        matches = sum(1 for p in patterns if re.search(p, response_lower))
        return min(1.0, matches / len(patterns) + 0.3)  # Base 0.3 for attempting
    
    def _call_pedagogical_judge(
        self, 
        response: str, 
        strategy: Optional[PedagogicalStrategy]
    ) -> Dict[str, float]:
        """Call LLM to judge pedagogical quality (expensive, optional)"""
        if not self.llm_client:
            return {"helpfulness": 0.5, "tone": 0.5, "scaffolding": 0.5, "confidence": 0.5}
        
        judge_prompt = f"""Rate this tutor response on pedagogical quality (0.0 to 1.0):

Response: "{response[:500]}"
Declared Strategy: {strategy.value if strategy else "none"}

Return JSON with:
- helpfulness: Did it address student's need? (0-1)
- tone: Is it encouraging and clear? (0-1)  
- scaffolding: Does it guide rather than tell? (0-1)
- confidence: Your confidence in these ratings (0-1)
"""
        try:
            result = self.llm_client.call_json(
                "You are a pedagogical quality judge.", 
                judge_prompt
            )
            return {
                "helpfulness": float(result.get("helpfulness", 0.5)),
                "tone": float(result.get("tone", 0.5)),
                "scaffolding": float(result.get("scaffolding", 0.5)),
                "confidence": float(result.get("confidence", 0.5)),
            }
        except Exception as e:
            logger.warning(f"Judge call failed: {e}")
            return {"helpfulness": 0.5, "tone": 0.5, "scaffolding": 0.5, "confidence": 0.5}


def parse_tutor_output(raw_output: str) -> tuple[str, Optional[PedagogicalStrategy], str]:
    """
    Parse tutor output in format:
    <think>reasoning</think>
    [Strategy: STRATEGY_NAME]
    Response text
    
    Returns: (thinking, strategy, response_text)
    """
    thinking = ""
    strategy = None
    response_text = raw_output
    
    # Extract thinking
    think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        response_text = raw_output[think_match.end():].strip()
    
    # Extract strategy
    strategy_match = re.search(r"\[Strategy:\s*(\w+)\]", response_text, re.IGNORECASE)
    if strategy_match:
        strategy_str = strategy_match.group(1)
        strategy = PedagogicalStrategy.from_string(strategy_str)
        response_text = response_text[strategy_match.end():].strip()
    
    return thinking, strategy, response_text
