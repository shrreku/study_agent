"""
Simple mastery estimator for the environment architecture.

Provides basic mastery updates based on step completion and interaction quality.
"""

from __future__ import annotations

from typing import Optional, Tuple


class SimpleMasteryEstimator:
    """Simple heuristic-based mastery estimator for MVP.
    
    This provides basic mastery updates. Can be replaced with more
    sophisticated models later.
    """
    
    def __init__(
        self,
        step_completion_gain: float = 0.05,
        correct_answer_gain: float = 0.1,
        incorrect_answer_penalty: float = -0.02,
    ):
        """Initialize estimator with gain parameters.
        
        Args:
            step_completion_gain: Mastery gain for completing a step
            correct_answer_gain: Mastery gain for correct answer
            incorrect_answer_penalty: Mastery change for wrong answer
        """
        self.step_completion_gain = step_completion_gain
        self.correct_answer_gain = correct_answer_gain
        self.incorrect_answer_penalty = incorrect_answer_penalty
    
    def estimate_from_step_completion(
        self,
        current_mastery: float,
        step_type: str,
    ) -> Tuple[float, float]:
        """Estimate mastery change from completing a plan step.
        
        Args:
            current_mastery: Current mastery level (0-1)
            step_type: Type of step completed
            
        Returns:
            Tuple of (new_mastery, mastery_delta)
        """
        # Different step types have different learning value
        step_multiplier = 1.0
        
        step_lower = step_type.lower()
        if "explain" in step_lower or "introduce" in step_lower:
            step_multiplier = 1.0
        elif "practice" in step_lower or "exercise" in step_lower:
            step_multiplier = 1.5  # Practice is more valuable
        elif "example" in step_lower:
            step_multiplier = 1.2
        elif "summary" in step_lower:
            step_multiplier = 0.8  # Summary is review, less gain
        
        # Calculate gain with diminishing returns
        # As mastery increases, gains decrease
        diminishing_factor = 1.0 - current_mastery
        gain = self.step_completion_gain * step_multiplier * diminishing_factor
        
        new_mastery = min(1.0, current_mastery + gain)
        
        return new_mastery, gain
    
    def estimate_from_answer(
        self,
        current_mastery: float,
        answer_correct: bool,
    ) -> Tuple[float, float]:
        """Estimate mastery change from answering a question.
        
        Args:
            current_mastery: Current mastery level (0-1)
            answer_correct: Whether answer was correct
            
        Returns:
            Tuple of (new_mastery, mastery_delta)
        """
        if answer_correct:
            # Positive gain for correct answer
            gain = self.correct_answer_gain * (1.0 - current_mastery)
            new_mastery = min(1.0, current_mastery + gain)
        else:
            # Small penalty for incorrect answer
            gain = self.incorrect_answer_penalty
            new_mastery = max(0.0, current_mastery + gain)
        
        return new_mastery, gain
    
    def estimate_from_phase_completion(
        self,
        current_mastery: float,
        phase: str,
        steps_completed: int,
    ) -> Tuple[float, float]:
        """Estimate mastery after completing a learning phase.
        
        Args:
            current_mastery: Current mastery level (0-1)
            phase: Phase completed (learning, practice, assessment)
            steps_completed: Number of steps in the phase
            
        Returns:
            Tuple of (new_mastery, mastery_delta)
        """
        # Base gain depends on phase and number of steps
        base_gain = min(0.2, steps_completed * 0.03)
        
        phase_lower = phase.lower()
        if "assessment" in phase_lower or "quiz" in phase_lower:
            # Assessment phase gives confidence boost
            phase_multiplier = 1.3
        elif "practice" in phase_lower:
            phase_multiplier = 1.2
        else:
            phase_multiplier = 1.0
        
        gain = base_gain * phase_multiplier * (1.0 - current_mastery)
        new_mastery = min(1.0, current_mastery + gain)
        
        return new_mastery, gain


def make_simple_mastery_estimator() -> SimpleMasteryEstimator:
    """Factory function for simple mastery estimator.
    
    Returns:
        SimpleMasteryEstimator instance
    """
    return SimpleMasteryEstimator()
