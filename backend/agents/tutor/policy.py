"""
Policy helper functions for the tutor agent.

Provides functions to map mastery levels to human-readable labels.
"""

from typing import Optional


def level_for_mastery(mastery: float) -> str:
    """
    Convert numeric mastery score to a human-readable level label.
    
    Args:
        mastery: Numeric mastery score (0.0 to 1.0)
        
    Returns:
        Level label: 'beginner', 'intermediate', 'advanced', or 'expert'
    """
    if mastery is None:
        return "unknown"
    
    mastery = float(mastery)
    
    if mastery < 0.3:
        return "beginner"
    elif mastery < 0.6:
        return "intermediate"
    elif mastery < 0.85:
        return "advanced"
    else:
        return "expert"
