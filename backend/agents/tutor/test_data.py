"""Test data generation for tutor MDP testing.

This module provides helpers to create comprehensive test data for testing
the tutor MDP, including concept plans with all pedagogical action types.
"""
from __future__ import annotations

from typing import Any, Dict, List
from uuid import uuid4


def create_test_concept_plan(concept_id: str = "test_concept") -> Dict[str, Any]:
    """Create a comprehensive concept plan with all pedagogical action types.
    
    Returns a plan dict with steps covering all 8 pedagogical actions:
    - EXPLAIN (introduction/explanation)
    - DEFINE_TERM (definition)
    - WORKED_EXAMPLE (example)
    - ASK_QUESTION (practice question)
    - GUIDED_PRACTICE (guided practice)
    - REFLECTION_PROMPT (reflection)
    - SUMMARY (summary)
    - QUIZ_MCQ (quiz phase)
    
    Args:
        concept_id: Concept identifier
        
    Returns:
        Concept plan dictionary
    """
    plan_id = f"plan_{uuid4().hex[:8]}"
    
    steps = [
        {
            "step_type": "introduction",
            "instruction": "Introduce the fundamental concepts of heat transfer",
            "subgoal": "Understand what heat transfer is",
            "pedagogical_action": "EXPLAIN",
            "estimated_duration": 120,
        },
        {
            "step_type": "definition",
            "instruction": "Define key terms: conduction, convection, radiation",
            "subgoal": "Learn core terminology",
            "pedagogical_action": "DEFINE_TERM",
            "estimated_duration": 90,
        },
        {
            "step_type": "example",
            "instruction": "Show worked example of conduction calculation",
            "subgoal": "See conduction in action",
            "pedagogical_action": "WORKED_EXAMPLE",
            "estimated_duration": 150,
        },
        {
            "step_type": "explanation",
            "instruction": "Explain the differences between conduction, convection, and radiation",
            "subgoal": "Understand heat transfer modes",
            "pedagogical_action": "EXPLAIN",
            "estimated_duration": 120,
        },
        {
            "step_type": "practice",
            "instruction": "Ask student to identify heat transfer mode in scenarios",
            "subgoal": "Apply knowledge to scenarios",
            "pedagogical_action": "ASK_QUESTION",
            "estimated_duration": 180,
        },
        {
            "step_type": "practice",
            "instruction": "Guide student through solving a heat transfer problem step-by-step",
            "subgoal": "Practice problem solving with guidance",
            "pedagogical_action": "GUIDED_PRACTICE",
            "estimated_duration": 240,
        },
        {
            "step_type": "reflection",
            "instruction": "Ask student to reflect on what they've learned about heat transfer",
            "subgoal": "Consolidate learning",
            "pedagogical_action": "REFLECTION_PROMPT",
            "estimated_duration": 90,
        },
        {
            "step_type": "summary",
            "instruction": "Summarize key concepts of heat transfer",
            "subgoal": "Review and consolidate",
            "pedagogical_action": "SUMMARY",
            "estimated_duration": 90,
        },
    ]
    
    return {
        "plan_id": plan_id,
        "concept_id": concept_id,
        "steps": steps,
        "total_steps": len(steps),
        "estimated_total_duration": sum(s["estimated_duration"] for s in steps),
    }


def create_test_mastery_map(concept_id: str = "test_concept") -> Dict[str, float]:
    """Create initial mastery map for testing.
    
    Args:
        concept_id: Concept identifier
        
    Returns:
        Dictionary mapping concept_id to mastery score
    """
    return {
        concept_id: 0.2,  # Starting mastery
        "conduction": 0.15,
        "convection": 0.1,
        "radiation": 0.1,
    }


def create_test_target_concepts() -> List[str]:
    """Create list of target concepts for testing.
    
    Returns:
        List of concept IDs to study
    """
    return ["test_concept", "advanced_heat_transfer"]


def get_test_concept_metadata() -> Dict[str, Any]:
    """Get metadata about the test concept.
    
    Returns:
        Concept metadata dictionary
    """
    return {
        "concept_id": "test_concept",
        "display_name": "Heat Transfer Fundamentals",
        "description": "Basic principles of heat transfer: conduction, convection, and radiation",
        "difficulty": "beginner",
        "prerequisites": [],
        "target_mastery": 0.8,
    }
