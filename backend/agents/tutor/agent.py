"""
Tutor Agent Entry Point

This module provides the `tutor_agent` function that serves as the entry point
for the tutor agent in the orchestrator dispatch system. It wraps the new
TutorOrchestrator-based implementation.
"""

from typing import Dict, Any


def tutor_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for the tutor agent.
    
    This function is called by the orchestrator dispatch system and delegates
    to the environment orchestrator which uses the new TutorOrchestrator.
    
    Args:
        payload: Request payload containing user_id, message, and optional controls
        
    Returns:
        Response dict with messages, ui_mode, and debug info
    """
    from .environment.orchestrator import run_environment_turn
    
    # Extract required fields
    user_id = payload.get("user_id")
    if not user_id:
        raise ValueError("user_id is required")
    
    # Delegate to the environment orchestrator
    return run_environment_turn(payload)
