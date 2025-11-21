"""
Base classes for the 3-layer MDP environment.

Provides abstract interfaces and common functionality for all environment layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar


@dataclass
class EnvironmentState:
    """Base class for environment state.
    
    All layer-specific states should inherit from this to ensure
    they carry minimal required metadata.
    """
    
    episode_id: str
    session_id: str
    user_id: str
    terminated: bool = False
    termination_reason: Optional[str] = None


@dataclass
class EnvironmentTransition:
    """Result of an environment step.
    
    Contains the new state, any outputs/artifacts from the transition,
    and metadata about what occurred.
    """
    
    next_state: EnvironmentState
    outputs: Dict[str, Any]
    info: Dict[str, Any]
    terminated: bool = False
    termination_reason: Optional[str] = None


S = TypeVar("S", bound=EnvironmentState)


class BaseEnvironment(ABC, Generic[S]):
    """Abstract base class for all environment layers.
    
    Each environment layer (Session, Concept, Tutor) implements this interface
    to provide consistent state management and transition logic.
    """
    
    def __init__(self, state: S):
        """Initialize environment with a state.
        
        Args:
            state: Initial state for this environment
        """
        self.state = state
    
    @abstractmethod
    def reset(self) -> S:
        """Reset the environment to initial state.
        
        Returns:
            Initial state for a new episode
        """
        pass
    
    @abstractmethod
    def step(self, action: Any, **kwargs) -> EnvironmentTransition:
        """Execute one step in the environment.
        
        Args:
            action: Action to take
            **kwargs: Additional context needed for the step
            
        Returns:
            EnvironmentTransition with next state and outputs
        """
        pass
    
    def get_state(self) -> S:
        """Get current environment state.
        
        Returns:
            Current state
        """
        return self.state
    
    def is_terminated(self) -> bool:
        """Check if environment episode has terminated.
        
        Returns:
            True if episode is done
        """
        return self.state.terminated
    
    def get_termination_reason(self) -> Optional[str]:
        """Get reason for termination if terminated.
        
        Returns:
            Termination reason or None
        """
        return self.state.termination_reason
