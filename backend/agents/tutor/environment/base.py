from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar, Protocol


S = TypeVar("S")


@dataclass
class EnvironmentTransition(Generic[S]):
    """Result of a single environment step.

    This is a light wrapper around the underlying MDP transition outcome,
    specialised for the environment layer. It is intentionally minimal for
    the MVP.
    """

    state: S
    outputs: Dict[str, Any]
    terminated: bool = False
    termination_reason: Optional[str] = None


class BaseEnvironment(Protocol, Generic[S]):
    """Minimal environment interface for the 3-layer MDP.

    Each concrete environment (session, concept, tutor) implements this
    interface with a deterministic transition function.
    """

    state: S

    def reset(self, *args: Any, **kwargs: Any) -> S:
        """Reset to an initial state and return it."""

    def step(self, action: Any, **kwargs: Any) -> EnvironmentTransition[S]:
        """Apply a deterministic transition given an action and optional kwargs."""

    def get_state(self) -> S:
        """Return the current state."""

    def is_terminated(self) -> bool:
        """Return True if the episode is terminated in this layer."""
