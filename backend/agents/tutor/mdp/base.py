from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar


class MDPState(Protocol):
    """Marker protocol for MDP state objects.

    Concrete state types (e.g., ConceptState) should be lightweight
    dataclasses that carry all the environment state needed for
    decision-making and logging.
    """


class MDPObservation(Protocol):
    """Marker protocol for MDP observation objects."""


class MDPAction(Protocol):
    """Marker protocol for MDP action enums / literals."""


S = TypeVar("S", bound=MDPState)
O = TypeVar("O", bound=MDPObservation)
A = TypeVar("A")  # Concrete action type (e.g., ConceptMDPAction)


@dataclass
class StepOutcome(Generic[S, O, A]):
    """Generic container for the outcome of a single MDP step.

    This is intentionally minimal and domain-agnostic so it can be reused
    for concept-level and session-level MDPs.
    """

    state: S
    observation: O
    action: A
    reward: Dict[str, float] = field(default_factory=dict)
    terminated: bool = False
    termination_reason: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)
