from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ValidatorComponentResult:
    name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "details": self.details,
            "flags": list(self.flags),
        }


@dataclass(frozen=True)
class ValidatorContext:
    observation: Dict[str, Any]
    response_text: str
    response_metadata: Dict[str, Any]


__all__ = [
    "ValidatorComponentResult",
    "ValidatorContext",
]

