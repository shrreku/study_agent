"""
Runtime context for tutor agent turns.

Provides the TurnContext dataclass used by the API layer.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TurnContext:
    """
    Context for a single tutor turn.
    
    This is the input context provided by the API layer to the orchestrator.
    """
    session_id: str
    user_id: str
    turn_index: int
    message: str
    
    # Optional fields
    target_concepts: Optional[List[str]] = None
    resource_id: Optional[str] = None
    dry_run: bool = False
    emit_state_requested: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)
    
    def get_payload_field(self, key: str, default: Any = None) -> Any:
        """Get a field from the payload dict."""
        return self.payload.get(key, default)
