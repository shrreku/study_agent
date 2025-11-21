"""
Button controller for the environment architecture.

Handles parsing button clicks and mapping them to environment actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ButtonClick:
    """Represents a button click from the user."""
    
    button_label: str
    button_type: str = "continue"  # continue, skip, end, etc.
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ButtonController:
    """Parses and handles button interactions for the environment."""
    
    @staticmethod
    def parse_from_payload(payload: Dict[str, Any]) -> Optional[ButtonClick]:
        """Parse button click from frontend payload.
        
        Args:
            payload: Frontend payload dictionary
            
        Returns:
            ButtonClick if button was clicked, None otherwise
        """
        # Check various payload formats
        if payload.get("button_clicked"):
            button_label = payload.get("button_label", "Continue")
            return ButtonClick(
                button_label=button_label,
                button_type=ButtonController._infer_button_type(button_label),
            )
        
        # Check for confirmed_action (legacy format)
        if payload.get("confirmed_action"):
            action = payload["confirmed_action"]
            if action in {"continue", "next", "yes"}:
                return ButtonClick(
                    button_label="Continue",
                    button_type="continue",
                )
        
        # Check for step_control
        step_control = payload.get("step_control")
        if isinstance(step_control, dict):
            control_type = step_control.get("type")
            if control_type in {"continue", "next"}:
                return ButtonClick(
                    button_label="Continue",
                    button_type="continue",
                )
        
        return None
    
    @staticmethod
    def _infer_button_type(label: str) -> str:
        """Infer button type from label.
        
        Args:
            label: Button label text
            
        Returns:
            Button type string
        """
        label_lower = label.lower().strip()
        
        if label_lower in {"continue", "next", "ok", "yes"}:
            return "continue"
        elif label_lower in {"skip", "skip to next"}:
            return "skip"
        elif label_lower in {"end", "end session", "finish"}:
            return "end"
        else:
            return "continue"
    
    @staticmethod
    def was_button_clicked(payload: Dict[str, Any]) -> bool:
        """Quick check if any button was clicked.
        
        Args:
            payload: Frontend payload
            
        Returns:
            True if button was clicked
        """
        return ButtonController.parse_from_payload(payload) is not None
