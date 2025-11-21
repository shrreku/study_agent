from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Citation:
    """Structured citation for source material with cleaned text."""
    chunk_id: str
    snippet_text: str  # Cleaned for display (no OCR artifacts)
    pedagogy_role: Optional[str] = None  # e.g., "definition", "example", "explanation"
    relevance_score: Optional[float] = None  # Similarity score 0-1


@dataclass
class TutorResponse:
    """Complete tutor response with clean separation of text and citations."""
    response_text: str  # Pure LLM-generated text, no concatenation
    action_type: str  # "explain", "ask", "hint", "reflect", "review", etc.
    confidence: float  # 0.0 - 1.0
    citations: List[Citation] = field(default_factory=list)
    grounding_mode: str = "llm_integrated"  # "llm_integrated" or "explicit_citation"
    inference_concept: Optional[str] = None
    source_chunk_ids: List[str] = field(default_factory=list)  # For backwards compatibility


@dataclass
class ActionResult:
    """Result of an action handler containing all response details."""
    action_type: str
    response_text: str
    confidence: float
    source_chunk_ids: List[str]
    inference_concept: Optional[str]
    action_params: Dict[str, Any]
    cold_start_triggered: bool = False
    mcq: Optional[Dict[str, Any]] = None
