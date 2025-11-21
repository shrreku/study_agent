"""
Response Generator: Clean, grounded response generation with structured citations.

This module replaces raw string concatenation with:
1. LLM-generated pure response text (no chunk concatenation)
2. Separately built citations with cleaned snippets
3. Support for two grounding modes: "llm_integrated" and "explicit_citation"

Phase 3 of Unified Architecture V2: eliminating OCR artifacts and improving response quality.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from prompts import get as prompt_get, render as prompt_render
from llm import call_json_chat
from constants import logger

from .actions.types import Citation, TutorResponse
from .runtime.utils import safe_float
from .config import TutorConfig

# Re-export for backwards compatibility
from .responses import clean_snippet_for_display, _fallback_response_text


@dataclass
class ResponseContext:
    """Context for response generation."""
    concept: Optional[str]
    level: str
    chunks: List[Dict[str, Any]]
    message: Optional[str] = None
    plan_thinking: Optional[str] = None
    plan_rationale: Optional[str] = None
    pedagogy_focus: Optional[List[str]] = None


class ResponseGenerator:
    """Generate clean, grounded tutor responses with structured citations.
    
    Key principles:
    - NO string concatenation of raw chunks
    - LLM generates pure response text or question
    - Citations built separately from response
    - Support for explicit_citation mode (pure text + separate citations)
    - Support for llm_integrated mode (LLM incorporates context naturally)
    
    Can be instantiated with explicit config or will use system config.
    """
    
    def __init__(
        self,
        default_grounding_mode: Optional[str] = None,
        citation_limit: Optional[int] = None,
        config: Optional[TutorConfig] = None,
    ):
        """Initialize the response generator.
        
        Args:
            default_grounding_mode: Override grounding mode ("llm_integrated" or "explicit_citation")
            citation_limit: Override max citations to include
            config: TutorConfig instance (will use system config if None)
        """
        self.config = config or self._get_default_config()
        
        # Use provided values or fall back to config
        self.default_grounding_mode = (
            default_grounding_mode or self.config.response_grounding_mode
        )
        self.citation_limit = citation_limit or self.config.citation_limit
    
    @staticmethod
    def _get_default_config() -> TutorConfig:
        """Get the system default configuration."""
        from .config import get_tutor_config
        return get_tutor_config()
    
    # ===== Citation Building =====
    
    def _build_citations(
        self,
        chunks: List[Dict[str, Any]],
        limit: Optional[int] = None,
    ) -> List[Citation]:
        """Build clean citation list from chunks.
        
        Args:
            chunks: Raw chunk dicts from retrieval
            limit: Override default citation limit
            
        Returns:
            List of Citation objects with cleaned snippets
        """
        limit = limit or self.citation_limit
        citations = []
        
        for chunk in chunks[:limit]:
            chunk_id = chunk.get("id") or chunk.get("chunk_id") or ""
            snippet_raw = (chunk.get("snippet") or "").strip()
            
            if not snippet_raw or not chunk_id:
                continue
            
            citations.append(Citation(
                chunk_id=chunk_id,
                snippet_text=clean_snippet_for_display(snippet_raw),
                pedagogy_role=chunk.get("pedagogy_role"),
                relevance_score=safe_float(chunk.get("score") or chunk.get("relevance_score")),
            ))
        
        return citations
    
    def _build_source_ids(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract source chunk IDs from chunks."""
        return [
            c.get("id") or c.get("chunk_id")
            for c in chunks
            if (c.get("id") or c.get("chunk_id"))
        ]
    
    # ===== Response Generation Methods =====
    
    def generate_explain_response(
        self,
        context: ResponseContext,
        grounding_mode: Optional[str] = None,
    ) -> TutorResponse:
        """Generate an explanation response.
        
        Args:
            context: ResponseContext with concept, level, chunks, message
            grounding_mode: Override default grounding mode
            
        Returns:
            TutorResponse with clean text and citations
        """
        grounding_mode = grounding_mode or self.default_grounding_mode
        
        # No chunks case
        if not context.chunks:
            return TutorResponse(
                response_text=(
                    "I couldn't find grounded material yet. "
                    "Let's review your materials first. Could you point me to the section?"
                ),
                action_type="explain",
                confidence=0.2,
                citations=[],
                grounding_mode=grounding_mode,
                inference_concept=context.concept,
            )
        
        # Build clean citations first
        citations = self._build_citations(context.chunks)
        source_ids = self._build_source_ids(context.chunks)
        
        # Format context for LLM
        if grounding_mode == "explicit_citation":
            # Pure explanation, LLM incorporates context naturally
            context_block = ""
        else:
            # LLM can reference context
            from .runtime.utils import format_context_snippets
            context_block = format_context_snippets(context.chunks)
        
        # Call LLM for explanation
        prompt = prompt_render(
            prompt_get("tutor.explain"),
            {
                "concept": context.concept or "the concept",
                "level": context.level,
                "context": context_block,
                "student_message": context.message or "",
            },
        )
        
        default_payload = {
            "response": _fallback_response_text(context.concept, context.chunks),
            "confidence": 0.5,
        }
        
        try:
            result = call_json_chat(
                prompt,
                default=default_payload,
                allow_text_fallback=True,
                text_field="response",
            )
        except Exception:
            logger.exception("response_gen_explain_failed")
            result = default_payload
        
        response_text = str(result.get("response") or default_payload["response"]).strip()
        if not response_text:
            response_text = default_payload["response"]
        confidence = safe_float(result.get("confidence")) or 0.5
        
        return TutorResponse(
            response_text=response_text,
            action_type="explain",
            confidence=confidence,
            citations=citations,
            grounding_mode=grounding_mode,
            inference_concept=context.concept,
            source_chunk_ids=source_ids,
        )
    
    def generate_ask_response(
        self,
        context: ResponseContext,
        grounding_mode: Optional[str] = None,
    ) -> TutorResponse:
        """Generate an assessment question.
        
        Args:
            context: ResponseContext with concept, level, chunks, message
            grounding_mode: Override default grounding mode
            
        Returns:
            TutorResponse with clean question text and citations
        """
        grounding_mode = grounding_mode or self.default_grounding_mode
        citations = self._build_citations(context.chunks)
        source_ids = self._build_source_ids(context.chunks)
        
        # Determine default question
        if context.concept:
            default_question = f"Can you explain {context.concept} in your own words?"
        else:
            default_question = "Can you summarize what you understand so far?"
        
        # Format context for LLM
        if grounding_mode == "explicit_citation":
            context_block = ""
        else:
            from .runtime.utils import format_context_snippets
            context_block = format_context_snippets(context.chunks)
        
        # Call LLM for question
        prompt = prompt_render(
            prompt_get("tutor.ask"),
            {
                "concept": context.concept or "the concept",
                "level": context.level,
                "context": context_block,
                "student_message": context.message or "",
            },
        )
        
        default_payload = {
            "question": default_question,
            "confidence": 0.7,
        }
        
        try:
            result = call_json_chat(prompt, default=default_payload)
        except Exception:
            logger.exception("response_gen_ask_failed")
            result = default_payload
        
        response_text = str(result.get("question") or default_question).strip()
        if not response_text:
            response_text = default_question
        confidence = safe_float(result.get("confidence")) or 0.7
        
        return TutorResponse(
            response_text=response_text,
            action_type="ask",
            confidence=confidence,
            citations=citations,
            grounding_mode=grounding_mode,
            inference_concept=context.concept,
            source_chunk_ids=source_ids,
        )
    
    def generate_hint_response(
        self,
        context: ResponseContext,
        grounding_mode: Optional[str] = None,
    ) -> TutorResponse:
        """Generate a hint response.
        
        Args:
            context: ResponseContext with concept, level, chunks, message
            grounding_mode: Override default grounding mode
            
        Returns:
            TutorResponse with hint text and citations
        """
        grounding_mode = grounding_mode or self.default_grounding_mode
        citations = self._build_citations(context.chunks)
        source_ids = self._build_source_ids(context.chunks)
        
        # Format context for LLM
        if grounding_mode == "explicit_citation":
            context_block = ""
        else:
            from .runtime.utils import format_context_snippets
            context_block = format_context_snippets(context.chunks)
        
        # Call LLM for hint
        prompt = prompt_render(
            prompt_get("tutor.hint"),
            {
                "concept": context.concept or "the concept",
                "level": context.level,
                "context": context_block,
                "student_message": context.message or "",
            },
        )
        
        default_payload = {
            "response": _fallback_response_text(context.concept, context.chunks),
            "confidence": 0.5,
        }
        
        try:
            result = call_json_chat(prompt, default=default_payload)
        except Exception:
            logger.exception("response_gen_hint_failed")
            result = default_payload
        
        response_text = str(result.get("response") or default_payload["response"]).strip()
        if not response_text:
            response_text = default_payload["response"]
        confidence = safe_float(result.get("confidence")) or 0.5
        
        return TutorResponse(
            response_text=response_text,
            action_type="hint",
            confidence=confidence,
            citations=citations,
            grounding_mode=grounding_mode,
            inference_concept=context.concept,
            source_chunk_ids=source_ids,
        )
    
    def generate_reflect_response(
        self,
        context: ResponseContext,
        grounding_mode: Optional[str] = None,
    ) -> TutorResponse:
        """Generate a reflection/assessment response.
        
        Args:
            context: ResponseContext with concept, level, chunks, message
            grounding_mode: Override default grounding mode
            
        Returns:
            TutorResponse with reflection prompt and citations
        """
        grounding_mode = grounding_mode or self.default_grounding_mode
        citations = self._build_citations(context.chunks)
        source_ids = self._build_source_ids(context.chunks)
        
        # Format context for LLM
        if grounding_mode == "explicit_citation":
            context_block = ""
        else:
            from .runtime.utils import format_context_snippets
            context_block = format_context_snippets(context.chunks)
        
        # Call LLM for reflection prompt
        prompt = prompt_render(
            prompt_get("tutor.reflect"),
            {
                "concept": context.concept or "the concept",
                "level": context.level,
                "context": context_block,
                "student_message": context.message or "",
            },
        )
        
        default_payload = {
            "response": "Could you summarize what you learned just now?",
            "confidence": 0.6,
        }
        
        try:
            result = call_json_chat(prompt, default=default_payload)
        except Exception:
            logger.exception("response_gen_reflect_failed")
            result = default_payload
        
        response_text = str(result.get("response") or default_payload["response"]).strip()
        if not response_text:
            response_text = default_payload["response"]
        confidence = safe_float(result.get("confidence")) or 0.6
        
        return TutorResponse(
            response_text=response_text,
            action_type="reflect",
            confidence=confidence,
            citations=citations,
            grounding_mode=grounding_mode,
            inference_concept=context.concept,
            source_chunk_ids=source_ids,
        )
    
    def generate_review_response(
        self,
        context: ResponseContext,
        grounding_mode: Optional[str] = None,
    ) -> TutorResponse:
        """Generate a review/summary response.
        
        Args:
            context: ResponseContext with concept, level, chunks, message
            grounding_mode: Override default grounding mode
            
        Returns:
            TutorResponse with review summary and citations
        """
        grounding_mode = grounding_mode or self.default_grounding_mode
        citations = self._build_citations(context.chunks)
        source_ids = self._build_source_ids(context.chunks)
        
        # Build deterministic review
        import textwrap
        concept_label = context.concept or "our topic"
        bullet_items: List[str] = []
        
        for chunk in context.chunks[:3]:
            snippet = (chunk.get("snippet") or "").strip()
            if snippet:
                # Clean snippet
                cleaned = clean_snippet_for_display(snippet, max_length=160)
                if cleaned:
                    bullet_items.append(cleaned)
        
        if not bullet_items:
            bullet_items = [
                f"Revisit the core definition of {concept_label}.",
                "Note the key relationships and examples discussed."
            ]
        
        intro = f"Quick review for {concept_label} ({context.level} level):"
        bullets = "\n".join(
            f"- {textwrap.shorten(item, width=140, placeholder='…')}"
            for item in bullet_items
        )
        response_text = f"{intro}\n{bullets}"
        
        return TutorResponse(
            response_text=response_text,
            action_type="review",
            confidence=0.6 if context.chunks else 0.5,
            citations=citations,
            grounding_mode=grounding_mode,
            inference_concept=context.concept,
            source_chunk_ids=source_ids,
        )
    
    def generate_prerequisite_review_response(
        self,
        target_concept: Optional[str],
        missing_prereqs: List[str],
        chunks: List[Dict[str, Any]],
        grounding_mode: Optional[str] = None,
    ) -> TutorResponse:
        """Generate a prerequisite review response.
        
        Args:
            target_concept: The concept we're trying to teach
            missing_prereqs: List of prerequisite concepts not yet mastered
            chunks: Retrieved chunks about the prerequisites
            grounding_mode: Override default grounding mode
            
        Returns:
            TutorResponse with prerequisite review
        """
        if not missing_prereqs:
            return TutorResponse(
                response_text="Great! You're ready to continue.",
                action_type="explain",
                confidence=0.8,
            )
        
        grounding_mode = grounding_mode or self.default_grounding_mode
        citations = self._build_citations(chunks)
        source_ids = self._build_source_ids(chunks)
        
        first_prereq = missing_prereqs[0]
        prereq_names = ", ".join(missing_prereqs[:2])
        
        # Format context for LLM
        if grounding_mode == "explicit_citation":
            context_block = ""
        else:
            from .runtime.utils import format_context_snippets
            context_block = format_context_snippets(chunks)
        
        # Call LLM for prerequisite review
        prompt = prompt_render(
            prompt_get("tutor.prereq_review"),
            {
                "target_concept": target_concept or "the target concept",
                "prereq_names": prereq_names,
                "first_prereq": first_prereq,
                "context": context_block,
            },
        )
        
        default_payload = {
            "response": (
                f"Great question about {target_concept or 'the topic'}! "
                f"Before we dive in, let's quickly review {first_prereq} using your materials."
            ),
            "confidence": 0.6,
        }
        
        try:
            result = call_json_chat(prompt, default=default_payload)
        except Exception:
            logger.exception("response_gen_prereq_review_failed")
            result = default_payload
        
        response_text = str(result.get("response") or default_payload["response"]).strip()
        if not response_text:
            response_text = default_payload["response"]
        confidence = safe_float(result.get("confidence")) or 0.6
        
        return TutorResponse(
            response_text=response_text,
            action_type="explain",
            confidence=confidence,
            citations=citations,
            grounding_mode=grounding_mode,
            inference_concept=target_concept,
            source_chunk_ids=source_ids,
        )
    
    def generate_orientation_response(
        self,
        message: str,
        focus_concept: Optional[str],
        learning_targets: List[str],
        learning_path: List[str],
        mastery_map: Dict[str, Dict[str, Any]],
        chunks: List[Dict[str, Any]],
    ) -> TutorResponse:
        """Generate an orientation response for session start.
        
        Args:
            message: Student's initial message
            focus_concept: Current focus concept
            learning_targets: Target concepts for session
            learning_path: Suggested learning path
            mastery_map: Student's current mastery
            chunks: Overview chunks
            
        Returns:
            TutorResponse with orientation and recommended concept
        """
        from .runtime.utils import format_concept_list, format_context_snippets
        
        def _format_mastery_snapshot(mastery: Dict[str, Dict[str, Any]], limit: int = 5) -> str:
            if not mastery:
                return "No mastery data yet."
            lines: List[str] = []
            for concept, data in list(mastery.items())[:limit]:
                try:
                    m = float((data or {}).get("mastery", 0.0) or 0.0)
                except Exception:
                    m = 0.0
                lines.append(f"- {concept}: {m:.2f}")
            return "\n".join(lines)
        
        orientation_prompt = prompt_render(
            prompt_get("tutor.orient"),
            {
                "student_message": message,
                "focus_concept": focus_concept or "",
                "target_concepts": format_concept_list(learning_targets),
                "learning_path": ", ".join(learning_path or []),
                "mastery_snapshot": _format_mastery_snapshot(mastery_map),
                "overview_snippets": format_context_snippets(chunks),
            },
        )
        
        default_recommended = (
            focus_concept
            or (learning_path[0] if learning_path else None)
            or (learning_targets[0] if learning_targets else "")
        )
        default_payload = {
            "response": (
                "Let's plan our study session. "
                "Based on your course materials, we can work through a short sequence of key topics. "
                "I'll suggest a starting point and a couple of next steps, then you can choose where to begin."
            ),
            "recommended_concept": default_recommended,
            "confidence": 0.7,
        }
        
        try:
            result = call_json_chat(orientation_prompt, default=default_payload)
        except Exception:
            logger.exception("response_gen_orientation_failed")
            result = default_payload
        
        response_text = str(result.get("response") or default_payload["response"]).strip()
        recommended_concept = str(result.get("recommended_concept") or default_recommended).strip() or None
        confidence = safe_float(result.get("confidence")) or 0.7
        
        citations = self._build_citations(chunks)
        source_ids = self._build_source_ids(chunks)
        
        return TutorResponse(
            response_text=response_text,
            action_type="orient",
            confidence=confidence,
            citations=citations,
            grounding_mode="llm_integrated",
            inference_concept=recommended_concept,
            source_chunk_ids=source_ids,
        )
    
    def generate_cold_start_question(
        self,
        concept: Optional[str],
        chunks: List[Dict[str, Any]],
        message: Optional[str] = None,
    ) -> TutorResponse:
        """Generate a cold-start assessment question.
        
        Args:
            concept: The concept to ask about
            chunks: Retrieved background chunks
            message: Student's previous message
            
        Returns:
            TutorResponse with cold-start question
        """
        default_question = f"What is a key idea about {concept}?" if concept else "What is a key idea here?"
        citations = self._build_citations(chunks)
        source_ids = self._build_source_ids(chunks)
        
        cold_prompt = prompt_render(
            prompt_get("tutor.ask"),
            {
                "concept": concept or "the concept",
                "level": "beginner",
                "context": "",
                "student_message": message or "",
            },
        )
        
        ask_default = {
            "question": default_question,
            "answer": "",
            "confidence": 0.4,
            "options": [],
        }
        
        try:
            ask_result = call_json_chat(cold_prompt, default=ask_default)
        except Exception:
            logger.exception("response_gen_cold_start_failed")
            ask_result = ask_default
        
        response_text = str(ask_result.get("question") or default_question).strip()
        confidence = safe_float(ask_result.get("confidence")) or 0.4
        
        return TutorResponse(
            response_text=response_text,
            action_type="ask",
            confidence=confidence,
            citations=citations,
            grounding_mode="llm_integrated",
            inference_concept=concept,
            source_chunk_ids=source_ids,
        )

