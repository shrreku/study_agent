from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import logging
from collections import OrderedDict

from llm import call_llm_json
from prompts import get as prompt_get, render as prompt_render

logger = logging.getLogger(__name__)


@dataclass
class ExampleRequest:
    concept: str
    difficulty: str
    context_type: str
    student_background: Optional[str] = None
    prerequisites_mastered: Optional[List[str]] = None
    avoid_patterns: Optional[List[str]] = None


@dataclass
class GeneratedExample:
    example_text: str
    explanation: str
    relevance_score: float
    difficulty: str
    context_type: str
    confidence: float


class ExampleGenerator:
    def __init__(self) -> None:
        try:
            self.min_relevance = float(os.getenv("TUTOR_EXAMPLE_MIN_RELEVANCE", "0.6") or 0.6)
        except Exception:
            self.min_relevance = 0.6
        try:
            self.min_confidence = float(os.getenv("TUTOR_EXAMPLE_MIN_CONFIDENCE", "0.5") or 0.5)
        except Exception:
            self.min_confidence = 0.5
        try:
            self.cache_size = int(os.getenv("TUTOR_EXAMPLE_CACHE_SIZE", "100") or 100)
        except Exception:
            self.cache_size = 100
        self._cache: "OrderedDict[str, GeneratedExample]" = OrderedDict()

    def _cache_key(self, request: ExampleRequest) -> str:
        parts = [
            (request.concept or "").strip().lower(),
            (request.difficulty or "").strip().lower(),
            (request.context_type or "").strip().lower(),
            (request.student_background or "").strip().lower(),
            ",".join(request.prerequisites_mastered or []),
            ",".join(request.avoid_patterns or []),
        ]
        return "|".join(parts)

    def _get_from_cache(self, key: str) -> Optional[GeneratedExample]:
        if key in self._cache:
            val = self._cache[key]
            self._cache.move_to_end(key)
            return val
        return None

    def _put_cache(self, key: str, value: GeneratedExample) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def generate_example(
        self,
        request: ExampleRequest,
        grounding_chunks: Optional[List[Dict]] = None,
    ) -> GeneratedExample:
        key = self._cache_key(request)
        cached = self._get_from_cache(key)
        if cached:
            return cached

        template = prompt_get("tutor.generate_example")

        chunk_context = ""
        if grounding_chunks:
            items: List[str] = []
            for chunk in grounding_chunks[:3]:
                snippet = str(chunk.get("snippet") or "")[:200]
                if snippet:
                    items.append(f"- {snippet}")
            chunk_context = "\n".join(items)

        prompt = prompt_render(
            template,
            {
                "concept": request.concept,
                "difficulty": request.difficulty,
                "context_type": request.context_type,
                "student_background": request.student_background or "general",
                "prerequisites": ", ".join(request.prerequisites_mastered or []),
                "course_examples": chunk_context or "(none provided)",
                "avoid_patterns": ", ".join(request.avoid_patterns or []),
            },
        )

        default_response = {
            "example": f"Consider how {request.concept} appears in everyday situations.",
            "explanation": "This connects to the concept by highlighting the core relation.",
            "relevance": 0.6,
            "confidence": 0.5,
        }

        try:
            result = call_llm_json(prompt, default=default_response)
        except Exception:
            logger.exception("example_generation_failed")
            result = default_response

        example_text = str(result.get("example") or default_response["example"]) 
        explanation = str(result.get("explanation") or default_response["explanation"]) 
        try:
            relevance = float(result.get("relevance", default_response["relevance"]))
        except Exception:
            relevance = float(default_response["relevance"])
        try:
            confidence = float(result.get("confidence", default_response["confidence"]))
        except Exception:
            confidence = float(default_response["confidence"])

        gen = GeneratedExample(
            example_text=example_text,
            explanation=explanation,
            relevance_score=relevance,
            difficulty=request.difficulty,
            context_type=request.context_type,
            confidence=confidence,
        )
        self._put_cache(key, gen)
        return gen

    def generate_bridge_example(
        self,
        from_concept: str,
        to_concept: str,
        student_level: str,
        grounding_chunks: Optional[List[Dict]] = None,
    ) -> GeneratedExample:
        template = prompt_get("tutor.bridge_example")

        chunk_context = ""
        if grounding_chunks:
            items: List[str] = []
            for chunk in grounding_chunks[:3]:
                snippet = str(chunk.get("snippet") or "")[:200]
                if snippet:
                    items.append(f"- {snippet}")
            chunk_context = "\n".join(items)

        prompt = prompt_render(
            template,
            {
                "from_concept": from_concept,
                "to_concept": to_concept,
                "student_level": student_level,
                "course_examples": chunk_context or "(none provided)",
            },
        )

        default_response = {
            "example": f"Think of {to_concept} as an extension of {from_concept}.",
            "explanation": "This builds on the known idea to introduce the new one.",
            "relevance": 0.6,
            "confidence": 0.5,
        }

        try:
            result = call_llm_json(prompt, default=default_response)
        except Exception:
            logger.exception("bridge_example_generation_failed")
            result = default_response

        example_text = str(result.get("example") or default_response["example"]) 
        explanation = str(result.get("explanation") or default_response["explanation"]) 
        try:
            relevance = float(result.get("relevance", default_response["relevance"]))
        except Exception:
            relevance = float(default_response["relevance"])
        try:
            confidence = float(result.get("confidence", default_response["confidence"]))
        except Exception:
            confidence = float(default_response["confidence"]) 

        return GeneratedExample(
            example_text=example_text,
            explanation=explanation,
            relevance_score=relevance,
            difficulty=student_level,
            context_type="bridge",
            confidence=confidence,
        )
