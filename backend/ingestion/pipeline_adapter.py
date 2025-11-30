"""Adapter to integrate new pipeline with existing API.

This module provides backward-compatible wrappers that can be used
as drop-in replacements for the legacy ingestion functions.

Usage in resources.py:
    # Replace:
    # chunker_fn = _get_chunker()
    # new_chunks = chunker_fn(local_path)
    
    # With:
    from ingestion.pipeline_adapter import chunker_adapter
    new_chunks = chunker_adapter(local_path, resource_id)
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("backend.ingestion.pipeline_adapter")


def _use_new_pipeline() -> bool:
    """Check if new pipeline should be used."""
    return os.getenv("USE_NEW_INGEST_PIPELINE", "false").lower() in ("true", "1", "yes")


def chunker_adapter(
    file_path: str,
    resource_id: Optional[str] = None,
    mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Adapter that returns chunks in the format expected by reindex_resource.
    
    Args:
        file_path: Path to document file
        resource_id: Resource UUID (optional, for new pipeline)
        mode: Ingestion mode ("fast", "standard", "full")
    
    Returns:
        List of chunk dicts compatible with legacy format
    """
    if not _use_new_pipeline():
        # Use legacy chunker
        from .chunker import enhanced_structural_chunk_resource
        return enhanced_structural_chunk_resource(file_path)
    
    # Use new pipeline
    from .pipeline import IngestPipeline, IngestMode
    
    # Determine mode
    if mode is None:
        mode = os.getenv("INGEST_MODE", "standard")
    
    try:
        ingest_mode = IngestMode(mode.lower())
    except ValueError:
        ingest_mode = IngestMode.STANDARD
    
    # Create pipeline (skip store phase, just parse and extract)
    pipeline = IngestPipeline(
        resource_id=resource_id or "temp",
        file_path=file_path,
        mode=ingest_mode,
    )
    
    # Run parse and extract phases only
    pipeline._phase_parse()
    pipeline._phase_extract()
    
    # Convert to legacy format
    return [_chunk_to_legacy_format(c) for c in pipeline.chunks]


def _chunk_to_legacy_format(chunk) -> Dict[str, Any]:
    """Convert ChunkData to legacy dict format."""
    return {
        "page_number": chunk.page_number,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "source_offset": chunk.source_offset,
        "full_text": chunk.full_text,
        "text_snippet": chunk.text_snippet,
        "section_title": chunk.section_title,
        "section_number": chunk.section_number,
        "section_level": chunk.section_level,
        "section_path": chunk.section_path,
        "token_count": chunk.token_count,
        "has_figure": chunk.has_figure,
        "has_equation": chunk.has_equation,
        "figure_labels": chunk.figure_labels,
        "equation_labels": chunk.equation_labels,
        "caption": None,
        "tags": chunk.to_tags_json(),
        "chunk_type_hint": chunk.chunk_type,
        "pedagogy_role": chunk.pedagogy_role,
        "content_type": chunk.content_type,
        "difficulty": chunk.difficulty,
        "cognitive_level": chunk.cognitive_level,
        "domain": chunk.domain,
        "topic": chunk.topic,
        "subtopic": chunk.subtopic,
        "key_concepts": chunk.concepts,
        "prerequisites": chunk.prerequisites,
        "formulas": chunk.formulas,
    }


def run_full_ingest(
    resource_id: str,
    file_path: str,
    mode: str = "standard",
) -> Dict[str, Any]:
    """Run full ingestion including database storage.
    
    This is a complete replacement for the reindex_resource flow
    when using the new pipeline.
    
    Args:
        resource_id: Resource UUID
        file_path: Path to document file  
        mode: Ingestion mode
    
    Returns:
        Dict with ingestion statistics
    """
    from .pipeline import ingest_resource
    
    result = ingest_resource(resource_id, file_path, mode)
    
    return {
        "resource_id": result.resource_id,
        "inserted": result.chunks_created,
        "updated": result.chunks_updated,
        "deleted": result.chunks_deleted,
        "concepts_created": result.concepts_created,
        "relationships_created": result.relationships_created,
        "elapsed_ms": result.elapsed_ms,
        "errors": result.errors,
        "success": result.success,
    }


def tag_and_extract(text: str, hint: Optional[str] = None) -> Dict[str, Any]:
    """Backward-compatible wrapper for chunk tagging.
    
    Replaces the legacy tag_and_extract function used in reindex_resource.
    
    Args:
        text: Chunk text to analyze
        hint: Optional chunk type hint
    
    Returns:
        Dict with concepts, math_expressions, chunk_type
    """
    if not _use_new_pipeline():
        # Use legacy tagger
        try:
            from prompts import get as prompt_get, render as prompt_render
            from llm import call_llm_json
            
            tmpl = prompt_get("ingest.chunk_tags")
            if tmpl:
                prompt = prompt_render(tmpl, {"chunk_text": text})
                result = call_llm_json(prompt, {"tags": []})
                tags = result.get("tags", [])
            else:
                tags = []
            
            return {
                "chunk_type": hint,
                "concepts": tags[:6],
                "math_expressions": []
            }
        except Exception:
            logger.exception("Legacy tagging failed")
            return {"chunk_type": hint, "concepts": [], "math_expressions": []}
    
    # Use new pipeline's heuristic extraction
    from .pipeline import ChunkData, IngestPipeline
    
    pipeline = IngestPipeline(resource_id="temp", file_path="temp")
    chunk = ChunkData(resource_id="temp", full_text=text, page_number=1)
    pipeline.chunks = [chunk]
    pipeline._extract_heuristic_single(chunk)
    
    return {
        "chunk_type": chunk.pedagogy_role if not hint else hint,
        "concepts": chunk.concepts,
        "math_expressions": []
    }
