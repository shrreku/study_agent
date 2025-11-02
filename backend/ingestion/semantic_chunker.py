"""Semantic-aware chunker with formula preservation and content-type classification.

This module provides advanced chunking that respects semantic boundaries,
preserves mathematical formulas with context, and supports multi-page concepts.
"""

from typing import List, Dict, Tuple, Any, Optional
import os
import re
import logging
from prompts import get as prompt_get, render as prompt_render
from llm import call_llm_json

logger = logging.getLogger("backend.semantic_chunker")


# Formula detection patterns (LaTeX and common math notation)
FORMULA_PATTERNS = [
    # LaTeX environments
    re.compile(r'\\begin\{equation\}.*?\\end\{equation\}', re.DOTALL),
    re.compile(r'\\begin\{align\}.*?\\end\{align\}', re.DOTALL),
    re.compile(r'\\begin\{eqnarray\}.*?\\end\{eqnarray\}', re.DOTALL),
    # Inline LaTeX
    re.compile(r'\$\$.*?\$\$', re.DOTALL),
    re.compile(r'\$[^\$]+\$'),
    # Common equation patterns
    re.compile(r'[a-zA-Z]\s*=\s*[^,\n]{3,}'),  # variable = expression
    re.compile(r'\w+\([^)]+\)\s*=\s*[^,\n]+'),  # function = expression
]


def _env_int(name: str, default: int) -> int:
    """Get integer from environment variable with fallback."""
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    """Get boolean from environment variable with fallback."""
    val = os.getenv(name, "").lower().strip()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return default


def detect_formulas(text: str) -> List[Dict[str, Any]]:
    """Detect formulas in text and return their positions.
    
    Args:
        text: Input text to scan for formulas
        
    Returns:
        List of dicts with 'start', 'end', 'text' keys for each formula
    """
    formulas = []
    for pattern in FORMULA_PATTERNS:
        for match in pattern.finditer(text):
            formulas.append({
                'start': match.start(),
                'end': match.end(),
                'text': match.group(0)
            })
    
    # Sort by start position and merge overlapping formulas
    formulas.sort(key=lambda x: x['start'])
    merged = []
    for formula in formulas:
        if merged and formula['start'] <= merged[-1]['end']:
            # Overlapping or adjacent - extend the previous formula
            merged[-1]['end'] = max(merged[-1]['end'], formula['end'])
            merged[-1]['text'] = text[merged[-1]['start']:merged[-1]['end']]
        else:
            merged.append(formula)
    
    return merged


def formula_aware_split(text: str, max_chunk_size: int = 500) -> List[Dict[str, Any]]:
    """Split text preserving formulas with context.
    
    Never splits in the middle of a formula. Ensures each formula has
    surrounding context (at least one sentence before and after when possible).
    
    Args:
        text: Input text to split
        max_chunk_size: Maximum chunk size in tokens (approximate)
        
    Returns:
        List of chunks with 'text', 'start', 'end', 'has_formula' keys
    """
    formulas = detect_formulas(text)
    
    if not formulas:
        # No formulas - can split normally
        return _simple_split(text, max_chunk_size)
    
    chunks = []
    current_pos = 0
    
    for formula in formulas:
        # Add context before formula (at least one sentence)
        context_start = max(0, formula['start'] - 200)  # ~200 chars context
        # Find sentence boundary
        before_text = text[context_start:formula['start']]
        sentence_match = re.search(r'[.!?]\s+', before_text)
        if sentence_match:
            context_start = context_start + sentence_match.end()
        
        # Add context after formula (at least one sentence)
        context_end = min(len(text), formula['end'] + 200)
        after_text = text[formula['end']:context_end]
        sentence_match = re.search(r'[.!?]\s+', after_text)
        if sentence_match:
            context_end = formula['end'] + sentence_match.end()
        
        # Create chunk with formula and context
        chunk_text = text[context_start:context_end]
        chunk_tokens = len(chunk_text.split())
        
        # If too large, try to reduce context while keeping formula intact
        if chunk_tokens > max_chunk_size:
            # Keep formula + minimal context (one sentence each side)
            chunk_text = text[max(0, formula['start'] - 100):min(len(text), formula['end'] + 100)]
        
        chunks.append({
            'text': chunk_text.strip(),
            'start': context_start,
            'end': context_end,
            'has_formula': True,
            'formula_count': 1
        })
        
        current_pos = context_end
    
    # Handle remaining text after last formula
    if current_pos < len(text):
        remaining = text[current_pos:].strip()
        if remaining:
            chunks.append({
                'text': remaining,
                'start': current_pos,
                'end': len(text),
                'has_formula': False,
                'formula_count': 0
            })
    
    return chunks


def _simple_split(text: str, max_tokens: int) -> List[Dict[str, Any]]:
    """Simple sentence-based split for text without formulas."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', text) if s.strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    current_start = 0
    
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # Flush current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start': current_start,
                'end': current_start + len(chunk_text),
                'has_formula': False,
                'formula_count': 0
            })
            current_chunk = [sentence]
            current_tokens = sentence_tokens
            current_start += len(chunk_text)
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Flush remaining
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'start': current_start,
            'end': current_start + len(chunk_text),
            'has_formula': False,
            'formula_count': 0
        })
    
    return chunks


def identify_semantic_units(text: str, page_number: int) -> Dict[str, Any]:
    """Use LLM to identify semantic boundaries and content types.
    
    Args:
        text: Page text to analyze
        page_number: Page number for context
        
    Returns:
        Dict with 'sections' and 'chunks' containing semantic units
    """
    # Trim text if too long
    max_chars = _env_int("INGEST_PAGE_MAX_CHARS", 6000)
    if len(text) > max_chars > 0:
        text = text[:max_chars]
    
    tmpl = prompt_get("ingest.page_structure_v2")
    if not tmpl:
        # Fallback to legacy prompt
        tmpl = prompt_get("ingest.page_structure")
    
    prompt = prompt_render(tmpl, {
        "page_number": page_number,
        "page_text": text,
    })
    
    default = {"sections": [], "chunks": []}
    
    try:
        result = call_llm_json(prompt, default)
        return result or default
    except Exception as e:
        logger.exception("semantic_unit_identification_failed", extra={"page": page_number})
        return default


def get_chunk_size_limits(content_type: str) -> Tuple[int, int]:
    """Get min/max token limits for a given content type.
    
    Args:
        content_type: Type of content (concept_intro, derivation, example, etc.)
        
    Returns:
        Tuple of (min_tokens, max_tokens)
    """
    # Environment variable overrides
    if content_type == "concept_intro":
        min_tokens = _env_int("SEMANTIC_CHUNK_MIN_TOKENS_CONCEPT", 100)
        max_tokens = _env_int("SEMANTIC_CHUNK_MAX_TOKENS_CONCEPT", 300)
    elif content_type == "derivation":
        min_tokens = _env_int("SEMANTIC_CHUNK_MIN_TOKENS_DERIVATION", 300)
        max_tokens = _env_int("SEMANTIC_CHUNK_MAX_TOKENS_DERIVATION", 500)
    elif content_type == "example":
        min_tokens = _env_int("SEMANTIC_CHUNK_MIN_TOKENS_EXAMPLE", 150)
        max_tokens = _env_int("SEMANTIC_CHUNK_MAX_TOKENS_EXAMPLE", 400)
    elif content_type == "summary":
        min_tokens = _env_int("SEMANTIC_CHUNK_MIN_TOKENS_SUMMARY", 50)
        max_tokens = _env_int("SEMANTIC_CHUNK_MAX_TOKENS_SUMMARY", 150)
    else:
        # Default for unknown types
        min_tokens = _env_int("SEMANTIC_CHUNK_MIN_TOKENS_DEFAULT", 100)
        max_tokens = _env_int("SEMANTIC_CHUNK_MAX_TOKENS_DEFAULT", 300)
    
    return (min_tokens, max_tokens)


def create_semantic_chunks(pages: List[str], 
                          content_aware: bool = True,
                          preserve_formulas: bool = True) -> List[Dict[str, Any]]:
    """Create semantic chunks from multiple pages with content-aware sizing.
    
    Args:
        pages: List of page texts
        content_aware: Use content-type aware chunk sizing
        preserve_formulas: Preserve formulas with context
        
    Returns:
        List of chunk dictionaries with metadata
    """
    # Check feature flags
    if not _env_bool("SEMANTIC_CHUNK_CONTENT_AWARE", content_aware):
        content_aware = False
    
    all_chunks = []
    
    # Track multi-page concepts
    previous_semantic_units = None
    
    for page_idx, page_text in enumerate(pages, start=1):
        if not page_text or not page_text.strip():
            continue
        
        # Identify semantic units using LLM
        structure = identify_semantic_units(page_text, page_idx)
        sections = structure.get("sections", [])
        semantic_units = structure.get("chunks", [])

        # Fallback: if no semantic units returned, chunk the whole page
        if not semantic_units:
            # Use default size limits for unknown content
            _min_tok, _max_tok = get_chunk_size_limits("unknown")
            if preserve_formulas and detect_formulas(page_text):
                sub_chunks = formula_aware_split(page_text, _max_tok)
            else:
                sub_chunks = _simple_split(page_text, _max_tok)

            for sub_chunk in sub_chunks:
                chunk_data = {
                    'page_number': page_idx,
                    'page_start': page_idx,
                    'page_end': page_idx,
                    'source_offset': sub_chunk.get('start', 0),
                    'full_text': sub_chunk.get('text', '').strip(),
                    'section_title': '',
                    'section_number': '',
                    'section_level': None,
                    'token_count': len((sub_chunk.get('text') or '').split()),
                    'has_figure': False,
                    'has_equation': bool(sub_chunk.get('has_formula', False)),
                    'figure_labels': [],
                    'equation_labels': [],
                    'caption': None,
                    'tags': [],
                    'content_type': 'unknown',
                    'cognitive_level': None,
                    'difficulty': None,
                    'text_snippet': (sub_chunk.get('text') or '')[:300],
                }
                if chunk_data['full_text']:
                    all_chunks.append(chunk_data)
            logger.info("Applied fallback chunking", extra={"page": page_idx, "chunks": len(sub_chunks)})
            continue

        # Process each semantic unit
        for unit in semantic_units:
            try:
                start = int(unit.get("start", 0))
                end = int(unit.get("end", 0))
                content_type = unit.get("type", "unknown")
            except (ValueError, TypeError):
                logger.warning("Invalid semantic unit boundaries", extra={"page": page_idx})
                continue
            
            if end <= start or start < 0:
                continue
            
            # Extract text
            unit_text = page_text[start:end].strip()
            if len(unit_text) < 20:
                continue
            
            # Get size limits for this content type
            min_tokens, max_tokens = get_chunk_size_limits(content_type)
            
            # Check if we need to split this unit
            unit_tokens = len(unit_text.split())
            
            if preserve_formulas and detect_formulas(unit_text):
                # Formula-aware splitting
                sub_chunks = formula_aware_split(unit_text, max_tokens)
            elif unit_tokens > max_tokens:
                # Simple split if too large
                sub_chunks = _simple_split(unit_text, max_tokens)
            else:
                # Keep as single chunk
                sub_chunks = [{
                    'text': unit_text,
                    'start': start,
                    'end': end,
                    'has_formula': bool(detect_formulas(unit_text)),
                    'formula_count': len(detect_formulas(unit_text))
                }]
            
            # Create chunk metadata
            for sub_chunk in sub_chunks:
                chunk_data = {
                    'page_number': page_idx,
                    'page_start': page_idx,
                    'page_end': page_idx,
                    'source_offset': start + sub_chunk.get('start', 0),
                    'full_text': sub_chunk['text'],
                    'section_title': unit.get('section_title', ''),
                    'section_number': unit.get('section_number', ''),
                    'section_level': unit.get('section_level'),
                    'token_count': len(sub_chunk['text'].split()),
                    'has_figure': unit.get('has_figure', False),
                    'has_equation': sub_chunk.get('has_formula', False),
                    'figure_labels': unit.get('figure_labels', []),
                    'equation_labels': unit.get('equation_labels', []),
                    'caption': unit.get('caption'),
                    'tags': unit.get('tags', []),
                    'content_type': content_type,
                    'cognitive_level': unit.get('cognitive_level'),
                    'difficulty': unit.get('difficulty'),
                    'text_snippet': sub_chunk['text'][:300],
                }
                
                all_chunks.append(chunk_data)
        
        previous_semantic_units = semantic_units
    
    # Post-processing: merge multi-page concepts if they're semantically continuous
    # (This is a placeholder for future enhancement)
    
    logger.info(f"Created {len(all_chunks)} semantic chunks from {len(pages)} pages")
    
    return all_chunks


def extract_hierarchical_tags(chunk_text: str) -> Dict[str, Any]:
    """Extract hierarchical educational metadata for a chunk.
    
    Args:
        chunk_text: Text of the chunk to analyze
        
    Returns:
        Dict with domain, topic, subtopic, prerequisites, learning objectives, etc.
    """
    default = {
        "domain": "",
        "topic": "",
        "subtopic": "",
        "prerequisites": [],
        "learning_objectives": [],
        "content_type": "unknown",
        "difficulty": "intermediate",
        "cognitive_level": "understand",
        "key_concepts": []
    }
    
    tmpl = prompt_get("ingest.chunk_tags_hierarchical")
    if not tmpl:
        # Fallback - return default structure
        return default
    
    prompt = prompt_render(tmpl, {"chunk_text": chunk_text})
    
    try:
        result = call_llm_json(prompt, default)
        return result or default
    except Exception as e:
        logger.exception("hierarchical_tagging_failed")
        return default


def extract_formula_metadata(text: str) -> List[Dict[str, Any]]:
    """Extract complete formula metadata from text.
    
    Args:
        text: Text containing formulas
        
    Returns:
        List of formula metadata dicts
    """
    # First check if there are any formulas
    if not detect_formulas(text):
        return []
    
    tmpl = prompt_get("ingest.formula_extraction")
    if not tmpl:
        return []
    
    prompt = prompt_render(tmpl, {"text": text})
    
    default = {"formulas": []}
    
    try:
        result = call_llm_json(prompt, default)
        return result.get("formulas", []) if result else []
    except Exception as e:
        logger.exception("formula_extraction_failed")
        return []
