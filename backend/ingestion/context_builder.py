"""Extended Context Windows and Figure Metadata Extraction.

This module provides enhanced context for chunks including:
- Previous/next chunk summaries for better semantic search
- Enhanced figure and table metadata
- Complexity metrics (Bloom's taxonomy, math level, study time)
- Cross-reference extraction (equations, figures, sections)
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import re
import logging

logger = logging.getLogger("backend.context_builder")

# Cross-reference patterns
EQUATION_REF_PATTERNS = [
    re.compile(r'\b(?:Eq|Equation)\.?\s*(\d+\.?\d*)', re.IGNORECASE),
    re.compile(r'\((\d+\.?\d*)\)'),  # (5.8)
    re.compile(r'\[(\d+\.?\d*)\]'),  # [5.8]
]

FIGURE_REF_PATTERNS = [
    re.compile(r'\b(?:Fig|Figure)\.?\s*(\d+\.?\d*)', re.IGNORECASE),
]

TABLE_REF_PATTERNS = [
    re.compile(r'\b(?:Tab|Table)\.?\s*(\d+\.?\d*)', re.IGNORECASE),
]

SECTION_REF_PATTERNS = [
    re.compile(r'\b(?:Sec|Section)\.?\s*(\d+\.?\d*)', re.IGNORECASE),
    re.compile(r'\bChapter\s+(\d+)', re.IGNORECASE),
]

# Figure type indicators
FIGURE_TYPE_KEYWORDS = {
    'graph': ['plot', 'curve', 'versus', 'vs', 'axis', 'axes'],
    'diagram': ['schematic', 'diagram', 'illustration', 'flowchart'],
    'photograph': ['photo', 'photograph', 'image', 'picture'],
    'schematic': ['circuit', 'schematic', 'wiring', 'layout'],
    'chart': ['chart', 'bar', 'pie', 'histogram'],
}

# Bloom's taxonomy level indicators
BLOOM_INDICATORS = {
    1: ['define', 'list', 'name', 'identify', 'recall', 'state'],  # Remember
    2: ['explain', 'describe', 'summarize', 'interpret', 'discuss'],  # Understand
    3: ['apply', 'calculate', 'solve', 'use', 'demonstrate', 'compute'],  # Apply
    4: ['analyze', 'compare', 'contrast', 'examine', 'differentiate'],  # Analyze
    5: ['evaluate', 'justify', 'critique', 'assess', 'judge'],  # Evaluate
    6: ['create', 'design', 'develop', 'formulate', 'construct'],  # Create
}

# Math level indicators
MATH_LEVEL_KEYWORDS = {
    'arithmetic': ['addition', 'subtraction', 'multiplication', 'division', 'percentage'],
    'algebra': ['equation', 'variable', 'solve', 'linear', 'quadratic'],
    'trigonometry': ['sine', 'cosine', 'tangent', 'angle', 'triangle'],
    'calculus': ['derivative', 'integral', 'differential', 'limit', 'gradient'],
    'advanced': ['partial differential', 'tensor', 'vector calculus', 'laplacian', 'fourier'],
}


def _env_int(key: str, default: int) -> int:
    """Get integer from environment variable with fallback."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def build_context_windows(chunk: Dict[str, Any], 
                          all_chunks: List[Dict[str, Any]], 
                          index: int) -> Dict[str, Any]:
    """Build extended context windows with prev/next summaries.
    
    Args:
        chunk: Current chunk to add context to
        all_chunks: All chunks in the resource
        index: Current chunk index in all_chunks
        
    Returns:
        Dict with context metadata
    """
    context_window_size = _env_int('CONTEXT_WINDOW_CHARS', 500)
    
    context = {
        'previous_chunk_summary': '',
        'next_chunk_preview': '',
        'section_context': '',
        'chapter_theme': '',
        'surrounding_text_before': '',
        'surrounding_text_after': '',
        'position_in_section': index + 1,
        'total_chunks': len(all_chunks)
    }
    
    # Previous chunk summary
    if index > 0:
        prev_chunk = all_chunks[index - 1]
        prev_text = prev_chunk.get('full_text', '')
        # Take first portion as summary
        context['previous_chunk_summary'] = prev_text[:200].strip()
        # Get surrounding text (last N chars of previous chunk)
        context['surrounding_text_before'] = prev_text[-context_window_size:].strip()
    
    # Next chunk preview
    if index < len(all_chunks) - 1:
        next_chunk = all_chunks[index + 1]
        next_text = next_chunk.get('full_text', '')
        # Take first portion as preview
        context['next_chunk_preview'] = next_text[:200].strip()
        # Get surrounding text (first N chars of next chunk)
        context['surrounding_text_after'] = next_text[:context_window_size].strip()
    
    # Section context
    section_title = chunk.get('section_title', '')
    if section_title:
        context['section_context'] = f"Part of section: {section_title}"
    
    # Chapter theme (try to infer from section structure)
    # Look for chapter-level sections in surrounding chunks
    chapter_candidates = []
    for c in all_chunks[max(0, index-5):min(len(all_chunks), index+5)]:
        s_level = c.get('section_level')
        s_title = c.get('section_title', '')
        if s_level == 1 and s_title:  # Chapter level
            chapter_candidates.append(s_title)
    
    if chapter_candidates:
        context['chapter_theme'] = chapter_candidates[0]
    
    return context


def extract_figure_metadata(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract enhanced figure and table metadata from chunk.
    
    Args:
        chunk: Chunk containing figures or tables
        
    Returns:
        List of figure metadata dicts
    """
    figures = []
    text = chunk.get('full_text', '')
    
    # Look for figure captions and labels
    # Pattern: "Figure X.Y: Caption text"
    fig_caption_pattern = re.compile(
        r'(?:Figure|Fig\.?)\s+(\d+\.?\d*)\s*[:\-]?\s*([^.]+(?:\.[^.]+)?)',
        re.IGNORECASE
    )
    
    for match in fig_caption_pattern.finditer(text):
        label = match.group(1)
        caption = match.group(2).strip()
        
        # Determine figure type from caption keywords
        fig_type = 'unknown'
        caption_lower = caption.lower()
        for ftype, keywords in FIGURE_TYPE_KEYWORDS.items():
            if any(kw in caption_lower for kw in keywords):
                fig_type = ftype
                break
        
        # Extract key concepts mentioned in caption
        described_concepts = []
        # Simple extraction: capitalized phrases
        concept_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
        for concept_match in concept_pattern.finditer(caption):
            concept = concept_match.group(0)
            if len(concept) > 3:  # Skip short words
                described_concepts.append(concept)
        
        # Try to extract axes information for graphs
        axes = {}
        if fig_type == 'graph':
            # Look for "x-axis:", "y-axis:", "versus", etc.
            x_axis_match = re.search(r'x[- ]axis:?\s*([^,;.]+)', caption_lower)
            y_axis_match = re.search(r'y[- ]axis:?\s*([^,;.]+)', caption_lower)
            vs_match = re.search(r'(\w+(?:\s+\w+)?)\s+versus\s+(\w+(?:\s+\w+)?)', caption_lower)
            
            if x_axis_match:
                axes['x'] = x_axis_match.group(1).strip()
            elif vs_match:
                axes['x'] = vs_match.group(2).strip()
            
            if y_axis_match:
                axes['y'] = y_axis_match.group(1).strip()
            elif vs_match:
                axes['y'] = vs_match.group(1).strip()
        
        figure_metadata = {
            'label': f"Figure {label}",
            'caption': caption,
            'type': fig_type,
            'described_concepts': described_concepts[:5],  # Limit to 5
            'axes': axes,
            'key_features': [],  # Could be enhanced with ML
            'interpretation': ''  # Could be enhanced with LLM
        }
        
        figures.append(figure_metadata)
    
    # Similar pattern for tables
    table_caption_pattern = re.compile(
        r'(?:Table)\s+(\d+\.?\d*)\s*[:\-]?\s*([^.]+(?:\.[^.]+)?)',
        re.IGNORECASE
    )
    
    for match in table_caption_pattern.finditer(text):
        label = match.group(1)
        caption = match.group(2).strip()
        
        table_metadata = {
            'label': f"Table {label}",
            'caption': caption,
            'type': 'table',
            'described_concepts': [],
            'columns': [],  # Could be extracted with more sophisticated parsing
            'interpretation': ''
        }
        
        figures.append(table_metadata)  # Include tables in figures list
    
    return figures


def compute_complexity_metrics(chunk: Dict[str, Any], 
                               tags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute complexity and difficulty metrics for a chunk.
    
    Args:
        chunk: Chunk to analyze
        tags: Optional hierarchical tags (if available)
        
    Returns:
        Dict with complexity metrics
    """
    text = chunk.get('full_text', '').lower()
    
    # Determine Bloom's taxonomy level
    bloom_level = 2  # Default to "Understand"
    for level, indicators in BLOOM_INDICATORS.items():
        if any(indicator in text for indicator in indicators):
            bloom_level = max(bloom_level, level)  # Take highest level found
    
    # Determine math level
    math_level = 'none'
    for level, keywords in MATH_LEVEL_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            math_level = level
            # Take most advanced level found
            if math_level == 'advanced':
                break
    
    # Count prerequisites
    prerequisite_count = len(tags.get('prerequisites', [])) if tags else 0
    
    # Count formulas
    formula_count = len(chunk.get('formulas', []))
    
    # Count abstract concepts
    abstract_indicators = ['theory', 'principle', 'axiom', 'theorem', 'postulate', 'law']
    abstract_concept_count = sum(1 for indicator in abstract_indicators if indicator in text)
    
    # Estimate study time based on multiple factors
    base_time = 5  # Base 5 minutes
    
    # Add time based on text length
    word_count = len(text.split())
    reading_time = word_count / 200  # Assume 200 words per minute for technical text
    
    # Add time for formulas (2 min per formula)
    formula_time = formula_count * 2
    
    # Add time based on Bloom level (higher = more time)
    bloom_time = bloom_level * 2
    
    # Add time for prerequisites (1 min per prerequisite to review)
    prereq_time = prerequisite_count * 1
    
    estimated_study_time = int(base_time + reading_time + formula_time + bloom_time + prereq_time)
    
    complexity = {
        'bloom_level': bloom_level,
        'math_level': math_level,
        'prerequisite_count': prerequisite_count,
        'formula_count': formula_count,
        'abstract_concept_count': abstract_concept_count,
        'estimated_study_time_minutes': estimated_study_time,
        'word_count': word_count
    }
    
    return complexity


def extract_cross_references(chunk: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract cross-references to equations, figures, sections, etc.
    
    Args:
        chunk: Chunk to extract references from
        
    Returns:
        Dict with lists of references by type
    """
    text = chunk.get('full_text', '')
    
    cross_refs = {
        'equations': [],
        'figures': [],
        'tables': [],
        'sections': [],
        'chapters': []
    }
    
    # Extract equation references
    for pattern in EQUATION_REF_PATTERNS:
        for match in pattern.finditer(text):
            eq_ref = match.group(1)
            ref_str = f"Eq. {eq_ref}"
            if ref_str not in cross_refs['equations']:
                cross_refs['equations'].append(ref_str)
    
    # Extract figure references
    for pattern in FIGURE_REF_PATTERNS:
        for match in pattern.finditer(text):
            fig_ref = match.group(1)
            ref_str = f"Fig. {fig_ref}"
            if ref_str not in cross_refs['figures']:
                cross_refs['figures'].append(ref_str)
    
    # Extract table references
    for pattern in TABLE_REF_PATTERNS:
        for match in pattern.finditer(text):
            tab_ref = match.group(1)
            ref_str = f"Table {tab_ref}"
            if ref_str not in cross_refs['tables']:
                cross_refs['tables'].append(ref_str)
    
    # Extract section and chapter references
    for pattern in SECTION_REF_PATTERNS:
        for match in pattern.finditer(text):
            ref = match.group(1)
            if 'chapter' in match.group(0).lower():
                ref_str = f"Chapter {ref}"
                if ref_str not in cross_refs['chapters']:
                    cross_refs['chapters'].append(ref_str)
            else:
                ref_str = f"Section {ref}"
                if ref_str not in cross_refs['sections']:
                    cross_refs['sections'].append(ref_str)
    
    # Sort references
    for key in cross_refs:
        cross_refs[key] = sorted(cross_refs[key])
    
    return cross_refs


def enhance_chunk_with_context(chunk: Dict[str, Any],
                               all_chunks: List[Dict[str, Any]],
                               index: int,
                               tags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Add all context enhancements to a chunk.
    
    This is the main entry point combining all context building functions.
    
    Args:
        chunk: Chunk to enhance
        all_chunks: All chunks in resource
        index: Current chunk index
        tags: Optional hierarchical tags
        
    Returns:
        Enhanced chunk with context metadata
    """
    # Build context windows
    context = build_context_windows(chunk, all_chunks, index)
    chunk['context'] = context
    
    # Extract figure metadata
    figures = extract_figure_metadata(chunk)
    if figures:
        chunk['figures'] = figures
        chunk['has_figures'] = True
        chunk['figure_count'] = len(figures)
    
    # Compute complexity metrics
    complexity = compute_complexity_metrics(chunk, tags)
    chunk['complexity'] = complexity
    
    # Extract cross-references
    cross_refs = extract_cross_references(chunk)
    chunk['cross_references'] = cross_refs
    chunk['has_cross_references'] = any(len(refs) > 0 for refs in cross_refs.values())
    
    return chunk
