"""Chunk Relationships and Prerequisite Linking.

This module builds a relationship graph between chunks including:
- Sequential links (previous/next)
- Prerequisite dependencies
- Semantic continuity scoring
- Learning sequences
- Topic transition detection
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from . import embed as embed_service

logger = logging.getLogger("backend.chunk_linker")


def link_chunk_relationships(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build basic prev/next/parent relationships between chunks.
    
    Args:
        chunks: List of chunks to link
        
    Returns:
        Chunks with relationship metadata added
    """
    for i, chunk in enumerate(chunks):
        relationships = {
            'previous_chunk_id': None,
            'next_chunk_id': None,
            'parent_section_id': None,
            'prerequisite_chunk_ids': [],
            'related_chunk_ids': [],
            'continuation_of': None,
            'prepares_for': []
        }
        
        # Link to previous chunk
        if i > 0:
            prev_chunk = chunks[i - 1]
            relationships['previous_chunk_id'] = prev_chunk.get('chunk_id', f'chunk_{i-1}')
        
        # Link to next chunk
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            relationships['next_chunk_id'] = next_chunk.get('chunk_id', f'chunk_{i+1}')
        
        # Identify parent section
        # Look backwards for the most recent section header
        current_section = chunk.get('section_title', '')
        if current_section:
            relationships['parent_section_id'] = current_section
        else:
            # Inherit from previous chunk
            if i > 0:
                relationships['parent_section_id'] = chunks[i-1].get('relationships', {}).get('parent_section_id')
        
        chunk['relationships'] = relationships
    
    return chunks


def extract_prerequisite_links(chunks: List[Dict[str, Any]], 
                               tags: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Link chunks based on prerequisite concepts.
    
    Args:
        chunks: List of chunks with hierarchical tags
        tags: Optional pre-extracted tags (defaults to using chunk tags)
        
    Returns:
        Chunks with prerequisite_chunk_ids populated
    """
    # Build concept index: concept -> list of chunk indices
    concept_index: Dict[str, List[int]] = {}
    
    for i, chunk in enumerate(chunks):
        key_concepts = chunk.get('key_concepts', [])
        # Also index domain/topic/subtopic
        domains = chunk.get('domain', [])
        topics = chunk.get('topic', [])
        
        all_concepts = key_concepts + domains + topics
        for concept in all_concepts:
            if isinstance(concept, str):
                concept_lower = concept.lower()
                if concept_lower not in concept_index:
                    concept_index[concept_lower] = []
                concept_index[concept_lower].append(i)
    
    # For each chunk, find prerequisite chunks
    for i, chunk in enumerate(chunks):
        prerequisites = chunk.get('prerequisites', [])
        prerequisite_chunk_ids = []
        
        for prereq in prerequisites:
            prereq_lower = prereq.lower()
            # Find chunks that define/explain this prerequisite
            if prereq_lower in concept_index:
                # Get chunks that appear before current chunk
                candidate_indices = [idx for idx in concept_index[prereq_lower] if idx < i]
                if candidate_indices:
                    # Take the most recent one
                    prereq_chunk_idx = candidate_indices[-1]
                    prereq_chunk_id = chunks[prereq_chunk_idx].get('chunk_id', f'chunk_{prereq_chunk_idx}')
                    if prereq_chunk_id not in prerequisite_chunk_ids:
                        prerequisite_chunk_ids.append(prereq_chunk_id)
        
        # Update relationships
        if 'relationships' not in chunk:
            chunk['relationships'] = {}
        chunk['relationships']['prerequisite_chunk_ids'] = prerequisite_chunk_ids
    
    return chunks


def compute_semantic_continuity(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calculate topic coherence and continuity between adjacent chunks.
    
    Args:
        chunks: List of chunks
        
    Returns:
        Chunks with continuity metadata added
    """
    for i, chunk in enumerate(chunks):
        continuity = {
            'starts_new_topic': False,
            'concludes_topic': False,
            'transition_type': 'continuation',
            'topic_coherence_score': 0.0,
            'concept_overlap': []
        }
        
        if i > 0:
            prev_chunk = chunks[i - 1]
            
            # Check if topic/section changed
            curr_topic = chunk.get('topic', [])
            prev_topic = prev_chunk.get('topic', [])
            curr_section = chunk.get('section_title', '')
            prev_section = prev_chunk.get('section_title', '')
            
            # Topic transition detection
            if curr_section != prev_section and curr_section:
                continuity['starts_new_topic'] = True
                continuity['transition_type'] = 'new_section'
            elif curr_topic != prev_topic:
                continuity['starts_new_topic'] = True
                continuity['transition_type'] = 'topic_shift'
            
            # Compute concept overlap
            curr_concepts = set(chunk.get('key_concepts', []))
            prev_concepts = set(prev_chunk.get('key_concepts', []))
            overlap = curr_concepts.intersection(prev_concepts)
            continuity['concept_overlap'] = list(overlap)
            
            # Semantic similarity using embeddings (if available)
            try:
                curr_text = chunk.get('full_text', '')[:500]  # First 500 chars
                prev_text = prev_chunk.get('full_text', '')[:500]
                
                if curr_text and prev_text:
                    # Get embeddings
                    embeddings = embed_service.encode_sentences([curr_text, prev_text])
                    if len(embeddings) == 2:
                        # Compute cosine similarity
                        vec1, vec2 = embeddings[0], embeddings[1]
                        dot_product = sum(a * b for a, b in zip(vec1, vec2))
                        mag1 = sum(a * a for a in vec1) ** 0.5
                        mag2 = sum(b * b for b in vec2) ** 0.5
                        if mag1 > 0 and mag2 > 0:
                            similarity = dot_product / (mag1 * mag2)
                            continuity['topic_coherence_score'] = round(similarity, 3)
            except Exception as e:
                logger.debug(f"Could not compute semantic similarity: {e}")
                # Use concept overlap as fallback
                if len(curr_concepts) > 0 or len(prev_concepts) > 0:
                    overlap_ratio = len(overlap) / max(len(curr_concepts), len(prev_concepts), 1)
                    continuity['topic_coherence_score'] = round(overlap_ratio, 3)
            
            # Detect transition types based on content
            curr_content_type = chunk.get('content_type', '')
            if curr_content_type == 'example' and continuity['topic_coherence_score'] > 0.5:
                continuity['transition_type'] = 'example'
            elif curr_content_type == 'derivation':
                continuity['transition_type'] = 'derivation'
            elif curr_content_type == 'summary':
                continuity['transition_type'] = 'summary'
                continuity['concludes_topic'] = True
        
        chunk['continuity'] = continuity
    
    return chunks


def build_learning_sequence(chunks: List[Dict[str, Any]], 
                            section: Optional[str] = None) -> List[Dict[str, Any]]:
    """Order chunks by learning dependencies and add sequence metadata.
    
    Args:
        chunks: List of chunks to sequence
        section: Optional section filter
        
    Returns:
        Chunks with sequence metadata added
    """
    # Filter by section if provided
    if section:
        section_chunks = [c for c in chunks if c.get('section_title') == section]
    else:
        section_chunks = chunks
    
    # Group by section for counting
    sections: Dict[str, List[int]] = {}
    for i, chunk in enumerate(chunks):
        sec = chunk.get('section_title', 'unknown')
        if sec not in sections:
            sections[sec] = []
        sections[sec].append(i)
    
    # Add sequence metadata
    for i, chunk in enumerate(chunks):
        sec = chunk.get('section_title', 'unknown')
        section_indices = sections.get(sec, [])
        
        sequence = {
            'position_in_section': section_indices.index(i) + 1 if i in section_indices else 0,
            'total_in_section': len(section_indices),
            'position_in_chapter': i + 1,
            'total_in_chapter': len(chunks),
            'builds_on': [],
            'prepares_for': [],
            'prerequisite_chain_depth': 0,
            'is_foundational': False
        }
        
        # Identify what this chunk builds on
        prereq_ids = chunk.get('relationships', {}).get('prerequisite_chunk_ids', [])
        if prereq_ids:
            # Get prerequisite concepts
            for prereq_id in prereq_ids:
                # Find the chunk with this ID
                for other_chunk in chunks:
                    if other_chunk.get('chunk_id') == prereq_id:
                        prereq_concepts = other_chunk.get('key_concepts', [])
                        sequence['builds_on'].extend(prereq_concepts[:3])  # Limit to 3
                        break
        
        # Calculate prerequisite chain depth (depth-first search)
        visited = set()
        depth = _compute_prereq_depth(chunk, chunks, visited)
        sequence['prerequisite_chain_depth'] = depth
        
        # Mark as foundational if no prerequisites
        if depth == 0 and not prereq_ids:
            sequence['is_foundational'] = True
        
        # Identify what this chunk prepares for (look ahead)
        prepares_for_concepts = []
        key_concepts = chunk.get('key_concepts', [])
        for j in range(i + 1, min(i + 6, len(chunks))):  # Look ahead up to 5 chunks
            future_chunk = chunks[j]
            future_prereqs = future_chunk.get('prerequisites', [])
            for prereq in future_prereqs:
                if any(prereq.lower() in concept.lower() for concept in key_concepts):
                    future_topics = future_chunk.get('topic', [])
                    prepares_for_concepts.extend(future_topics)
        
        sequence['prepares_for'] = list(set(prepares_for_concepts))[:3]  # Unique, limit to 3
        
        chunk['sequence'] = sequence
    
    return chunks


def _compute_prereq_depth(chunk: Dict[str, Any], 
                          all_chunks: List[Dict[str, Any]], 
                          visited: Set[str]) -> int:
    """Recursively compute prerequisite chain depth.
    
    Args:
        chunk: Current chunk
        all_chunks: All chunks
        visited: Set of visited chunk IDs to avoid cycles
        
    Returns:
        Maximum depth of prerequisite chain
    """
    chunk_id = chunk.get('chunk_id', '')
    if chunk_id in visited:
        return 0  # Avoid cycles
    
    visited.add(chunk_id)
    
    prereq_ids = chunk.get('relationships', {}).get('prerequisite_chunk_ids', [])
    if not prereq_ids:
        return 0
    
    max_depth = 0
    for prereq_id in prereq_ids:
        # Find prerequisite chunk
        for other_chunk in all_chunks:
            if other_chunk.get('chunk_id') == prereq_id:
                depth = 1 + _compute_prereq_depth(other_chunk, all_chunks, visited.copy())
                max_depth = max(max_depth, depth)
                break
    
    return max_depth


def identify_topic_transitions(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect where new topics start and end in the chunk sequence.
    
    Args:
        chunks: List of chunks
        
    Returns:
        Chunks with topic_transition flags added
    """
    for i, chunk in enumerate(chunks):
        topic_transition = {
            'is_topic_start': False,
            'is_topic_end': False,
            'transition_confidence': 0.0,
            'new_topic_name': '',
            'previous_topic_name': ''
        }
        
        if i > 0:
            prev_chunk = chunks[i - 1]
            curr_section = chunk.get('section_title', '')
            prev_section = prev_chunk.get('section_title', '')
            
            # Strong signal: section change
            if curr_section and curr_section != prev_section:
                topic_transition['is_topic_start'] = True
                topic_transition['transition_confidence'] = 0.95
                topic_transition['new_topic_name'] = curr_section
                topic_transition['previous_topic_name'] = prev_section
            
            # Check content type transitions
            curr_type = chunk.get('content_type', '')
            prev_type = prev_chunk.get('content_type', '')
            
            # Summary followed by new content = topic end/start
            if prev_type == 'summary' and curr_type in ['concept_intro', 'definition']:
                topic_transition['is_topic_start'] = True
                topic_transition['transition_confidence'] = 0.75
            
            # Check continuity score
            continuity = chunk.get('continuity', {})
            coherence = continuity.get('topic_coherence_score', 0.0)
            
            if coherence < 0.3 and not topic_transition['is_topic_start']:
                # Low coherence = likely new topic
                topic_transition['is_topic_start'] = True
                topic_transition['transition_confidence'] = 0.6
                curr_topics = chunk.get('topic', [])
                if curr_topics:
                    topic_transition['new_topic_name'] = curr_topics[0] if isinstance(curr_topics, list) else curr_topics
        
        chunk['topic_transition'] = topic_transition
    
    # Second pass: mark topic ends based on next chunk's topic starts
    for i in range(len(chunks) - 1):
        next_chunk = chunks[i + 1]
        if next_chunk.get('topic_transition', {}).get('is_topic_start'):
            chunks[i]['topic_transition']['is_topic_end'] = True
    
    return chunks


def link_all_relationships(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply all relationship linking functions to chunks.
    
    This is the main entry point combining all linking functions.
    
    Args:
        chunks: List of chunks to link
        
    Returns:
        Fully linked chunks with all relationship metadata
    """
    # Add chunk IDs if not present
    for i, chunk in enumerate(chunks):
        if 'chunk_id' not in chunk:
            chunk['chunk_id'] = f"chunk_{i}_{chunk.get('page_number', 0)}"
    
    # Apply linking functions in order
    chunks = link_chunk_relationships(chunks)
    chunks = extract_prerequisite_links(chunks)
    chunks = compute_semantic_continuity(chunks)
    chunks = build_learning_sequence(chunks)
    chunks = identify_topic_transitions(chunks)
    
    logger.info(f"Linked {len(chunks)} chunks with relationships")
    
    return chunks
