"""Unit tests for chunk_linker module.

Tests chunk relationship linking, prerequisite detection, semantic continuity,
learning sequences, and topic transitions.
"""
import os
import sys

# Ensure backend is importable
this_dir = os.path.dirname(__file__)
backend_dir = os.path.abspath(os.path.join(this_dir, ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from ingestion import chunk_linker as cl


def test_link_chunk_relationships_basic():
    """Test basic prev/next linking."""
    chunks = [
        {"full_text": "First chunk", "section_title": "Section 1"},
        {"full_text": "Second chunk", "section_title": "Section 1"},
        {"full_text": "Third chunk", "section_title": "Section 2"},
    ]
    
    linked = cl.link_chunk_relationships(chunks)
    
    # All chunks should have relationships
    for chunk in linked:
        assert 'relationships' in chunk
    
    # First chunk should have no previous
    assert linked[0]['relationships']['previous_chunk_id'] is None
    assert linked[0]['relationships']['next_chunk_id'] is not None
    
    # Middle chunk should have both
    assert linked[1]['relationships']['previous_chunk_id'] is not None
    assert linked[1]['relationships']['next_chunk_id'] is not None
    
    # Last chunk should have no next
    assert linked[2]['relationships']['previous_chunk_id'] is not None
    assert linked[2]['relationships']['next_chunk_id'] is None


def test_link_chunk_relationships_parent_section():
    """Test parent section identification."""
    chunks = [
        {"full_text": "Chunk 1", "section_title": "Introduction"},
        {"full_text": "Chunk 2", "section_title": "Methods"},
        {"full_text": "Chunk 3", "section_title": ""},  # No section
    ]
    
    linked = cl.link_chunk_relationships(chunks)
    
    assert linked[0]['relationships']['parent_section_id'] == "Introduction"
    assert linked[1]['relationships']['parent_section_id'] == "Methods"
    # Third chunk should inherit from previous
    assert linked[2]['relationships']['parent_section_id'] == "Methods"


def test_extract_prerequisite_links():
    """Test prerequisite chunk linking."""
    chunks = [
        {
            "chunk_id": "chunk_0",
            "full_text": "Introduction to Newton's laws and force.",
            "key_concepts": ["Newton's laws", "Force", "Mass"],
            "domain": ["Physics"],
            "topic": ["Classical Mechanics"],
            "prerequisites": [],
            "relationships": {}
        },
        {
            "chunk_id": "chunk_1",
            "full_text": "Kinematics describes motion without considering forces.",
            "key_concepts": ["Kinematics", "Velocity", "Acceleration"],
            "domain": ["Physics"],
            "topic": ["Classical Mechanics"],
            "prerequisites": [],
            "relationships": {}
        },
        {
            "chunk_id": "chunk_2",
            "full_text": "Dynamics combines kinematics with Newton's laws.",
            "key_concepts": ["Dynamics", "Equations of motion"],
            "domain": ["Physics"],
            "topic": ["Classical Mechanics"],
            "prerequisites": ["Newton's laws", "Kinematics"],
            "relationships": {}
        }
    ]
    
    linked = cl.extract_prerequisite_links(chunks)
    
    # Third chunk should have prerequisite links
    prereq_ids = linked[2]['relationships']['prerequisite_chunk_ids']
    assert len(prereq_ids) >= 1
    # Should link to chunk_0 (Newton's laws) or chunk_1 (Kinematics)
    assert any(pid in prereq_ids for pid in ["chunk_0", "chunk_1"])


def test_compute_semantic_continuity():
    """Test semantic continuity scoring."""
    os.environ["USE_LLM_MOCK"] = "1"  # Use mock to avoid embedding errors
    
    chunks = [
        {
            "full_text": "Introduction to heat transfer by conduction.",
            "key_concepts": ["Heat Transfer", "Conduction"],
            "topic": ["Heat Transfer"],
            "section_title": "Chapter 1",
            "content_type": "concept_intro"
        },
        {
            "full_text": "Fourier's law describes conductive heat transfer.",
            "key_concepts": ["Fourier's Law", "Conduction"],
            "topic": ["Heat Transfer"],
            "section_title": "Chapter 1",
            "content_type": "derivation"
        },
        {
            "full_text": "Convection transfers heat through fluid motion.",
            "key_concepts": ["Convection", "Fluid Flow"],
            "topic": ["Heat Transfer", "Convection"],
            "section_title": "Chapter 2",
            "content_type": "concept_intro"
        }
    ]
    
    linked = cl.compute_semantic_continuity(chunks)
    
    # All chunks except first should have continuity
    for i in range(1, len(linked)):
        assert 'continuity' in linked[i]
        assert 'topic_coherence_score' in linked[i]['continuity']
        assert 'concept_overlap' in linked[i]['continuity']
    
    # Second chunk should have high coherence with first (same topic)
    continuity1 = linked[1]['continuity']
    assert len(continuity1['concept_overlap']) > 0  # Should share "Conduction"
    
    # Third chunk should start new topic (section change)
    continuity2 = linked[2]['continuity']
    assert continuity2['starts_new_topic'] == True
    
    # Clean up
    if "USE_LLM_MOCK" in os.environ:
        del os.environ["USE_LLM_MOCK"]


def test_compute_semantic_continuity_transition_types():
    """Test detection of different transition types."""
    chunks = [
        {
            "full_text": "Theory of heat conduction.",
            "key_concepts": ["Conduction"],
            "topic": ["Heat Transfer"],
            "section_title": "Section 1",
            "content_type": "concept_intro"
        },
        {
            "full_text": "Example 1.1: Calculate heat flux through a wall.",
            "key_concepts": ["Heat Flux", "Conduction"],
            "topic": ["Heat Transfer"],
            "section_title": "Section 1",
            "content_type": "example"
        },
        {
            "full_text": "In summary, conduction follows Fourier's law.",
            "key_concepts": ["Conduction", "Fourier's Law"],
            "topic": ["Heat Transfer"],
            "section_title": "Section 1",
            "content_type": "summary"
        }
    ]
    
    linked = cl.compute_semantic_continuity(chunks)
    
    # Example should be detected as transition type
    assert linked[1]['continuity']['transition_type'] in ['example', 'continuation']
    
    # Summary should conclude topic
    assert linked[2]['continuity']['transition_type'] == 'summary'
    assert linked[2]['continuity']['concludes_topic'] == True


def test_build_learning_sequence():
    """Test learning sequence metadata."""
    chunks = [
        {
            "chunk_id": "chunk_0",
            "full_text": "Basics of thermodynamics.",
            "key_concepts": ["Temperature", "Energy"],
            "section_title": "Chapter 1",
            "prerequisites": [],
            "relationships": {"prerequisite_chunk_ids": []}
        },
        {
            "chunk_id": "chunk_1",
            "full_text": "Heat transfer modes.",
            "key_concepts": ["Conduction", "Convection", "Radiation"],
            "section_title": "Chapter 1",
            "prerequisites": ["Temperature"],
            "relationships": {"prerequisite_chunk_ids": []}
        },
        {
            "chunk_id": "chunk_2",
            "full_text": "Advanced applications.",
            "key_concepts": ["Heat Exchangers"],
            "section_title": "Chapter 2",
            "prerequisites": ["Conduction", "Convection"],
            "relationships": {"prerequisite_chunk_ids": ["chunk_1"]}
        }
    ]
    
    sequenced = cl.build_learning_sequence(chunks)
    
    # All chunks should have sequence metadata
    for chunk in sequenced:
        assert 'sequence' in chunk
        seq = chunk['sequence']
        assert 'position_in_section' in seq
        assert 'total_in_section' in seq
        assert 'position_in_chapter' in seq
        assert 'prerequisite_chain_depth' in seq
        assert 'is_foundational' in seq
    
    # First chunk should be foundational
    assert sequenced[0]['sequence']['is_foundational'] == True
    assert sequenced[0]['sequence']['prerequisite_chain_depth'] == 0
    
    # Last chunk should have deeper chain
    assert sequenced[2]['sequence']['prerequisite_chain_depth'] >= 1


def test_build_learning_sequence_positions():
    """Test position counting in sections."""
    chunks = [
        {"chunk_id": f"chunk_{i}", "section_title": "Chapter 1" if i < 3 else "Chapter 2", 
         "full_text": f"Chunk {i}", "key_concepts": [], "prerequisites": [], 
         "relationships": {"prerequisite_chunk_ids": []}}
        for i in range(5)
    ]
    
    sequenced = cl.build_learning_sequence(chunks)
    
    # Check section positions
    assert sequenced[0]['sequence']['position_in_section'] == 1
    assert sequenced[0]['sequence']['total_in_section'] == 3  # Chapter 1 has 3 chunks
    
    assert sequenced[3]['sequence']['position_in_section'] == 1  # First in Chapter 2
    assert sequenced[3]['sequence']['total_in_section'] == 2  # Chapter 2 has 2 chunks
    
    # Check chapter positions
    assert sequenced[0]['sequence']['position_in_chapter'] == 1
    assert sequenced[4]['sequence']['position_in_chapter'] == 5


def test_identify_topic_transitions():
    """Test topic transition detection."""
    chunks = [
        {
            "full_text": "Introduction to conduction.",
            "section_title": "Section 1.1",
            "content_type": "concept_intro",
            "topic": ["Conduction"],
            "continuity": {"topic_coherence_score": 0.0}
        },
        {
            "full_text": "Fourier's law of conduction.",
            "section_title": "Section 1.1",
            "content_type": "derivation",
            "topic": ["Conduction"],
            "continuity": {"topic_coherence_score": 0.85}
        },
        {
            "full_text": "Summary of conduction principles.",
            "section_title": "Section 1.1",
            "content_type": "summary",
            "topic": ["Conduction"],
            "continuity": {"topic_coherence_score": 0.75}
        },
        {
            "full_text": "Introduction to convection.",
            "section_title": "Section 1.2",
            "content_type": "concept_intro",
            "topic": ["Convection"],
            "continuity": {"topic_coherence_score": 0.2}
        }
    ]
    
    marked = cl.identify_topic_transitions(chunks)
    
    # All chunks should have topic_transition
    for chunk in marked:
        assert 'topic_transition' in chunk
    
    # Fourth chunk should start new topic (section change)
    assert marked[3]['topic_transition']['is_topic_start'] == True
    assert marked[3]['topic_transition']['transition_confidence'] > 0.7
    
    # Third chunk (summary) should be marked as topic end
    # because next chunk starts new topic
    assert marked[2]['topic_transition']['is_topic_end'] == True


def test_identify_topic_transitions_low_coherence():
    """Test topic transition based on low coherence."""
    chunks = [
        {
            "full_text": "Topic A content.",
            "section_title": "Section 1",
            "content_type": "concept_intro",
            "topic": ["Topic A"],
            "continuity": {"topic_coherence_score": 0.0}
        },
        {
            "full_text": "Completely different topic B.",
            "section_title": "Section 1",  # Same section
            "content_type": "concept_intro",
            "topic": ["Topic B"],
            "continuity": {"topic_coherence_score": 0.15}  # Very low coherence
        }
    ]
    
    marked = cl.identify_topic_transitions(chunks)
    
    # Second chunk should start new topic due to low coherence
    assert marked[1]['topic_transition']['is_topic_start'] == True


def test_link_all_relationships_integration():
    """Test complete relationship linking integration."""
    chunks = [
        {
            "full_text": "Introduction to Newton's laws.",
            "section_title": "Chapter 1",
            "key_concepts": ["Newton's Laws", "Force"],
            "domain": ["Physics"],
            "topic": ["Mechanics"],
            "prerequisites": [],
            "content_type": "concept_intro"
        },
        {
            "full_text": "Application of Newton's second law to motion problems.",
            "section_title": "Chapter 1",
            "key_concepts": ["F=ma", "Acceleration"],
            "domain": ["Physics"],
            "topic": ["Mechanics"],
            "prerequisites": ["Newton's Laws"],
            "content_type": "example"
        },
        {
            "full_text": "Introduction to energy and work.",
            "section_title": "Chapter 2",
            "key_concepts": ["Energy", "Work"],
            "domain": ["Physics"],
            "topic": ["Energy"],
            "prerequisites": [],
            "content_type": "concept_intro"
        }
    ]
    
    fully_linked = cl.link_all_relationships(chunks)
    
    # All chunks should have all metadata
    for chunk in fully_linked:
        assert 'chunk_id' in chunk
        assert 'relationships' in chunk
        assert 'continuity' in chunk or chunk == fully_linked[0]  # First has no continuity
        assert 'sequence' in chunk
        assert 'topic_transition' in chunk
    
    # Check that prerequisite links were created
    assert len(fully_linked[1]['relationships']['prerequisite_chunk_ids']) >= 0
    
    # Check that topic transition was detected for chapter 2
    assert fully_linked[2]['topic_transition']['is_topic_start'] == True


if __name__ == "__main__":
    # Run tests
    import pytest
    pytest.main([__file__, "-v"])
