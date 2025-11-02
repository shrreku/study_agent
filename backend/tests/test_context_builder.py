"""Unit tests for context_builder module.

Tests context window building, figure metadata extraction, complexity metrics,
and cross-reference parsing.
"""
import os
import sys

# Ensure backend is importable
this_dir = os.path.dirname(__file__)
backend_dir = os.path.abspath(os.path.join(this_dir, ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from ingestion import context_builder as cb


def test_build_context_windows_middle_chunk():
    """Test context window building for a middle chunk."""
    chunks = [
        {"full_text": "First chunk text about heat transfer.", "section_title": "Chapter 1"},
        {"full_text": "Second chunk with more detailed explanation of conduction.", "section_title": "Chapter 1"},
        {"full_text": "Third chunk introducing convection concepts.", "section_title": "Chapter 1"},
    ]
    
    context = cb.build_context_windows(chunks[1], chunks, 1)
    
    assert isinstance(context, dict)
    assert 'previous_chunk_summary' in context
    assert 'next_chunk_preview' in context
    assert 'surrounding_text_before' in context
    assert 'surrounding_text_after' in context
    
    # Should have previous and next summaries
    assert len(context['previous_chunk_summary']) > 0
    assert len(context['next_chunk_preview']) > 0
    assert 'heat transfer' in context['previous_chunk_summary'].lower()
    assert 'convection' in context['next_chunk_preview'].lower()


def test_build_context_windows_first_chunk():
    """Test context window for first chunk (no previous)."""
    chunks = [
        {"full_text": "First chunk", "section_title": "Intro"},
        {"full_text": "Second chunk", "section_title": "Intro"},
    ]
    
    context = cb.build_context_windows(chunks[0], chunks, 0)
    
    assert context['previous_chunk_summary'] == ''
    assert context['surrounding_text_before'] == ''
    assert len(context['next_chunk_preview']) > 0


def test_build_context_windows_last_chunk():
    """Test context window for last chunk (no next)."""
    chunks = [
        {"full_text": "First chunk", "section_title": "Intro"},
        {"full_text": "Last chunk", "section_title": "Intro"},
    ]
    
    context = cb.build_context_windows(chunks[1], chunks, 1)
    
    assert context['next_chunk_preview'] == ''
    assert context['surrounding_text_after'] == ''
    assert len(context['previous_chunk_summary']) > 0


def test_extract_figure_metadata_with_figures():
    """Test figure extraction from chunk with figure captions."""
    chunk = {
        "full_text": """
        The results are shown in Figure 5.3: Temperature decay in lumped capacitance system.
        This plot shows exponential behavior versus time.
        See also Table 2.1: Properties of common materials.
        """
    }
    
    figures = cb.extract_figure_metadata(chunk)
    
    assert isinstance(figures, list)
    assert len(figures) >= 1
    
    # Should find at least the figure
    fig_labels = [f['label'] for f in figures]
    assert any('5.3' in label for label in fig_labels)
    
    # Check structure
    for fig in figures:
        assert 'label' in fig
        assert 'caption' in fig
        assert 'type' in fig


def test_extract_figure_metadata_graph_type():
    """Test figure type detection for graphs."""
    chunk = {
        "full_text": "Figure 2.5: Plot of temperature versus time showing exponential decay."
    }
    
    figures = cb.extract_figure_metadata(chunk)
    
    assert len(figures) >= 1
    # Should detect as graph due to "versus" keyword
    assert figures[0]['type'] in ['graph', 'unknown']


def test_extract_figure_metadata_no_figures():
    """Test figure extraction on text without figures."""
    chunk = {
        "full_text": "This is plain text with no figures or tables mentioned."
    }
    
    figures = cb.extract_figure_metadata(chunk)
    
    assert isinstance(figures, list)
    assert len(figures) == 0


def test_compute_complexity_metrics_simple():
    """Test complexity computation for simple chunk."""
    chunk = {
        "full_text": "Define heat transfer as the movement of thermal energy.",
        "formulas": []
    }
    
    tags = {
        "prerequisites": [],
        "domain": ["Physics"]
    }
    
    complexity = cb.compute_complexity_metrics(chunk, tags)
    
    assert isinstance(complexity, dict)
    assert 'bloom_level' in complexity
    assert 'math_level' in complexity
    assert 'prerequisite_count' in complexity
    assert 'formula_count' in complexity
    assert 'estimated_study_time_minutes' in complexity
    
    # Should have low Bloom level (define = level 1)
    assert complexity['bloom_level'] >= 1
    
    # Should have estimated study time
    assert complexity['estimated_study_time_minutes'] > 0


def test_compute_complexity_metrics_advanced():
    """Test complexity computation for advanced chunk."""
    chunk = {
        "full_text": """
        Analyze the partial differential equation for heat conduction.
        The Laplacian operator appears in the formulation.
        This requires understanding of vector calculus and Fourier series.
        Theory of thermal diffusion is fundamental.
        """,
        "formulas": [
            {"type": "differential_equation"},
            {"type": "differential_equation"},
            {"type": "integral"}
        ]
    }
    
    tags = {
        "prerequisites": ["Vector Calculus", "Fourier Analysis", "PDEs"],
        "domain": ["Physics", "Mathematics"]
    }
    
    complexity = cb.compute_complexity_metrics(chunk, tags)
    
    # Should have high Bloom level (analyze = level 4)
    assert complexity['bloom_level'] >= 4
    
    # Should detect advanced math level
    assert complexity['math_level'] in ['advanced', 'calculus']
    
    # Should have high prerequisite count
    assert complexity['prerequisite_count'] == 3
    
    # Should have multiple formulas
    assert complexity['formula_count'] == 3
    
    # Should have higher study time
    assert complexity['estimated_study_time_minutes'] > 10


def test_compute_complexity_metrics_with_apply():
    """Test Bloom level detection for apply level."""
    chunk = {
        "full_text": "Calculate the heat flux using Fourier's law. Apply the formula to solve the problem.",
        "formulas": [{"type": "algebraic"}]
    }
    
    complexity = cb.compute_complexity_metrics(chunk, None)
    
    # Should detect apply level (calculate, solve)
    assert complexity['bloom_level'] >= 3


def test_extract_cross_references_equations():
    """Test extraction of equation references."""
    chunk = {
        "full_text": """
        Using Eq. 5.8 and Equation 5.9, we can derive the result.
        See also (5.10) for the boundary condition.
        """
    }
    
    refs = cb.extract_cross_references(chunk)
    
    assert isinstance(refs, dict)
    assert 'equations' in refs
    assert 'figures' in refs
    assert 'tables' in refs
    assert 'sections' in refs
    
    # Should find equation references
    assert len(refs['equations']) >= 2
    assert any('5.8' in eq for eq in refs['equations'])
    assert any('5.9' in eq for eq in refs['equations'])


def test_extract_cross_references_figures():
    """Test extraction of figure references."""
    chunk = {
        "full_text": "As shown in Figure 3.2 and Fig. 3.5, the temperature increases."
    }
    
    refs = cb.extract_cross_references(chunk)
    
    assert len(refs['figures']) >= 2
    assert any('3.2' in fig for fig in refs['figures'])
    assert any('3.5' in fig for fig in refs['figures'])


def test_extract_cross_references_sections():
    """Test extraction of section and chapter references."""
    chunk = {
        "full_text": "Refer to Section 2.3 and Chapter 5 for more details."
    }
    
    refs = cb.extract_cross_references(chunk)
    
    assert len(refs['sections']) >= 1
    assert len(refs['chapters']) >= 1
    assert any('2.3' in sec for sec in refs['sections'])
    assert any('5' in ch for ch in refs['chapters'])


def test_extract_cross_references_no_refs():
    """Test cross-reference extraction on text without references."""
    chunk = {
        "full_text": "This is plain text with no cross-references."
    }
    
    refs = cb.extract_cross_references(chunk)
    
    assert all(len(refs[key]) == 0 for key in refs)


def test_enhance_chunk_with_context_complete():
    """Test complete context enhancement on a chunk."""
    chunks = [
        {
            "full_text": "Introduction to heat transfer concepts.",
            "section_title": "Chapter 1",
            "key_concepts": ["Heat Transfer"],
            "formulas": []
        },
        {
            "full_text": """
            Define conduction as heat transfer through solid material.
            See Figure 1.2: Heat flow diagram.
            Apply Fourier's law from Eq. 1.5.
            """,
            "section_title": "Chapter 1",
            "key_concepts": ["Conduction", "Fourier's Law"],
            "formulas": [{"type": "algebraic"}]
        },
        {
            "full_text": "Next topic: Convection heat transfer.",
            "section_title": "Chapter 1",
            "key_concepts": ["Convection"],
            "formulas": []
        }
    ]
    
    tags = {
        "prerequisites": [],
        "domain": ["Physics"]
    }
    
    enhanced = cb.enhance_chunk_with_context(chunks[1], chunks, 1, tags)
    
    # Should have context
    assert 'context' in enhanced
    assert 'previous_chunk_summary' in enhanced['context']
    
    # Should have figures
    assert 'figures' in enhanced
    assert len(enhanced['figures']) >= 1
    assert enhanced['has_figures'] == True
    
    # Should have complexity
    assert 'complexity' in enhanced
    assert 'bloom_level' in enhanced['complexity']
    
    # Should have cross-references
    assert 'cross_references' in enhanced
    assert len(enhanced['cross_references']['figures']) >= 1
    assert len(enhanced['cross_references']['equations']) >= 1
    assert enhanced['has_cross_references'] == True


if __name__ == "__main__":
    # Run tests
    import pytest
    pytest.main([__file__, "-v"])
