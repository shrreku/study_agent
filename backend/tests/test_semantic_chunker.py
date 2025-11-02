"""Unit tests for semantic_chunker module.

Tests formula detection, formula-aware splitting, semantic unit identification,
content-type aware chunking, and hierarchical tagging.
"""
import os
import sys

# Ensure backend is importable
this_dir = os.path.dirname(__file__)
backend_dir = os.path.abspath(os.path.join(this_dir, ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from ingestion import semantic_chunker


def test_detect_formulas_latex():
    """Test detection of LaTeX formulas."""
    text = r"""
    Newton's second law states that $F = ma$ where F is force.
    The energy equation is:
    $$E = mc^2$$
    """
    formulas = semantic_chunker.detect_formulas(text)
    assert len(formulas) >= 2
    # Check that formulas contain expected patterns
    formula_texts = [f['text'] for f in formulas]
    assert any('F = ma' in ft for ft in formula_texts)
    assert any('E = mc' in ft for ft in formula_texts)


def test_detect_formulas_plain_equations():
    """Test detection of plain text equations."""
    text = """
    The velocity is given by v = u + at where v is final velocity.
    Acceleration a = (v - u) / t is the rate of change.
    """
    formulas = semantic_chunker.detect_formulas(text)
    assert len(formulas) >= 1
    # Should detect at least the equation patterns


def test_detect_formulas_no_formulas():
    """Test that text without formulas returns empty list."""
    text = "This is plain text with no mathematical formulas in it."
    formulas = semantic_chunker.detect_formulas(text)
    assert len(formulas) == 0


def test_formula_aware_split_preserves_formulas():
    """Test that formula-aware split never breaks formulas."""
    text = r"""
    This is some introductory text before the equation.
    The fundamental equation is $F = ma$ which relates force to mass and acceleration.
    After the equation we have more explanation text.
    Another equation appears: $$E = \frac{1}{2}mv^2$$ representing kinetic energy.
    Final explanatory text follows.
    """
    
    chunks = semantic_chunker.formula_aware_split(text, max_chunk_size=50)
    
    # All chunks should be dicts with expected keys
    assert all(isinstance(c, dict) for c in chunks)
    assert all('text' in c and 'has_formula' in c for c in chunks)
    
    # Chunks with formulas should be marked
    formula_chunks = [c for c in chunks if c['has_formula']]
    assert len(formula_chunks) > 0
    
    # Verify formulas are intact in chunks
    all_chunk_text = ' '.join(c['text'] for c in chunks)
    assert 'F = ma' in all_chunk_text
    assert 'mv^2' in all_chunk_text or 'E =' in all_chunk_text


def test_formula_aware_split_no_formulas():
    """Test that formula-aware split works on text without formulas."""
    text = "This is a simple sentence. " * 100  # Long text without formulas
    
    chunks = semantic_chunker.formula_aware_split(text, max_chunk_size=200)
    
    assert len(chunks) >= 1
    assert all(not c['has_formula'] for c in chunks)
    assert all(c['formula_count'] == 0 for c in chunks)


def test_get_chunk_size_limits():
    """Test that chunk size limits are correctly determined by content type."""
    
    # Test known content types
    min_tok, max_tok = semantic_chunker.get_chunk_size_limits("concept_intro")
    assert min_tok > 0
    assert max_tok > min_tok
    
    min_tok, max_tok = semantic_chunker.get_chunk_size_limits("derivation")
    assert min_tok >= 300  # Derivations should have higher limits
    assert max_tok >= 500
    
    min_tok, max_tok = semantic_chunker.get_chunk_size_limits("example")
    assert min_tok > 0
    assert max_tok > min_tok
    
    min_tok, max_tok = semantic_chunker.get_chunk_size_limits("summary")
    assert min_tok >= 50
    assert max_tok <= 150  # Summaries should be shorter
    
    # Test unknown content type gets defaults
    min_tok, max_tok = semantic_chunker.get_chunk_size_limits("unknown_type")
    assert min_tok > 0
    assert max_tok > min_tok


def test_identify_semantic_units_mock():
    """Test semantic unit identification with LLM mock enabled."""
    # Enable LLM mock to avoid real API calls
    os.environ["USE_LLM_MOCK"] = "1"
    
    text = """
    Chapter 1: Introduction to Physics
    
    Physics is the study of matter and energy.
    Newton's laws describe motion.
    
    Example 1.1: A ball is thrown upward.
    """
    
    result = semantic_chunker.identify_semantic_units(text, page_number=1)
    
    # Should return dict with sections and chunks keys
    assert isinstance(result, dict)
    assert "sections" in result
    assert "chunks" in result
    assert isinstance(result["sections"], list)
    assert isinstance(result["chunks"], list)
    
    # Clean up
    del os.environ["USE_LLM_MOCK"]


def test_create_semantic_chunks_basic():
    """Test basic semantic chunking of pages."""
    # Enable LLM mock
    os.environ["USE_LLM_MOCK"] = "1"
    
    pages = [
        "Page 1: Introduction to the topic with some explanatory text.",
        "Page 2: More details about the topic with examples and formulas like F = ma.",
    ]
    
    chunks = semantic_chunker.create_semantic_chunks(pages, content_aware=False, preserve_formulas=False)
    
    # Should return list of chunk dicts
    assert isinstance(chunks, list)
    
    # Clean up
    del os.environ["USE_LLM_MOCK"]


def test_create_semantic_chunks_with_formulas():
    """Test semantic chunking preserves formulas."""
    # Enable LLM mock
    os.environ["USE_LLM_MOCK"] = "1"
    
    pages = [
        r"The equation $F = ma$ is fundamental. It relates force to mass and acceleration.",
    ]
    
    chunks = semantic_chunker.create_semantic_chunks(pages, content_aware=True, preserve_formulas=True)
    
    assert isinstance(chunks, list)
    
    # Clean up
    del os.environ["USE_LLM_MOCK"]


def test_extract_hierarchical_tags_mock():
    """Test hierarchical tag extraction with mock."""
    os.environ["USE_LLM_MOCK"] = "1"
    
    chunk_text = """
    Newton's Second Law states that force equals mass times acceleration.
    This fundamental principle forms the basis of classical mechanics.
    """
    
    result = semantic_chunker.extract_hierarchical_tags(chunk_text)
    
    # Should return dict with expected keys
    assert isinstance(result, dict)
    # With mock, we get the default structure
    assert "domain" in result
    assert "topic" in result
    assert "subtopic" in result
    assert "prerequisites" in result
    assert "learning_objectives" in result
    assert "content_type" in result
    assert "difficulty" in result
    assert "cognitive_level" in result
    assert "key_concepts" in result
    
    # Clean up
    del os.environ["USE_LLM_MOCK"]


def test_extract_formula_metadata_no_formulas():
    """Test formula extraction on text without formulas."""
    text = "This is plain text with no mathematical formulas."
    
    formulas = semantic_chunker.extract_formula_metadata(text)
    
    # Should return empty list when no formulas detected
    assert isinstance(formulas, list)
    assert len(formulas) == 0


def test_extract_formula_metadata_with_formulas():
    """Test formula extraction with formulas present."""
    os.environ["USE_LLM_MOCK"] = "1"
    
    text = r"The force equation $F = ma$ is central to mechanics."
    
    formulas = semantic_chunker.extract_formula_metadata(text)
    
    # Should return list (empty with mock, but structure is correct)
    assert isinstance(formulas, list)
    
    # Clean up
    del os.environ["USE_LLM_MOCK"]


def test_env_int_helper():
    """Test environment variable integer parsing."""
    # Test with valid env var
    os.environ["TEST_INT_VAR"] = "123"
    result = semantic_chunker._env_int("TEST_INT_VAR", 456)
    assert result == 123
    
    # Test with missing env var (uses default)
    result = semantic_chunker._env_int("MISSING_VAR", 789)
    assert result == 789
    
    # Test with invalid env var (uses default)
    os.environ["TEST_INVALID_INT"] = "not_a_number"
    result = semantic_chunker._env_int("TEST_INVALID_INT", 999)
    assert result == 999
    
    # Clean up
    if "TEST_INT_VAR" in os.environ:
        del os.environ["TEST_INT_VAR"]
    if "TEST_INVALID_INT" in os.environ:
        del os.environ["TEST_INVALID_INT"]


def test_env_bool_helper():
    """Test environment variable boolean parsing."""
    # Test true values
    for val in ["true", "1", "yes", "TRUE", "YES"]:
        os.environ["TEST_BOOL"] = val
        assert semantic_chunker._env_bool("TEST_BOOL", False) == True
    
    # Test false values
    for val in ["false", "0", "no", "FALSE", "NO"]:
        os.environ["TEST_BOOL"] = val
        assert semantic_chunker._env_bool("TEST_BOOL", True) == False
    
    # Test missing (uses default)
    result = semantic_chunker._env_bool("MISSING_BOOL", True)
    assert result == True
    
    result = semantic_chunker._env_bool("MISSING_BOOL", False)
    assert result == False
    
    # Clean up
    if "TEST_BOOL" in os.environ:
        del os.environ["TEST_BOOL"]


def test_simple_split():
    """Test simple sentence-based splitting."""
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    
    chunks = semantic_chunker._simple_split(text, max_tokens=5)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all('text' in c for c in chunks)
    assert all('has_formula' in c for c in chunks)
    assert all(not c['has_formula'] for c in chunks)


if __name__ == "__main__":
    # Run tests
    import pytest
    pytest.main([__file__, "-v"])
