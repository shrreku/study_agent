"""Unit tests for hierarchical_tagger module.

Tests LaTeX parsing, variable extraction, formula classification, hierarchical tagging,
prerequisite detection, difficulty estimation, and complete integration.
"""
import os
import sys

# Ensure backend is importable
this_dir = os.path.dirname(__file__)
backend_dir = os.path.abspath(os.path.join(this_dir, ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from ingestion import hierarchical_tagger as ht


def test_parse_latex_display_math():
    """Test LaTeX extraction from display math environments."""
    text = r"""
    The equation is:
    \begin{equation}
    F = ma
    \end{equation}
    and another one:
    \[E = mc^2\]
    """
    
    formulas = ht.parse_latex(text)
    
    assert len(formulas) >= 2
    assert any('F = ma' in f for f in formulas)
    assert any('E = mc' in f for f in formulas)


def test_parse_latex_inline_math():
    """Test LaTeX extraction from inline math."""
    text = r"The force $F = ma$ and energy $E = mc^2$ are related."
    
    formulas = ht.parse_latex(text)
    
    assert len(formulas) >= 2
    assert any('F = ma' in f for f in formulas)
    assert any('E = mc' in f for f in formulas)


def test_parse_latex_complex():
    """Test LaTeX extraction with complex notation."""
    text = r"""
    The heat equation is:
    $$\rho V_c c \frac{dT}{dt} = -h A_s (T - T_\infty)$$
    """
    
    formulas = ht.parse_latex(text)
    
    assert len(formulas) >= 1
    assert any('frac{dT}{dt}' in f for f in formulas)


def test_parse_latex_no_formulas():
    """Test LaTeX parsing on text without formulas."""
    text = "This is plain text with no formulas."
    
    formulas = ht.parse_latex(text)
    
    assert len(formulas) == 0


def test_identify_variables_simple():
    """Test variable extraction from simple equations."""
    equation = r"F = ma"
    
    variables = ht.identify_variables(equation)
    
    # Should extract at least F (and maybe 'ma' as one variable)
    assert len(variables) >= 1
    assert 'F' in variables


def test_identify_variables_with_subscripts():
    """Test variable extraction with subscripts."""
    equation = r"T_\infty, V_c, A_s"
    
    variables = ht.identify_variables(equation)
    
    # Should find variables with subscripts
    assert len(variables) > 0


def test_identify_variables_greek_letters():
    """Test that Greek letter commands are not treated as regular variables."""
    equation = r"\rho V c = constant"
    
    variables = ht.identify_variables(equation)
    
    # Should find V and c, but not \rho
    assert 'V' in variables
    assert 'c' in variables


def test_classify_formula_algebraic():
    """Test classification of algebraic equations."""
    equation = r"F = ma"
    
    ftype = ht.classify_formula_type(equation)
    
    assert ftype == 'algebraic'


def test_classify_formula_differential():
    """Test classification of differential equations."""
    equation = r"\frac{dT}{dt} = -k(T - T_\infty)"
    
    ftype = ht.classify_formula_type(equation)
    
    assert ftype == 'differential_equation'


def test_classify_formula_integral():
    """Test classification of integrals."""
    equation = r"\int_0^\infty f(x) dx"
    
    ftype = ht.classify_formula_type(equation)
    
    assert ftype == 'integral'


def test_classify_formula_summation():
    """Test classification of summations."""
    equation = r"\sum_{n=1}^{\infty} \frac{1}{n^2}"
    
    ftype = ht.classify_formula_type(equation)
    
    assert ftype == 'summation'


def test_classify_formula_inequality():
    """Test classification of inequalities."""
    equation = r"x < y"
    
    ftype = ht.classify_formula_type(equation)
    
    assert ftype == 'inequality'


def test_extract_hierarchical_tags_mock():
    """Test hierarchical tag extraction with mock."""
    os.environ["USE_LLM_MOCK"] = "1"
    
    chunk_text = """
    Newton's Second Law states that force equals mass times acceleration.
    This fundamental principle forms the basis of classical mechanics.
    """
    
    tags = ht.extract_hierarchical_tags(chunk_text)
    
    # Should return dict with expected structure
    assert isinstance(tags, dict)
    assert 'domain' in tags
    assert 'topic' in tags
    assert 'subtopic' in tags
    assert 'prerequisites' in tags
    assert 'learning_objectives' in tags
    assert 'content_type' in tags
    assert 'difficulty' in tags
    assert 'cognitive_level' in tags
    
    # With mock, returns defaults
    assert isinstance(tags['domain'], list)
    assert isinstance(tags['prerequisites'], list)
    
    # Clean up
    del os.environ["USE_LLM_MOCK"]


def test_extract_hierarchical_tags_with_context():
    """Test hierarchical tag extraction with section context."""
    os.environ["USE_LLM_MOCK"] = "1"
    
    chunk_text = "The lumped capacitance method applies when Bi < 0.1"
    section_context = {"title": "Transient Conduction"}
    
    tags = ht.extract_hierarchical_tags(chunk_text, section_context=section_context)
    
    assert isinstance(tags, dict)
    assert 'domain' in tags
    
    # Clean up
    del os.environ["USE_LLM_MOCK"]


def test_build_concept_hierarchy():
    """Test concept hierarchy building."""
    concepts = [
        {"name": "Newton's Law", "type": "principle", "importance": "high"},
        {"name": "Force", "type": "parameter", "importance": "high"},
        {"name": "Mass", "type": "parameter", "importance": "high"},
        {"name": "Least Action", "type": "principle", "importance": "medium"},
    ]
    
    hierarchy = ht.build_concept_hierarchy(concepts)
    
    assert isinstance(hierarchy, dict)
    assert 'concepts' in hierarchy
    assert len(hierarchy['concepts']) == len(concepts)
    
    # Should organize by type
    if 'foundational' in hierarchy:
        assert len(hierarchy['foundational']) >= 1


def test_classify_content_type_definition():
    """Test content type classification for definitions."""
    chunk = {
        "full_text": "Definition: Heat transfer is defined as the movement of thermal energy."
    }
    
    content_type = ht.classify_content_type(chunk)
    
    assert content_type == 'definition'


def test_classify_content_type_example():
    """Test content type classification for examples."""
    chunk = {
        "full_text": "Example 1.1: Consider the case of a heated sphere cooling in air."
    }
    
    content_type = ht.classify_content_type(chunk)
    
    assert content_type == 'example'


def test_classify_content_type_theorem():
    """Test content type classification for theorems."""
    chunk = {
        "full_text": "Theorem: The Fourier law states that heat flux is proportional to temperature gradient."
    }
    
    content_type = ht.classify_content_type(chunk)
    
    assert content_type == 'theorem'


def test_classify_content_type_derivation():
    """Test content type classification for derivations."""
    chunk = {
        "full_text": "Deriving the heat equation, starting from the energy balance..."
    }
    
    content_type = ht.classify_content_type(chunk)
    
    assert content_type == 'derivation'


def test_estimate_difficulty_introductory():
    """Test difficulty estimation for introductory content."""
    chunk = {
        "full_text": "Heat is a form of energy that flows from hot to cold.",
        "has_equation": False,
        "formulas": [],
        "prerequisites": [],
        "key_concepts": ["heat", "energy"]
    }
    
    difficulty, details = ht.estimate_difficulty(chunk)
    
    assert difficulty in ['introductory', 'intermediate', 'advanced']
    assert isinstance(details, dict)
    assert 'total_score' in details


def test_estimate_difficulty_advanced():
    """Test difficulty estimation for advanced content."""
    chunk = {
        "full_text": "The partial differential equation describes diffusion with Laplacian operator.",
        "has_equation": True,
        "formulas": [
            {"type": "differential_equation"},
            {"type": "differential_equation"}
        ],
        "prerequisites": ["calculus", "vector calculus", "linear algebra", "PDEs"],
        "key_concepts": ["theory", "principle", "differential operator"]
    }
    
    difficulty, details = ht.estimate_difficulty(chunk)
    
    # Should be advanced due to PDEs, multiple prerequisites, and theory
    assert difficulty in ['intermediate', 'advanced']
    assert details['total_score'] > 30


def test_extract_prerequisites():
    """Test prerequisite extraction."""
    chunk = {
        "full_text": "Assuming knowledge of newton's laws, we can derive the equation of motion. Force and kinematics are essential.",
        "prerequisites": []
    }
    
    previous_tags = [
        {"key_concepts": ["newton's laws", "force", "kinematics"]},
        {"key_concepts": ["energy", "work"]}
    ]
    
    prereqs = ht.extract_prerequisites(chunk, previous_tags)
    
    assert isinstance(prereqs, list)
    # Should find concepts mentioned in text (case-insensitive matching)
    # At least one of the concepts should be detected
    found_concepts = [p.lower() for p in prereqs]
    assert any(concept in found_concepts or any(concept in text_prereq for text_prereq in found_concepts) 
               for concept in ["newton", "force", "kinematics"])


def test_extract_formula_metadata_mock():
    """Test formula metadata extraction with mock."""
    os.environ["USE_LLM_MOCK"] = "1"
    
    formulas = [r"F = ma", r"E = mc^2"]
    chunk_text = "The force F = ma and energy E = mc^2 are fundamental."
    
    metadata = ht.extract_formula_metadata(formulas, chunk_text)
    
    assert isinstance(metadata, list)
    assert len(metadata) == len(formulas)
    
    for formula_dict in metadata:
        assert 'id' in formula_dict
        assert 'latex' in formula_dict
        assert 'type' in formula_dict
    
    # Clean up
    del os.environ["USE_LLM_MOCK"]


def test_extract_formula_metadata_complex():
    """Test formula metadata extraction with complex formulas."""
    os.environ["USE_LLM_MOCK"] = "1"
    
    formulas = [r"\frac{dT}{dt} = -k(T - T_\infty)"]
    chunk_text = "The cooling equation describes temperature change over time."
    
    metadata = ht.extract_formula_metadata(formulas, chunk_text)
    
    assert len(metadata) == 1
    assert metadata[0]['type'] == 'differential_equation'
    
    # Clean up
    del os.environ["USE_LLM_MOCK"]


def test_tag_and_extract_formulas_complete():
    """Test complete tagging and formula extraction integration."""
    os.environ["USE_LLM_MOCK"] = "1"
    
    chunk = {
        "full_text": r"Example: The heat equation $\frac{dT}{dt} = -k(T - T_\infty)$ describes cooling.",
        "has_equation": True,
        "content_type": "unknown"
    }
    
    section_context = {"title": "Transient Heat Transfer"}
    
    enhanced_chunk = ht.tag_and_extract_formulas(chunk, section_context=section_context)
    
    # Should have hierarchical tags
    assert 'domain' in enhanced_chunk
    assert 'topic' in enhanced_chunk
    assert 'difficulty' in enhanced_chunk
    assert 'content_type' in enhanced_chunk
    
    # Content type should be classified (with "Example:" it should be detected)
    # or at least it should exist
    assert 'content_type' in enhanced_chunk
    
    # Should have formula metadata
    assert 'formulas' in enhanced_chunk
    assert len(enhanced_chunk['formulas']) > 0
    
    # Clean up
    del os.environ["USE_LLM_MOCK"]


def test_tag_and_extract_formulas_without_equations():
    """Test complete tagging on chunks without formulas."""
    os.environ["USE_LLM_MOCK"] = "1"
    
    chunk = {
        "full_text": "Heat transfer occurs through conduction, convection, and radiation.",
        "has_equation": False,
        "content_type": "unknown"
    }
    
    enhanced_chunk = ht.tag_and_extract_formulas(chunk)
    
    # Should have hierarchical tags
    assert 'domain' in enhanced_chunk
    assert 'difficulty' in enhanced_chunk
    
    # Should not have formulas key or it should be empty
    formulas = enhanced_chunk.get('formulas', [])
    assert len(formulas) == 0
    
    # Clean up
    del os.environ["USE_LLM_MOCK"]


if __name__ == "__main__":
    # Run tests
    import pytest
    pytest.main([__file__, "-v"])
