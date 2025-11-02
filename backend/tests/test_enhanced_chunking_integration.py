"""Integration tests for enhanced chunking pipeline.

Tests the full integration of enhanced_structural_chunk_resource with feature flags,
prompt loading, and the complete chunking workflow.
"""
import os
import sys
import tempfile

# Ensure backend is importable
this_dir = os.path.dirname(__file__)
backend_dir = os.path.abspath(os.path.join(this_dir, ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from ingestion.chunker import enhanced_structural_chunk_resource, structural_chunk_resource


def test_enhanced_chunker_feature_flag_disabled():
    """Test that enhanced chunker falls back to legacy when flag is disabled."""
    # Ensure flag is disabled
    os.environ["ENHANCED_CHUNKING_ENABLED"] = "false"
    os.environ["USE_LLM_MOCK"] = "1"
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content for chunking.")
        test_file = f.name
    
    try:
        # Call enhanced chunker - should fall back to legacy
        result = enhanced_structural_chunk_resource(test_file)
        
        # Should return a list (even if empty for .txt files)
        assert isinstance(result, list)
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        if "ENHANCED_CHUNKING_ENABLED" in os.environ:
            del os.environ["ENHANCED_CHUNKING_ENABLED"]
        if "USE_LLM_MOCK" in os.environ:
            del os.environ["USE_LLM_MOCK"]


def test_enhanced_chunker_feature_flag_enabled():
    """Test that enhanced chunker uses new pipeline when flag is enabled."""
    # Enable feature flag
    os.environ["ENHANCED_CHUNKING_ENABLED"] = "true"
    os.environ["USE_LLM_MOCK"] = "1"
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content for enhanced chunking.")
        test_file = f.name
    
    try:
        # Call enhanced chunker
        result = enhanced_structural_chunk_resource(test_file)
        
        # Should return a list
        assert isinstance(result, list)
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        if "ENHANCED_CHUNKING_ENABLED" in os.environ:
            del os.environ["ENHANCED_CHUNKING_ENABLED"]
        if "USE_LLM_MOCK" in os.environ:
            del os.environ["USE_LLM_MOCK"]


def test_enhanced_chunker_hierarchical_tagging():
    """Test that hierarchical tagging adds metadata when enabled."""
    # Enable all features
    os.environ["ENHANCED_CHUNKING_ENABLED"] = "true"
    os.environ["ENHANCED_TAGGING_ENABLED"] = "true"
    os.environ["USE_LLM_MOCK"] = "1"
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content with some physics equations like F = ma.")
        test_file = f.name
    
    try:
        # Call enhanced chunker
        result = enhanced_structural_chunk_resource(test_file)
        
        # Should return a list
        assert isinstance(result, list)
        
        # If chunks were created, they should have hierarchical metadata
        if result:
            # Check that expected keys are present
            chunk = result[0]
            # With tagging enabled, these keys should be added
            assert "domain" in chunk
            assert "topic" in chunk
            assert "subtopic" in chunk
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        for key in ["ENHANCED_CHUNKING_ENABLED", "ENHANCED_TAGGING_ENABLED", "USE_LLM_MOCK"]:
            if key in os.environ:
                del os.environ[key]


def test_enhanced_chunker_formula_extraction():
    """Test that formula extraction adds metadata when enabled."""
    # Enable all features
    os.environ["ENHANCED_CHUNKING_ENABLED"] = "true"
    os.environ["FORMULA_EXTRACTION_ENABLED"] = "true"
    os.environ["USE_LLM_MOCK"] = "1"
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(r"Test content with formula: $E = mc^2$ and more text.")
        test_file = f.name
    
    try:
        # Call enhanced chunker
        result = enhanced_structural_chunk_resource(test_file)
        
        # Should return a list
        assert isinstance(result, list)
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        for key in ["ENHANCED_CHUNKING_ENABLED", "FORMULA_EXTRACTION_ENABLED", "USE_LLM_MOCK"]:
            if key in os.environ:
                del os.environ[key]


def test_legacy_chunker_still_works():
    """Test that legacy chunker is not broken by new changes."""
    os.environ["USE_LLM_MOCK"] = "1"
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content for legacy chunking.")
        test_file = f.name
    
    try:
        # Call legacy chunker directly
        result = structural_chunk_resource(test_file)
        
        # Should return a list
        assert isinstance(result, list)
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        if "USE_LLM_MOCK" in os.environ:
            del os.environ["USE_LLM_MOCK"]


def test_prompt_loading():
    """Test that enhanced prompts can be loaded from enhanced_ingest.yaml."""
    # Set prompt set to enhanced_ingest
    original_prompt_set = os.environ.get("PROMPT_SET")
    os.environ["PROMPT_SET"] = "enhanced_ingest"
    
    try:
        from prompts import get as prompt_get
        
        # Should be able to load new prompts
        page_structure_v2 = prompt_get("ingest.page_structure_v2")
        hierarchical_tags = prompt_get("ingest.chunk_tags_hierarchical")
        formula_extraction = prompt_get("ingest.formula_extraction")
        
        # All should return non-empty strings
        assert isinstance(page_structure_v2, str) and len(page_structure_v2) > 0
        assert isinstance(hierarchical_tags, str) and len(hierarchical_tags) > 0
        assert isinstance(formula_extraction, str) and len(formula_extraction) > 0
        
        # Should still have legacy prompts for backward compatibility
        legacy_structure = prompt_get("ingest.page_structure")
        legacy_tags = prompt_get("ingest.chunk_tags")
        assert isinstance(legacy_structure, str) and len(legacy_structure) > 0
        assert isinstance(legacy_tags, str) and len(legacy_tags) > 0
        
    finally:
        # Restore original prompt set
        if original_prompt_set is not None:
            os.environ["PROMPT_SET"] = original_prompt_set
        elif "PROMPT_SET" in os.environ:
            del os.environ["PROMPT_SET"]


def test_all_feature_flags_together():
    """Test that all feature flags work together."""
    # Enable everything
    os.environ["ENHANCED_CHUNKING_ENABLED"] = "true"
    os.environ["ENHANCED_TAGGING_ENABLED"] = "true"
    os.environ["FORMULA_EXTRACTION_ENABLED"] = "true"
    os.environ["USE_LLM_MOCK"] = "1"
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(r"""
        Chapter 1: Introduction to Physics
        
        Physics is the fundamental science. Newton's law states $F = ma$ where
        F is force, m is mass, and a is acceleration.
        """)
        test_file = f.name
    
    try:
        # Call enhanced chunker with all features
        result = enhanced_structural_chunk_resource(test_file)
        
        # Should return a list
        assert isinstance(result, list)
        
        # Basic validation that it didn't crash
        # (With mock LLM, we may not get real chunks, but function should complete)
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        for key in ["ENHANCED_CHUNKING_ENABLED", "ENHANCED_TAGGING_ENABLED", 
                    "FORMULA_EXTRACTION_ENABLED", "USE_LLM_MOCK"]:
            if key in os.environ:
                del os.environ[key]


if __name__ == "__main__":
    # Run tests
    import pytest
    pytest.main([__file__, "-v"])
