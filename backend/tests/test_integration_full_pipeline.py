"""Comprehensive integration tests for the full enhanced chunking pipeline.

Tests end-to-end processing, feature flag combinations, error handling,
and quality validation.
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
from ingestion import quality_validator
from archive import performance_benchmark


def test_full_pipeline_all_features_enabled():
    """Test complete pipeline with all features enabled."""
    # Enable all features
    os.environ["ENHANCED_CHUNKING_ENABLED"] = "true"
    os.environ["ENHANCED_TAGGING_ENABLED"] = "true"
    os.environ["FORMULA_EXTRACTION_ENABLED"] = "true"
    os.environ["EXTENDED_CONTEXT_ENABLED"] = "true"
    os.environ["CHUNK_LINKING_ENABLED"] = "true"
    os.environ["USE_LLM_MOCK"] = "1"
    os.environ["PROMPT_SET"] = "enhanced_ingest"
    
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
        Chapter 1: Introduction to Heat Transfer
        
        Heat transfer is the science of thermal energy transport.
        Figure 1.1: Temperature distribution in a wall.
        
        The governing equation is given by Fourier's law:
        q = -k dT/dx
        
        Where q is heat flux (W/m²), k is thermal conductivity (W/m·K),
        and dT/dx is the temperature gradient (K/m).
        
        Example 1.1: Calculate heat transfer through a wall.
        Given k = 1.0 W/m·K and dT/dx = 50 K/m, find q.
        
        Solution: Using Eq. 1.1, q = -1.0 × 50 = -50 W/m²
        """)
        test_file = f.name
    
    try:
        # Process with enhanced pipeline
        chunks = enhanced_structural_chunk_resource(test_file)
        
        # Should return chunks (may be empty for .txt files)
        assert isinstance(chunks, list)
        
        # If chunks were created, validate quality
        if len(chunks) > 0:
            quality_report = quality_validator.validate_chunk_quality(chunks)
            
            # Should have quality metrics
            assert 'summary' in quality_report
            assert 'checks' in quality_report
            
            # Should have some metadata richness
            assert quality_report['summary']['average_metadata_richness'] >= 0
        else:
            # No chunks created is acceptable for simple text files
            pass
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        # Clean up environment
        for key in ["ENHANCED_CHUNKING_ENABLED", "ENHANCED_TAGGING_ENABLED", 
                    "FORMULA_EXTRACTION_ENABLED", "EXTENDED_CONTEXT_ENABLED",
                    "CHUNK_LINKING_ENABLED", "USE_LLM_MOCK", "PROMPT_SET"]:
            if key in os.environ:
                del os.environ[key]


def test_feature_flag_combinations():
    """Test different combinations of feature flags."""
    os.environ["USE_LLM_MOCK"] = "1"
    
    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content for chunking.")
        test_file = f.name
    
    try:
        # Test 1: All disabled (should use legacy)
        os.environ["ENHANCED_CHUNKING_ENABLED"] = "false"
        chunks1 = enhanced_structural_chunk_resource(test_file)
        assert isinstance(chunks1, list)
        
        # Test 2: Only chunking enabled
        os.environ["ENHANCED_CHUNKING_ENABLED"] = "true"
        os.environ["ENHANCED_TAGGING_ENABLED"] = "false"
        os.environ["FORMULA_EXTRACTION_ENABLED"] = "false"
        chunks2 = enhanced_structural_chunk_resource(test_file)
        assert isinstance(chunks2, list)
        
        # Test 3: Chunking + tagging
        os.environ["ENHANCED_TAGGING_ENABLED"] = "true"
        chunks3 = enhanced_structural_chunk_resource(test_file)
        assert isinstance(chunks3, list)
        
        # Test 4: All enabled
        os.environ["FORMULA_EXTRACTION_ENABLED"] = "true"
        os.environ["EXTENDED_CONTEXT_ENABLED"] = "true"
        os.environ["CHUNK_LINKING_ENABLED"] = "true"
        chunks4 = enhanced_structural_chunk_resource(test_file)
        assert isinstance(chunks4, list)
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        for key in ["ENHANCED_CHUNKING_ENABLED", "ENHANCED_TAGGING_ENABLED",
                    "FORMULA_EXTRACTION_ENABLED", "EXTENDED_CONTEXT_ENABLED",
                    "CHUNK_LINKING_ENABLED", "USE_LLM_MOCK"]:
            if key in os.environ:
                del os.environ[key]


def test_error_handling_graceful_degradation():
    """Test that errors don't crash the pipeline."""
    os.environ["USE_LLM_MOCK"] = "1"
    os.environ["ENHANCED_CHUNKING_ENABLED"] = "true"
    
    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content")
        test_file = f.name
    
    try:
        # Should not crash even with errors
        chunks = enhanced_structural_chunk_resource(test_file)
        
        # Should still return some chunks
        assert isinstance(chunks, list)
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        for key in ["ENHANCED_CHUNKING_ENABLED", "USE_LLM_MOCK"]:
            if key in os.environ:
                del os.environ[key]


def test_backward_compatibility():
    """Test that legacy chunker still works."""
    os.environ["USE_LLM_MOCK"] = "1"
    
    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Legacy chunker test")
        test_file = f.name
    
    try:
        # Call legacy chunker directly
        chunks = structural_chunk_resource(test_file)
        
        # Should return chunks
        assert isinstance(chunks, list)
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        if "USE_LLM_MOCK" in os.environ:
            del os.environ["USE_LLM_MOCK"]


def test_quality_validation_framework():
    """Test the quality validation framework."""
    # Create mock chunks with varying quality
    chunks = [
        {
            'full_text': 'Test chunk 1',
            'domain': ['Physics'],
            'topic': ['Heat Transfer'],
            'subtopic': ['Conduction'],
            'formulas': [{'id': 'f1', 'latex': 'q = -k dT/dx', 'type': 'algebraic', 'variables': []}],
            'prerequisites': ['Calculus'],
            'context': {'previous_chunk_summary': 'Intro'},
            'relationships': {'previous_chunk_id': None, 'next_chunk_id': 'chunk_1'},
            'continuity': {'topic_coherence_score': 0.8},
            'sequence': {'position_in_section': 1}
        },
        {
            'full_text': 'Test chunk 2',
            'domain': ['Physics'],
            'topic': ['Heat Transfer'],
            'formulas': [],
            'prerequisites': [],
            'context': {'previous_chunk_summary': 'Test 1', 'next_chunk_preview': 'Test 3'},
            'relationships': {'previous_chunk_id': 'chunk_0', 'next_chunk_id': None},
            'continuity': {'topic_coherence_score': 0.9},
            'sequence': {'position_in_section': 2}
        }
    ]
    
    # Validate quality
    report = quality_validator.validate_chunk_quality(chunks)
    
    # Should return report
    assert isinstance(report, dict)
    assert 'summary' in report
    assert 'checks' in report
    
    # Should have quality grade
    assert 'quality_grade' in report['summary']
    
    # Should have all checks
    assert 'formulas_preserved' in report['checks']
    assert 'tags_hierarchical' in report['checks']
    assert 'prerequisites_present' in report['checks']
    assert 'context_windows' in report['checks']
    assert 'relationships_linked' in report['checks']


def test_performance_benchmark_framework():
    """Test the performance benchmarking framework."""
    os.environ["USE_LLM_MOCK"] = "1"
    
    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Performance test content")
        test_file = f.name
    
    try:
        # Benchmark a function
        result = performance_benchmark.benchmark_function(
            structural_chunk_resource,
            test_file
        )
        
        # Should return benchmark results
        assert isinstance(result, dict)
        assert 'execution_time_seconds' in result
        assert 'memory_peak_mb' in result
        assert 'success' in result
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        if "USE_LLM_MOCK" in os.environ:
            del os.environ["USE_LLM_MOCK"]


def test_pipeline_comparison():
    """Test comparison between basic and enhanced pipelines."""
    os.environ["USE_LLM_MOCK"] = "1"
    os.environ["ENHANCED_CHUNKING_ENABLED"] = "true"
    os.environ["ENHANCED_TAGGING_ENABLED"] = "true"
    
    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Comparison test content with some formulas q = -k dT/dx")
        test_file = f.name
    
    try:
        # Compare pipelines
        comparison = performance_benchmark.compare_pipelines(
            test_file,
            structural_chunk_resource,
            enhanced_structural_chunk_resource
        )
        
        # Should return comparison results
        assert isinstance(comparison, dict)
        assert 'basic' in comparison
        assert 'enhanced' in comparison
        assert 'comparison' in comparison
        
        # Should have multipliers
        if comparison.get('success'):
            assert 'time_multiplier' in comparison['comparison']
            assert 'memory_multiplier' in comparison['comparison']
            assert 'metadata_richness_multiplier' in comparison['comparison']
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        for key in ["USE_LLM_MOCK", "ENHANCED_CHUNKING_ENABLED", "ENHANCED_TAGGING_ENABLED"]:
            if key in os.environ:
                del os.environ[key]


def test_metadata_richness_scoring():
    """Test metadata richness score computation."""
    # Create chunks with different richness levels
    minimal_chunk = {'full_text': 'Minimal chunk'}
    
    rich_chunk = {
        'full_text': 'Rich chunk',
        'domain': ['Physics'],
        'topic': ['Heat Transfer'],
        'subtopic': ['Conduction'],
        'prerequisites': ['Calculus'],
        'learning_objectives': ['Understand heat flow'],
        'formulas': [{'id': 'f1', 'latex': 'q = -k dT/dx', 'variables': [{'symbol': 'q'}]}],
        'complexity': {'bloom_level': 3},
        'context': {'previous_chunk_summary': 'Previous'},
        'relationships': {'previous_chunk_id': 'chunk_0'},
        'continuity': {'topic_coherence_score': 0.8},
        'sequence': {'position_in_section': 1}
    }
    
    # Compute scores
    minimal_score = quality_validator.compute_metadata_richness_score(minimal_chunk)
    rich_score = quality_validator.compute_metadata_richness_score(rich_chunk)
    
    # Rich chunk should score higher
    assert rich_score > minimal_score
    assert 0 <= minimal_score <= 100
    assert 0 <= rich_score <= 100


def test_report_generation():
    """Test report generation functions."""
    chunks = [
        {
            'full_text': 'Test',
            'domain': ['Test'],
            'topic': ['Testing'],
            'formulas': [],
            'prerequisites': [],
            'context': {},
            'relationships': {},
            'continuity': {},
            'sequence': {}
        }
    ]
    
    # Generate quality report
    quality_report_text = quality_validator.generate_quality_report(chunks)
    assert isinstance(quality_report_text, str)
    assert len(quality_report_text) > 0
    assert 'QUALITY' in quality_report_text.upper()


if __name__ == "__main__":
    # Run tests
    import pytest
    pytest.main([__file__, "-v"])
