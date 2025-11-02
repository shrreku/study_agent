"""Performance Benchmarking Suite for Enhanced Chunking Pipeline.

This module provides comprehensive performance benchmarks comparing
basic vs enhanced chunking pipelines.
"""

from typing import List, Dict, Any, Callable
import time
import os
import logging
import tracemalloc
from contextlib import contextmanager

logger = logging.getLogger("backend.performance_benchmark")


@contextmanager
def measure_time():
    """Context manager to measure execution time."""
    start = time.time()
    yield lambda: time.time() - start
    end_time = time.time() - start


@contextmanager
def measure_memory():
    """Context manager to measure memory usage."""
    tracemalloc.start()
    yield lambda: tracemalloc.get_traced_memory()
    tracemalloc.stop()


def benchmark_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Benchmark a function's performance.
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Dict with benchmark results
    """
    # Measure time
    start_time = time.time()
    tracemalloc.start()
    
    try:
        result = func(*args, **kwargs)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
        logger.exception(f"Benchmark function failed: {func.__name__}")
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'function': func.__name__,
        'success': success,
        'error': error,
        'execution_time_seconds': round(end_time - start_time, 3),
        'memory_current_mb': round(current / 1024 / 1024, 2),
        'memory_peak_mb': round(peak / 1024 / 1024, 2),
        'result': result
    }


def benchmark_chunking_pipeline(resource_path: str, 
                                chunker_func: Callable,
                                pipeline_name: str) -> Dict[str, Any]:
    """Benchmark a complete chunking pipeline.
    
    Args:
        resource_path: Path to resource to process
        chunker_func: Chunking function to benchmark
        pipeline_name: Name of the pipeline for reporting
        
    Returns:
        Dict with comprehensive benchmark results
    """
    logger.info(f"Benchmarking {pipeline_name} on {resource_path}")
    
    # Overall pipeline benchmark
    overall = benchmark_function(chunker_func, resource_path)
    
    if not overall['success']:
        return {
            'pipeline': pipeline_name,
            'resource': resource_path,
            'success': False,
            'error': overall['error']
        }
    
    chunks = overall['result']
    
    # Compute metrics
    total_chunks = len(chunks)
    total_text = sum(len(c.get('full_text', '')) for c in chunks)
    avg_chunk_size = total_text / max(total_chunks, 1)
    
    # Count chunks by type
    content_types = {}
    for chunk in chunks:
        ctype = chunk.get('content_type', 'unknown')
        content_types[ctype] = content_types.get(ctype, 0) + 1
    
    # Count metadata
    chunks_with_formulas = sum(1 for c in chunks if c.get('formulas'))
    chunks_with_tags = sum(1 for c in chunks if c.get('domain'))
    chunks_with_context = sum(1 for c in chunks if c.get('context'))
    chunks_with_relationships = sum(1 for c in chunks if c.get('relationships'))
    
    return {
        'pipeline': pipeline_name,
        'resource': resource_path,
        'success': True,
        'execution_time_seconds': overall['execution_time_seconds'],
        'memory_peak_mb': overall['memory_peak_mb'],
        'total_chunks': total_chunks,
        'total_text_chars': total_text,
        'avg_chunk_size_chars': round(avg_chunk_size, 2),
        'time_per_chunk_ms': round((overall['execution_time_seconds'] / max(total_chunks, 1)) * 1000, 2),
        'content_type_distribution': content_types,
        'metadata_counts': {
            'with_formulas': chunks_with_formulas,
            'with_tags': chunks_with_tags,
            'with_context': chunks_with_context,
            'with_relationships': chunks_with_relationships
        }
    }


def compare_pipelines(resource_path: str,
                     basic_chunker: Callable,
                     enhanced_chunker: Callable) -> Dict[str, Any]:
    """Compare basic vs enhanced chunking pipelines.
    
    Args:
        resource_path: Path to resource to process
        basic_chunker: Basic chunking function
        enhanced_chunker: Enhanced chunking function
        
    Returns:
        Dict with comparison results
    """
    logger.info(f"Comparing pipelines on {resource_path}")
    
    # Benchmark both pipelines
    basic_results = benchmark_chunking_pipeline(resource_path, basic_chunker, "Basic")
    enhanced_results = benchmark_chunking_pipeline(resource_path, enhanced_chunker, "Enhanced")
    
    if not basic_results['success'] or not enhanced_results['success']:
        return {
            'success': False,
            'basic': basic_results,
            'enhanced': enhanced_results
        }
    
    # Compute comparisons
    time_multiplier = enhanced_results['execution_time_seconds'] / max(basic_results['execution_time_seconds'], 0.001)
    memory_multiplier = enhanced_results['memory_peak_mb'] / max(basic_results['memory_peak_mb'], 0.1)
    
    # Metadata richness comparison
    basic_metadata = sum(basic_results['metadata_counts'].values())
    enhanced_metadata = sum(enhanced_results['metadata_counts'].values())
    metadata_multiplier = enhanced_metadata / max(basic_metadata, 1)
    
    return {
        'success': True,
        'resource': resource_path,
        'basic': basic_results,
        'enhanced': enhanced_results,
        'comparison': {
            'time_multiplier': round(time_multiplier, 2),
            'memory_multiplier': round(memory_multiplier, 2),
            'metadata_richness_multiplier': round(metadata_multiplier, 2),
            'time_within_target': time_multiplier < 3.0,
            'memory_within_target': memory_multiplier < 2.0
        }
    }


def benchmark_suite(resource_paths: List[str],
                   basic_chunker: Callable,
                   enhanced_chunker: Callable) -> Dict[str, Any]:
    """Run comprehensive benchmark suite on multiple resources.
    
    Args:
        resource_paths: List of resource paths to benchmark
        basic_chunker: Basic chunking function
        enhanced_chunker: Enhanced chunking function
        
    Returns:
        Dict with aggregate benchmark results
    """
    logger.info(f"Running benchmark suite on {len(resource_paths)} resources")
    
    comparisons = []
    for path in resource_paths:
        if os.path.exists(path):
            comparison = compare_pipelines(path, basic_chunker, enhanced_chunker)
            comparisons.append(comparison)
        else:
            logger.warning(f"Resource not found: {path}")
    
    # Aggregate results
    successful_comparisons = [c for c in comparisons if c.get('success')]
    
    if not successful_comparisons:
        return {
            'success': False,
            'error': 'No successful comparisons',
            'total_resources': len(resource_paths),
            'successful': 0
        }
    
    # Compute averages
    avg_time_multiplier = sum(c['comparison']['time_multiplier'] for c in successful_comparisons) / len(successful_comparisons)
    avg_memory_multiplier = sum(c['comparison']['memory_multiplier'] for c in successful_comparisons) / len(successful_comparisons)
    avg_metadata_multiplier = sum(c['comparison']['metadata_richness_multiplier'] for c in successful_comparisons) / len(successful_comparisons)
    
    time_within_target = sum(1 for c in successful_comparisons if c['comparison']['time_within_target'])
    memory_within_target = sum(1 for c in successful_comparisons if c['comparison']['memory_within_target'])
    
    return {
        'success': True,
        'total_resources': len(resource_paths),
        'successful_comparisons': len(successful_comparisons),
        'failed_comparisons': len(comparisons) - len(successful_comparisons),
        'averages': {
            'time_multiplier': round(avg_time_multiplier, 2),
            'memory_multiplier': round(avg_memory_multiplier, 2),
            'metadata_richness_multiplier': round(avg_metadata_multiplier, 2)
        },
        'targets_met': {
            'time_within_3x': time_within_target,
            'time_within_3x_percentage': round((time_within_target / len(successful_comparisons)) * 100, 2),
            'memory_within_2x': memory_within_target,
            'memory_within_2x_percentage': round((memory_within_target / len(successful_comparisons)) * 100, 2)
        },
        'detailed_results': comparisons
    }


def generate_benchmark_report(benchmark_results: Dict[str, Any],
                             output_file: str = None) -> str:
    """Generate formatted benchmark report.
    
    Args:
        benchmark_results: Results from benchmark_suite
        output_file: Optional file path to write report
        
    Returns:
        Formatted report string
    """
    if not benchmark_results.get('success'):
        return f"Benchmark failed: {benchmark_results.get('error', 'Unknown error')}"
    
    report_lines = [
        "=" * 80,
        "PERFORMANCE BENCHMARK REPORT",
        "=" * 80,
        "",
        "SUMMARY",
        "-" * 80,
        f"Total Resources Tested: {benchmark_results['total_resources']}",
        f"Successful Comparisons: {benchmark_results['successful_comparisons']}",
        f"Failed Comparisons: {benchmark_results['failed_comparisons']}",
        "",
        "AVERAGE PERFORMANCE",
        "-" * 80,
        f"Time Multiplier (Enhanced/Basic): {benchmark_results['averages']['time_multiplier']}x",
        f"Memory Multiplier (Enhanced/Basic): {benchmark_results['averages']['memory_multiplier']}x",
        f"Metadata Richness Multiplier: {benchmark_results['averages']['metadata_richness_multiplier']}x",
        "",
        "TARGETS MET",
        "-" * 80,
        f"Time Within 3x Target: {benchmark_results['targets_met']['time_within_3x']}/{benchmark_results['successful_comparisons']} ({benchmark_results['targets_met']['time_within_3x_percentage']}%)",
        f"Memory Within 2x Target: {benchmark_results['targets_met']['memory_within_2x']}/{benchmark_results['successful_comparisons']} ({benchmark_results['targets_met']['memory_within_2x_percentage']}%)",
        "",
        "DETAILED RESULTS",
        "-" * 80,
    ]
    
    for i, result in enumerate(benchmark_results['detailed_results'], 1):
        if result.get('success'):
            report_lines.extend([
                f"\n{i}. {os.path.basename(result['resource'])}",
                f"   Basic: {result['basic']['execution_time_seconds']}s, {result['basic']['total_chunks']} chunks, {result['basic']['memory_peak_mb']}MB",
                f"   Enhanced: {result['enhanced']['execution_time_seconds']}s, {result['enhanced']['total_chunks']} chunks, {result['enhanced']['memory_peak_mb']}MB",
                f"   Time: {result['comparison']['time_multiplier']}x, Memory: {result['comparison']['memory_multiplier']}x",
                f"   Metadata Richness: {result['comparison']['metadata_richness_multiplier']}x",
                f"   ✅ Targets Met" if result['comparison']['time_within_target'] and result['comparison']['memory_within_target'] else "   ⚠️  Targets Exceeded"
            ])
        else:
            report_lines.append(f"\n{i}. {result.get('resource', 'Unknown')} - FAILED")
    
    report_lines.extend([
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        logger.info(f"Benchmark report written to {output_file}")
    
    return report
