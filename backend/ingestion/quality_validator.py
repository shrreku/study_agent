"""Quality Validation Framework for Enhanced Chunking Pipeline.

This module provides comprehensive quality checks and metrics for validating
the enhanced chunking pipeline output.
"""

from typing import List, Dict, Any, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger("backend.quality_validator")


def check_formula_preservation(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check that formulas are preserved correctly without splits.
    
    Args:
        chunks: List of chunks to validate
        
    Returns:
        Dict with preservation metrics
    """
    total_formulas = 0
    preserved_formulas = 0
    chunks_with_formulas = 0
    
    for chunk in chunks:
        formulas = chunk.get('formulas', [])
        if formulas:
            chunks_with_formulas += 1
            total_formulas += len(formulas)
            # Check each formula has required fields
            for formula in formulas:
                if all(k in formula for k in ['id', 'latex', 'type']):
                    preserved_formulas += 1
    
    preservation_rate = (preserved_formulas / max(total_formulas, 1)) * 100
    
    return {
        'total_formulas': total_formulas,
        'preserved_formulas': preserved_formulas,
        'chunks_with_formulas': chunks_with_formulas,
        'preservation_rate': round(preservation_rate, 2),
        'passed': preservation_rate >= 95.0
    }


def check_tag_hierarchy(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check that hierarchical tags are properly structured.
    
    Args:
        chunks: List of chunks to validate
        
    Returns:
        Dict with hierarchy metrics
    """
    chunks_with_domain = 0
    chunks_with_topic = 0
    chunks_with_subtopic = 0
    chunks_with_complete_hierarchy = 0
    
    for chunk in chunks:
        has_domain = bool(chunk.get('domain'))
        has_topic = bool(chunk.get('topic'))
        has_subtopic = bool(chunk.get('subtopic'))
        
        if has_domain:
            chunks_with_domain += 1
        if has_topic:
            chunks_with_topic += 1
        if has_subtopic:
            chunks_with_subtopic += 1
        if has_domain and has_topic:
            chunks_with_complete_hierarchy += 1
    
    total = len(chunks)
    coverage = (chunks_with_complete_hierarchy / max(total, 1)) * 100
    
    return {
        'total_chunks': total,
        'chunks_with_domain': chunks_with_domain,
        'chunks_with_topic': chunks_with_topic,
        'chunks_with_subtopic': chunks_with_subtopic,
        'chunks_with_complete_hierarchy': chunks_with_complete_hierarchy,
        'coverage_rate': round(coverage, 2),
        'passed': coverage >= 90.0
    }


def check_formula_metadata(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check completeness of formula metadata.
    
    Args:
        chunks: List of chunks to validate
        
    Returns:
        Dict with formula metadata completeness metrics
    """
    total_formulas = 0
    formulas_with_variables = 0
    formulas_with_type = 0
    formulas_complete = 0
    
    for chunk in chunks:
        formulas = chunk.get('formulas', [])
        for formula in formulas:
            total_formulas += 1
            
            has_variables = bool(formula.get('variables'))
            has_type = bool(formula.get('type'))
            has_latex = bool(formula.get('latex'))
            
            if has_variables:
                formulas_with_variables += 1
            if has_type:
                formulas_with_type += 1
            if has_variables and has_type and has_latex:
                formulas_complete += 1
    
    completeness = (formulas_complete / max(total_formulas, 1)) * 100
    
    return {
        'total_formulas': total_formulas,
        'formulas_with_variables': formulas_with_variables,
        'formulas_with_type': formulas_with_type,
        'formulas_complete': formulas_complete,
        'completeness_rate': round(completeness, 2),
        'passed': completeness >= 70.0
    }


def check_prerequisites(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check prerequisite coverage and linking.
    
    Args:
        chunks: List of chunks to validate
        
    Returns:
        Dict with prerequisite metrics
    """
    chunks_with_prerequisites = 0
    total_prerequisites = 0
    chunks_with_prerequisite_links = 0
    total_links = 0
    
    for chunk in chunks:
        prereqs = chunk.get('prerequisites', [])
        if prereqs:
            chunks_with_prerequisites += 1
            total_prerequisites += len(prereqs)
        
        prereq_links = chunk.get('relationships', {}).get('prerequisite_chunk_ids', [])
        if prereq_links:
            chunks_with_prerequisite_links += 1
            total_links += len(prereq_links)
    
    total = len(chunks)
    coverage = (chunks_with_prerequisites / max(total, 1)) * 100
    linking_rate = (chunks_with_prerequisite_links / max(chunks_with_prerequisites, 1)) * 100
    
    return {
        'total_chunks': total,
        'chunks_with_prerequisites': chunks_with_prerequisites,
        'total_prerequisites': total_prerequisites,
        'chunks_with_links': chunks_with_prerequisite_links,
        'total_links': total_links,
        'coverage_rate': round(coverage, 2),
        'linking_rate': round(linking_rate, 2),
        'passed': coverage >= 30.0  # Lower threshold as not all chunks have prereqs
    }


def check_context_windows(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check that context windows are populated.
    
    Args:
        chunks: List of chunks to validate
        
    Returns:
        Dict with context window metrics
    """
    chunks_with_context = 0
    chunks_with_prev_summary = 0
    chunks_with_next_preview = 0
    chunks_with_surrounding = 0
    
    for i, chunk in enumerate(chunks):
        context = chunk.get('context', {})
        if context:
            chunks_with_context += 1
            
            if context.get('previous_chunk_summary') and i > 0:
                chunks_with_prev_summary += 1
            if context.get('next_chunk_preview') and i < len(chunks) - 1:
                chunks_with_next_preview += 1
            if context.get('surrounding_text_before') or context.get('surrounding_text_after'):
                chunks_with_surrounding += 1
    
    total = len(chunks)
    # Exclude first/last for prev/next
    eligible_prev = max(total - 1, 1)
    eligible_next = max(total - 1, 1)
    
    coverage = (chunks_with_context / max(total, 1)) * 100
    prev_rate = (chunks_with_prev_summary / eligible_prev) * 100
    next_rate = (chunks_with_next_preview / eligible_next) * 100
    
    return {
        'total_chunks': total,
        'chunks_with_context': chunks_with_context,
        'chunks_with_prev_summary': chunks_with_prev_summary,
        'chunks_with_next_preview': chunks_with_next_preview,
        'chunks_with_surrounding': chunks_with_surrounding,
        'coverage_rate': round(coverage, 2),
        'prev_summary_rate': round(prev_rate, 2),
        'next_preview_rate': round(next_rate, 2),
        'passed': coverage >= 95.0
    }


def check_relationships(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check that chunks are properly linked with relationships.
    
    Args:
        chunks: List of chunks to validate
        
    Returns:
        Dict with relationship metrics
    """
    chunks_with_relationships = 0
    chunks_with_prev = 0
    chunks_with_next = 0
    chunks_with_continuity = 0
    chunks_with_sequence = 0
    
    for i, chunk in enumerate(chunks):
        relationships = chunk.get('relationships', {})
        if relationships:
            chunks_with_relationships += 1
            
            if relationships.get('previous_chunk_id') and i > 0:
                chunks_with_prev += 1
            if relationships.get('next_chunk_id') and i < len(chunks) - 1:
                chunks_with_next += 1
        
        if chunk.get('continuity'):
            chunks_with_continuity += 1
        if chunk.get('sequence'):
            chunks_with_sequence += 1
    
    total = len(chunks)
    eligible_prev = max(total - 1, 1)
    eligible_next = max(total - 1, 1)
    
    linking_rate = (chunks_with_relationships / max(total, 1)) * 100
    prev_rate = (chunks_with_prev / eligible_prev) * 100
    next_rate = (chunks_with_next / eligible_next) * 100
    
    return {
        'total_chunks': total,
        'chunks_with_relationships': chunks_with_relationships,
        'chunks_with_prev': chunks_with_prev,
        'chunks_with_next': chunks_with_next,
        'chunks_with_continuity': chunks_with_continuity,
        'chunks_with_sequence': chunks_with_sequence,
        'linking_rate': round(linking_rate, 2),
        'prev_link_rate': round(prev_rate, 2),
        'next_link_rate': round(next_rate, 2),
        'passed': linking_rate >= 90.0
    }


def compute_metadata_richness_score(chunk: Dict[str, Any]) -> float:
    """Compute a richness score for chunk metadata.
    
    Args:
        chunk: Single chunk to score
        
    Returns:
        Richness score (0-100)
    """
    score = 0
    max_score = 100
    
    # Basic fields (10 points)
    if chunk.get('full_text'):
        score += 10
    
    # Hierarchical taxonomy (20 points)
    if chunk.get('domain'):
        score += 7
    if chunk.get('topic'):
        score += 7
    if chunk.get('subtopic'):
        score += 6
    
    # Prerequisites and learning objectives (15 points)
    if chunk.get('prerequisites'):
        score += 8
    if chunk.get('learning_objectives'):
        score += 7
    
    # Formula metadata (15 points)
    formulas = chunk.get('formulas', [])
    if formulas:
        score += 10
        if any(f.get('variables') for f in formulas):
            score += 5
    
    # Complexity metrics (10 points)
    if chunk.get('complexity'):
        score += 10
    
    # Context windows (10 points)
    if chunk.get('context'):
        score += 10
    
    # Relationships (10 points)
    if chunk.get('relationships'):
        score += 10
    
    # Continuity and sequence (10 points)
    if chunk.get('continuity'):
        score += 5
    if chunk.get('sequence'):
        score += 5
    
    return min(score, max_score)


def validate_chunk_quality(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate overall chunk quality with comprehensive checks.
    
    Args:
        chunks: List of chunks to validate
        
    Returns:
        Dict with complete quality report
    """
    logger.info(f"Validating quality for {len(chunks)} chunks")
    
    checks = {
        'formulas_preserved': check_formula_preservation(chunks),
        'tags_hierarchical': check_tag_hierarchy(chunks),
        'formulas_complete': check_formula_metadata(chunks),
        'prerequisites_present': check_prerequisites(chunks),
        'context_windows': check_context_windows(chunks),
        'relationships_linked': check_relationships(chunks)
    }
    
    # Compute metadata richness scores
    richness_scores = [compute_metadata_richness_score(c) for c in chunks]
    avg_richness = sum(richness_scores) / max(len(richness_scores), 1)
    
    # Overall pass/fail
    all_passed = all(check['passed'] for check in checks.values())
    
    # Summary
    summary = {
        'total_chunks': len(chunks),
        'all_checks_passed': all_passed,
        'average_metadata_richness': round(avg_richness, 2),
        'checks_passed': sum(1 for c in checks.values() if c['passed']),
        'checks_failed': sum(1 for c in checks.values() if not c['passed']),
        'quality_grade': _compute_quality_grade(checks, avg_richness)
    }
    
    return {
        'summary': summary,
        'checks': checks,
        'richness_distribution': {
            'min': round(min(richness_scores) if richness_scores else 0, 2),
            'max': round(max(richness_scores) if richness_scores else 0, 2),
            'avg': round(avg_richness, 2)
        }
    }


def _compute_quality_grade(checks: Dict[str, Dict], avg_richness: float) -> str:
    """Compute overall quality grade.
    
    Args:
        checks: Dict of check results
        avg_richness: Average metadata richness score
        
    Returns:
        Grade string (A+, A, B, C, D, F)
    """
    passed_count = sum(1 for c in checks.values() if c['passed'])
    total_count = len(checks)
    pass_rate = (passed_count / total_count) * 100
    
    # Combine pass rate and richness
    combined_score = (pass_rate * 0.6) + (avg_richness * 0.4)
    
    if combined_score >= 95:
        return 'A+'
    elif combined_score >= 90:
        return 'A'
    elif combined_score >= 80:
        return 'B'
    elif combined_score >= 70:
        return 'C'
    elif combined_score >= 60:
        return 'D'
    else:
        return 'F'


def generate_quality_report(chunks: List[Dict[str, Any]], 
                           output_file: str = None) -> str:
    """Generate a formatted quality report.
    
    Args:
        chunks: List of chunks to validate
        output_file: Optional file path to write report
        
    Returns:
        Formatted report string
    """
    validation = validate_chunk_quality(chunks)
    
    report_lines = [
        "=" * 80,
        "CHUNK QUALITY VALIDATION REPORT",
        "=" * 80,
        "",
        "SUMMARY",
        "-" * 80,
        f"Total Chunks: {validation['summary']['total_chunks']}",
        f"All Checks Passed: {validation['summary']['all_checks_passed']}",
        f"Quality Grade: {validation['summary']['quality_grade']}",
        f"Average Metadata Richness: {validation['summary']['average_metadata_richness']}/100",
        f"Checks Passed: {validation['summary']['checks_passed']}/{validation['summary']['checks_passed'] + validation['summary']['checks_failed']}",
        "",
        "DETAILED CHECKS",
        "-" * 80,
    ]
    
    for check_name, check_result in validation['checks'].items():
        status = "✅ PASS" if check_result['passed'] else "❌ FAIL"
        report_lines.append(f"\n{check_name.upper().replace('_', ' ')}: {status}")
        for key, value in check_result.items():
            if key != 'passed':
                report_lines.append(f"  {key}: {value}")
    
    report_lines.extend([
        "",
        "METADATA RICHNESS DISTRIBUTION",
        "-" * 80,
        f"Minimum: {validation['richness_distribution']['min']}/100",
        f"Maximum: {validation['richness_distribution']['max']}/100",
        f"Average: {validation['richness_distribution']['avg']}/100",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        logger.info(f"Quality report written to {output_file}")
    
    return report
