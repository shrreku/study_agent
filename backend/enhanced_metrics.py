"""Enhanced Metrics Collection for Chunking Pipeline.

This module provides comprehensive metrics collection for monitoring
the enhanced chunking pipeline in production.
"""

from typing import Dict, Any, List, Optional
import time
import logging
from contextlib import contextmanager
from metrics import MetricsCollector

logger = logging.getLogger("backend.enhanced_metrics")


class EnhancedPipelineMetrics:
    """Comprehensive metrics collector for enhanced chunking pipeline."""
    
    def __init__(self):
        self.mc = MetricsCollector.get_global()
        self._stage_timers = {}
        self._llm_call_count = 0
        self._formula_count = 0
        self._tag_count = 0
    
    @contextmanager
    def time_stage(self, stage_name: str):
        """Context manager to time a pipeline stage.
        
        Args:
            stage_name: Name of the stage to time
        """
        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self.mc.timing(f"stage_{stage_name}_time_ms", duration_ms)
            self._stage_timers[stage_name] = duration_ms
    
    def record_page_extraction(self, page_count: int, duration_ms: float):
        """Record page extraction metrics."""
        self.mc.timing("page_extraction_time_ms", duration_ms)
        self.mc.gauge("pages_extracted", page_count)
        if page_count > 0:
            self.mc.gauge("time_per_page_ms", duration_ms / page_count)
    
    def record_semantic_chunking(self, chunk_count: int, duration_ms: float, 
                                 avg_chunk_size: float):
        """Record semantic chunking metrics."""
        self.mc.timing("semantic_chunking_time_ms", duration_ms)
        self.mc.gauge("chunks_created", chunk_count)
        self.mc.gauge("avg_chunk_size_tokens", avg_chunk_size)
        if chunk_count > 0:
            self.mc.gauge("time_per_chunk_ms", duration_ms / chunk_count)
    
    def record_hierarchical_tagging(self, chunk_count: int, duration_ms: float,
                                   tags_extracted: int):
        """Record hierarchical tagging metrics."""
        self.mc.timing("hierarchical_tagging_time_ms", duration_ms)
        self.mc.increment("tags_extracted", tags_extracted)
        self.mc.gauge("tags_per_chunk", tags_extracted / max(chunk_count, 1))
        self._tag_count += tags_extracted
    
    def record_formula_extraction(self, formula_count: int, duration_ms: float,
                                  complete_formulas: int):
        """Record formula extraction metrics."""
        self.mc.timing("formula_extraction_time_ms", duration_ms)
        self.mc.increment("formulas_extracted", formula_count)
        self.mc.increment("formulas_complete", complete_formulas)
        if formula_count > 0:
            completeness_rate = (complete_formulas / formula_count) * 100
            self.mc.gauge("formula_completeness_rate", completeness_rate)
        self._formula_count += formula_count
    
    def record_context_building(self, chunk_count: int, duration_ms: float,
                                context_coverage: float):
        """Record context building metrics."""
        self.mc.timing("context_building_time_ms", duration_ms)
        self.mc.gauge("context_coverage_rate", context_coverage)
    
    def record_chunk_linking(self, chunk_count: int, duration_ms: float,
                            relationship_count: int):
        """Record chunk linking metrics."""
        self.mc.timing("chunk_linking_time_ms", duration_ms)
        self.mc.increment("relationships_created", relationship_count)
        self.mc.gauge("relationships_per_chunk", relationship_count / max(chunk_count, 1))
    
    def record_llm_call(self, duration_ms: float, success: bool, 
                       token_count: Optional[int] = None):
        """Record LLM API call metrics."""
        self.mc.timing("llm_call_latency_ms", duration_ms)
        self.mc.increment("llm_calls_total")
        if success:
            self.mc.increment("llm_calls_success")
        else:
            self.mc.increment("llm_calls_failure")
        if token_count:
            self.mc.gauge("llm_tokens_used", token_count)
        self._llm_call_count += 1
    
    def record_quality_metrics(self, quality_report: Dict[str, Any]):
        """Record quality validation metrics."""
        checks = quality_report.get('checks', {})
        
        # Formula preservation
        formula_check = checks.get('formulas_preserved', {})
        if formula_check:
            self.mc.gauge("formula_preservation_rate", 
                         formula_check.get('preservation_rate', 0))
        
        # Tag hierarchy
        tag_check = checks.get('tags_hierarchical', {})
        if tag_check:
            self.mc.gauge("tag_hierarchy_coverage", 
                         tag_check.get('coverage_rate', 0))
        
        # Formula completeness
        formula_meta_check = checks.get('formulas_complete', {})
        if formula_meta_check:
            self.mc.gauge("formula_metadata_completeness", 
                         formula_meta_check.get('completeness_rate', 0))
        
        # Relationships
        rel_check = checks.get('relationships_linked', {})
        if rel_check:
            self.mc.gauge("relationship_linking_rate", 
                         rel_check.get('linking_rate', 0))
        
        # Overall quality
        summary = quality_report.get('summary', {})
        self.mc.gauge("metadata_richness_avg", 
                     summary.get('average_metadata_richness', 0))
    
    def record_error(self, stage: str, error_type: str):
        """Record pipeline errors."""
        self.mc.increment(f"error_{stage}")
        self.mc.increment(f"error_type_{error_type}")
    
    def record_cost_estimate(self, llm_cost: float, storage_cost: float,
                            compute_cost: float):
        """Record estimated costs."""
        self.mc.gauge("llm_api_cost_usd", llm_cost)
        self.mc.gauge("storage_cost_usd", storage_cost)
        self.mc.gauge("compute_cost_usd", compute_cost)
        total_cost = llm_cost + storage_cost + compute_cost
        self.mc.gauge("total_cost_usd", total_cost)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        return {
            'stage_timers': self._stage_timers,
            'llm_calls': self._llm_call_count,
            'formulas_extracted': self._formula_count,
            'tags_extracted': self._tag_count
        }


def estimate_llm_cost(token_count: int, model: str = "gpt-3.5-turbo") -> float:
    """Estimate LLM API cost based on token count.
    
    Args:
        token_count: Number of tokens used
        model: Model name
        
    Returns:
        Estimated cost in USD
    """
    # Pricing as of 2024 (approximate)
    pricing = {
        "gpt-3.5-turbo": 0.002 / 1000,  # $0.002 per 1K tokens
        "gpt-4": 0.03 / 1000,            # $0.03 per 1K tokens
        "gpt-4-turbo": 0.01 / 1000       # $0.01 per 1K tokens
    }
    
    rate = pricing.get(model, 0.002 / 1000)
    return token_count * rate


def estimate_storage_cost(metadata_size_mb: float) -> float:
    """Estimate storage cost for metadata.
    
    Args:
        metadata_size_mb: Size of metadata in MB
        
    Returns:
        Monthly storage cost in USD
    """
    # Assuming $0.023 per GB/month (AWS S3 pricing)
    cost_per_gb_month = 0.023
    return (metadata_size_mb / 1024) * cost_per_gb_month


def check_alert_conditions(metrics_summary: Dict[str, Any],
                           quality_report: Dict[str, Any]) -> List[Dict[str, str]]:
    """Check for alerting conditions.
    
    Args:
        metrics_summary: Summary of metrics
        quality_report: Quality validation report
        
    Returns:
        List of alerts to trigger
    """
    alerts = []
    
    # Check processing time
    total_time = sum(metrics_summary.get('stage_timers', {}).values())
    if total_time > 15000:  # >15 seconds is concerning
        alerts.append({
            'severity': 'warning',
            'metric': 'processing_time',
            'message': f'Processing time {total_time:.0f}ms exceeds threshold',
            'value': total_time
        })
    
    # Check formula preservation
    checks = quality_report.get('checks', {})
    formula_check = checks.get('formulas_preserved', {})
    if formula_check and formula_check.get('preservation_rate', 100) < 80:
        alerts.append({
            'severity': 'critical',
            'metric': 'formula_preservation',
            'message': f"Formula preservation rate {formula_check['preservation_rate']}% below 80%",
            'value': formula_check['preservation_rate']
        })
    
    # Check tag completeness
    tag_check = checks.get('tags_hierarchical', {})
    if tag_check and tag_check.get('coverage_rate', 100) < 70:
        alerts.append({
            'severity': 'warning',
            'metric': 'tag_completeness',
            'message': f"Tag coverage {tag_check['coverage_rate']}% below 70%",
            'value': tag_check['coverage_rate']
        })
    
    return alerts
