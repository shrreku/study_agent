"""Compatibility wrappers forwarding to the kg_pipeline writer utilities."""
from kg_pipeline.writer import (
    merge_section_node,
    link_chunk_to_section,
    merge_chunk_figures,
    merge_chunk_formulas,
    merge_chunk_pedagogy_relations,
    merge_next_chunk,
)

__all__ = [
    "merge_section_node",
    "link_chunk_to_section",
    "merge_chunk_figures",
    "merge_chunk_formulas",
    "merge_chunk_pedagogy_relations",
    "merge_next_chunk",
]
