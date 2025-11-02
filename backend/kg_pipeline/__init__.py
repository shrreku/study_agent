"""Utilities and writers for the knowledge-graph pipeline."""

# Base utilities
from .base import (
    canonicalize_concept,
    count_occurrences,
    ensure_neo4j_constraints,
    managed_driver,
)

# Concept operations
from .concepts import merge_concepts_in_neo4j

# Relationship operations
from .relationships import (
    merge_alias,
    merge_prerequisite_edge,
    merge_related_concepts,
)

# Writer operations
from .writer import (
    link_chunk_to_section,
    merge_chunk_figures,
    merge_chunk_formulas,
    merge_chunk_formulas_enhanced,
    merge_chunk_pedagogy_relations,
    merge_next_chunk,
    merge_section_node,
)

__all__ = [
    # Base
    "canonicalize_concept",
    "count_occurrences",
    "ensure_neo4j_constraints",
    "managed_driver",
    # Concepts
    "merge_concepts_in_neo4j",
    # Relationships
    "merge_alias",
    "merge_prerequisite_edge",
    "merge_related_concepts",
    # Writer
    "link_chunk_to_section",
    "merge_chunk_figures",
    "merge_chunk_formulas",
    "merge_chunk_formulas_enhanced",
    "merge_chunk_pedagogy_relations",
    "merge_next_chunk",
    "merge_section_node",
]
