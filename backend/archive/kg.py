"""Knowledge graph utility re-exports."""

from .kg_base import canonicalize_concept, ensure_neo4j_constraints, managed_driver
from .kg_concepts import merge_concepts_in_neo4j
from kg_pipeline.writer import (
    link_chunk_to_section,
    merge_chunk_figures,
    merge_chunk_formulas,
    merge_chunk_pedagogy_relations,
    merge_next_chunk,
)
from .kg_relationships import merge_alias, merge_prerequisite_edge, merge_related_concepts

__all__ = [
    "canonicalize_concept",
    "ensure_neo4j_constraints",
    "managed_driver",
    "merge_concepts_in_neo4j",
    "merge_related_concepts",
    "merge_prerequisite_edge",
    "merge_alias",
    "link_chunk_to_section",
    "merge_chunk_figures",
    "merge_chunk_formulas",
    "merge_chunk_pedagogy_relations",
    "merge_next_chunk",
]
