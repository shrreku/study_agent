"""
Cross-book ontology builder and consolidation utilities.

This module handles:
- Concept clustering across multiple books
- Canonical name resolution
- DomainConcept node creation
- Noise detection and cleanup
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import ConceptType, DomainConcept, OntologyEdge

logger = logging.getLogger("backend.ingestion.v2.ontology")


# =============================================================================
# Concept Clustering
# =============================================================================

@dataclass
class ConceptOccurrence:
    """A single occurrence of a concept in a book."""
    name: str
    canonical_name: str
    book_id: str
    concept_type: ConceptType
    frequency: int
    definition: str = ""
    context_snippets: List[str] = field(default_factory=list)


@dataclass
class ConceptCluster:
    """A cluster of potentially related concept names."""
    cluster_id: str
    concept_names: Set[str] = field(default_factory=set)
    occurrences: List[ConceptOccurrence] = field(default_factory=list)
    
    # Resolved canonical form
    canonical_name: Optional[str] = None
    display_name: Optional[str] = None
    concept_type: Optional[ConceptType] = None
    definition: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    @property
    def total_frequency(self) -> int:
        return sum(o.frequency for o in self.occurrences)
    
    @property
    def book_count(self) -> int:
        return len(set(o.book_id for o in self.occurrences))


class ConceptClusterer:
    """Clusters similar concept names across books."""
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        use_llm_for_ambiguous: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.use_llm = use_llm_for_ambiguous
        self.clusters: Dict[str, ConceptCluster] = {}
        self._name_to_cluster: Dict[str, str] = {}
    
    def add_concepts(self, book_id: str, concepts: Dict[str, Any]):
        """Add concepts from a book to the clustering."""
        for canonical, data in concepts.items():
            occurrence = ConceptOccurrence(
                name=data.get("display_name", canonical),
                canonical_name=canonical,
                book_id=book_id,
                concept_type=ConceptType(data.get("concept_type", "unknown")),
                frequency=data.get("frequency", 1),
                definition=data.get("definition", ""),
            )
            self._add_occurrence(occurrence)
    
    def _add_occurrence(self, occurrence: ConceptOccurrence):
        """Add a single occurrence, clustering as needed."""
        canonical = occurrence.canonical_name
        
        # Check if exact match exists
        if canonical in self._name_to_cluster:
            cluster_id = self._name_to_cluster[canonical]
            self.clusters[cluster_id].occurrences.append(occurrence)
            return
        
        # Find similar cluster
        best_cluster = None
        best_sim = 0.0
        
        for cluster_id, cluster in self.clusters.items():
            for name in cluster.concept_names:
                sim = self._similarity(canonical, name)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cluster_id
        
        if best_sim >= self.similarity_threshold and best_cluster:
            self.clusters[best_cluster].concept_names.add(canonical)
            self.clusters[best_cluster].occurrences.append(occurrence)
            self._name_to_cluster[canonical] = best_cluster
        else:
            # Create new cluster
            cluster_id = f"cluster_{len(self.clusters)}"
            self.clusters[cluster_id] = ConceptCluster(
                cluster_id=cluster_id,
                concept_names={canonical},
                occurrences=[occurrence],
            )
            self._name_to_cluster[canonical] = cluster_id
    
    def _similarity(self, a: str, b: str) -> float:
        """Compute similarity between two concept names."""
        a_lower = a.lower().replace("_", " ").replace("-", " ")
        b_lower = b.lower().replace("_", " ").replace("-", " ")
        
        # Exact match
        if a_lower == b_lower:
            return 1.0
        
        # One is substring of other
        if a_lower in b_lower or b_lower in a_lower:
            return 0.9
        
        # Sequence matching
        return SequenceMatcher(None, a_lower, b_lower).ratio()
    
    def resolve_clusters(self) -> List[ConceptCluster]:
        """Resolve all clusters to canonical forms."""
        resolved = []
        
        for cluster in self.clusters.values():
            if len(cluster.concept_names) == 1:
                # Single concept, use as-is
                occ = cluster.occurrences[0]
                cluster.canonical_name = occ.canonical_name
                cluster.display_name = occ.name
                cluster.concept_type = occ.concept_type
                cluster.definition = occ.definition
                cluster.confidence = 0.95
            elif self.use_llm and len(cluster.concept_names) > 1:
                # Use LLM to resolve
                self._resolve_with_llm(cluster)
            else:
                # Heuristic resolution
                self._resolve_heuristic(cluster)
            
            resolved.append(cluster)
        
        return resolved
    
    def _resolve_heuristic(self, cluster: ConceptCluster):
        """Resolve cluster using heuristics."""
        # Pick most frequent name
        name_freq = defaultdict(int)
        for occ in cluster.occurrences:
            name_freq[occ.canonical_name] += occ.frequency
        
        canonical = max(name_freq, key=name_freq.get)
        
        # Find best definition
        best_def = ""
        for occ in cluster.occurrences:
            if occ.canonical_name == canonical and occ.definition:
                best_def = occ.definition
                break
        
        # Most common type
        type_freq = defaultdict(int)
        for occ in cluster.occurrences:
            type_freq[occ.concept_type] += occ.frequency
        
        cluster.canonical_name = canonical
        cluster.display_name = canonical.replace("_", " ").title()
        cluster.concept_type = max(type_freq, key=type_freq.get)
        cluster.definition = best_def
        cluster.aliases = [n for n in cluster.concept_names if n != canonical]
        cluster.confidence = 0.7
    
    def _resolve_with_llm(self, cluster: ConceptCluster):
        """Resolve cluster using LLM."""
        try:
            import yaml
            prompts_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "prompts", "ingest_v2.yaml"
            )
            with open(prompts_path) as f:
                prompts = yaml.safe_load(f)
            template = prompts.get("ontology", {}).get("cluster_concepts", "")
        except Exception:
            self._resolve_heuristic(cluster)
            return
        
        if not template:
            self._resolve_heuristic(cluster)
            return
        
        # Build prompt
        concept_list = ", ".join(sorted(cluster.concept_names))
        sample_occs = []
        for occ in cluster.occurrences[:5]:
            sample_occs.append(f"- {occ.name} ({occ.book_id}): {occ.definition[:100]}")
        
        prompt = template.replace("{{concept_cluster}}", concept_list)
        prompt = prompt.replace("{{sample_occurrences}}", "\n".join(sample_occs))
        
        try:
            from llm.common import call_json_chat
            result = call_json_chat(prompt, default={})
            
            if result.get("same_concept", True):
                cluster.canonical_name = result.get("canonical_name", list(cluster.concept_names)[0])
                cluster.display_name = result.get("display_name", cluster.canonical_name.title())
                try:
                    cluster.concept_type = ConceptType(result.get("concept_type", "unknown"))
                except ValueError:
                    cluster.concept_type = ConceptType.UNKNOWN
                cluster.definition = result.get("definition", "")
                cluster.aliases = result.get("aliases", [])
                cluster.confidence = result.get("confidence", 0.8)
            else:
                # Split cluster (TODO: implement)
                self._resolve_heuristic(cluster)
        except Exception as e:
            logger.warning(f"LLM cluster resolution failed: {e}")
            self._resolve_heuristic(cluster)


# =============================================================================
# Ontology Builder
# =============================================================================

class OntologyBuilder:
    """Builds cross-book domain ontology."""
    
    def __init__(self, domain: str = "STEM"):
        self.domain = domain
        self.domain_concepts: Dict[str, DomainConcept] = {}
        self.edges: List[OntologyEdge] = []
        self.clusterer = ConceptClusterer()
    
    def add_book_concepts(self, book_id: str, concepts: Dict[str, Any]):
        """Add concepts from a book."""
        self.clusterer.add_concepts(book_id, concepts)
    
    def build(self) -> Tuple[Dict[str, DomainConcept], List[OntologyEdge]]:
        """Build the ontology from all added books."""
        # Resolve clusters
        clusters = self.clusterer.resolve_clusters()
        
        # Create domain concepts
        for cluster in clusters:
            if cluster.canonical_name:
                dc = DomainConcept(
                    canonical_name=cluster.canonical_name,
                    display_name=cluster.display_name or cluster.canonical_name,
                    concept_type=cluster.concept_type or ConceptType.UNKNOWN,
                    domain=self.domain,
                    aliases=cluster.aliases,
                    definition=cluster.definition or "",
                    source_books=[o.book_id for o in cluster.occurrences],
                )
                self.domain_concepts[dc.canonical_name] = dc
        
        # Build hierarchy edges
        self._build_hierarchy()
        
        return self.domain_concepts, self.edges
    
    def _build_hierarchy(self):
        """Build IS_A and PART_OF hierarchy using LLM."""
        if len(self.domain_concepts) < 5:
            return
        
        try:
            import yaml
            prompts_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "prompts", "ingest_v2.yaml"
            )
            with open(prompts_path) as f:
                prompts = yaml.safe_load(f)
            template = prompts.get("ontology", {}).get("build_hierarchy", "")
        except Exception:
            return
        
        if not template:
            return
        
        # Build concept list
        concepts_str = "\n".join(
            f"- {dc.display_name} ({dc.concept_type.value})"
            for dc in list(self.domain_concepts.values())[:50]
        )
        
        prompt = template.replace("{{concepts_with_types}}", concepts_str)
        
        try:
            from llm.common import call_json_chat
            result = call_json_chat(prompt, default={"is_a_edges": [], "part_of_edges": []})
            
            for edge in result.get("is_a_edges", []):
                child = edge.get("child", "").lower()
                parent = edge.get("parent", "").lower()
                if child in self.domain_concepts and parent in self.domain_concepts:
                    self.edges.append(OntologyEdge(
                        source=child, target=parent,
                        relation_type="IS_A",
                        confidence=edge.get("confidence", 0.8),
                    ))
            
            for edge in result.get("part_of_edges", []):
                part = edge.get("part", "").lower()
                whole = edge.get("whole", "").lower()
                if part in self.domain_concepts and whole in self.domain_concepts:
                    self.edges.append(OntologyEdge(
                        source=part, target=whole,
                        relation_type="PART_OF",
                        confidence=edge.get("confidence", 0.8),
                    ))
        except Exception as e:
            logger.warning(f"Hierarchy building failed: {e}")


# =============================================================================
# Noise Detection
# =============================================================================

class NoiseDetector:
    """Detects and filters noisy concepts."""
    
    # Known noise patterns
    NOISE_PATTERNS = [
        r"^(eq|eqn|fig|table|chapter|section|example|problem)\s*\d*$",
        r"^\d+(\.\d+)*$",
        r"^[a-z]$",
        r"^(the|a|an|this|that|it|we|you)$",
    ]
    
    # Generic terms that are usually noise
    GENERIC_TERMS = {
        "example", "method", "equation", "formula", "expression",
        "result", "solution", "value", "case", "condition",
        "system", "process", "approach", "technique", "problem",
    }
    
    def __init__(self, min_frequency: int = 2, min_confidence: float = 0.5):
        self.min_frequency = min_frequency
        self.min_confidence = min_confidence
        import re
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.NOISE_PATTERNS]
    
    def is_noisy(
        self,
        name: str,
        frequency: int,
        confidence: float,
        cooccurring: List[str] = None,
    ) -> Tuple[bool, str]:
        """Check if a concept is likely noise."""
        name_lower = name.lower().strip()
        
        # Pattern match
        for pattern in self._patterns:
            if pattern.match(name_lower):
                return True, "matches_noise_pattern"
        
        # Generic term without context
        if name_lower in self.GENERIC_TERMS:
            return True, "generic_term"
        
        # Low frequency + low confidence
        if frequency < self.min_frequency and confidence < self.min_confidence:
            return True, "low_frequency_confidence"
        
        # Single character or too short
        if len(name_lower) <= 2:
            return True, "too_short"
        
        return False, ""
    
    def filter_concepts(
        self,
        concepts: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Filter noisy concepts, return (clean, removed)."""
        clean = {}
        removed = []
        
        for name, data in concepts.items():
            is_noise, reason = self.is_noisy(
                name,
                data.get("frequency", 1),
                data.get("confidence", 0.5),
            )
            if is_noise:
                removed.append(name)
                logger.debug(f"Filtered noise: {name} ({reason})")
            else:
                clean[name] = data
        
        return clean, removed


# =============================================================================
# Neo4j Ontology Writer
# =============================================================================

class Neo4jOntologyWriter:
    """Writes ontology to Neo4j."""
    
    def __init__(self):
        self._driver = None
    
    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        return self._driver
    
    def write_ontology(
        self,
        domain_concepts: Dict[str, DomainConcept],
        edges: List[OntologyEdge],
    ):
        """Write domain ontology to Neo4j."""
        driver = self._get_driver()
        
        with driver.session() as session:
            # Create DomainConcept nodes
            concepts = [dc.to_dict() for dc in domain_concepts.values()]
            session.execute_write(lambda tx: tx.run("""
                UNWIND $concepts AS c
                MERGE (dc:DomainConcept {canonical_name: c.canonical_name})
                SET dc.display_name = c.display_name,
                    dc.concept_type = c.concept_type,
                    dc.domain = c.domain,
                    dc.definition = c.definition,
                    dc.aliases = c.aliases,
                    dc.source_books = c.source_books
            """, concepts=concepts))
            
            # Create MAPS_TO edges (book concept -> domain concept)
            for dc in domain_concepts.values():
                maps_to = [{"book_concept": dc.canonical_name, "domain_concept": dc.canonical_name}]
                for alias in dc.aliases:
                    maps_to.append({"book_concept": alias, "domain_concept": dc.canonical_name})
                
                session.execute_write(lambda tx: tx.run("""
                    UNWIND $maps AS m
                    MATCH (bc:Concept {canonical_name: m.book_concept})
                    MATCH (dc:DomainConcept {canonical_name: m.domain_concept})
                    MERGE (bc)-[:MAPS_TO]->(dc)
                """, maps=maps_to))
            
            # Create ontology edges
            is_a = [e for e in edges if e.relation_type == "IS_A"]
            part_of = [e for e in edges if e.relation_type == "PART_OF"]
            
            if is_a:
                session.execute_write(lambda tx: tx.run("""
                    UNWIND $edges AS e
                    MATCH (child:DomainConcept {canonical_name: e.source})
                    MATCH (parent:DomainConcept {canonical_name: e.target})
                    MERGE (child)-[r:IS_A]->(parent)
                    SET r.confidence = e.confidence
                """, edges=[{"source": e.source, "target": e.target, "confidence": e.confidence} for e in is_a]))
            
            if part_of:
                session.execute_write(lambda tx: tx.run("""
                    UNWIND $edges AS e
                    MATCH (part:DomainConcept {canonical_name: e.source})
                    MATCH (whole:DomainConcept {canonical_name: e.target})
                    MERGE (part)-[r:PART_OF]->(whole)
                    SET r.confidence = e.confidence
                """, edges=[{"source": e.source, "target": e.target, "confidence": e.confidence} for e in part_of]))
        
        logger.info(f"Wrote ontology: {len(domain_concepts)} domain concepts, {len(edges)} edges")
