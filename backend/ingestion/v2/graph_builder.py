"""
Knowledge Graph Builder for the v2 ingestion pipeline.

Builds and maintains:
- Per-book pedagogical graph (Concept, Chunk nodes + relationships)
- Cross-book domain ontology (DomainConcept nodes + canonical edges)
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import (
    ConceptType,
    DomainConcept,
    EnrichedChunk,
    ExtractedConcept,
    LocalRelation,
    OntologyEdge,
    RelationType,
)

logger = logging.getLogger("backend.ingestion.v2.graph_builder")


# =============================================================================
# Graph Data Structures
# =============================================================================

@dataclass
class ConceptNode:
    """Aggregated concept data for a single book."""
    canonical_name: str
    display_name: str
    concept_type: ConceptType = ConceptType.UNKNOWN
    
    # Aggregated metadata
    frequency: int = 0
    first_page: int = 0
    chapters_seen: Set[str] = field(default_factory=set)
    pedagogy_roles: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    difficulties: List[str] = field(default_factory=list)
    
    # Best definition found
    definition_text: str = ""
    definition_confidence: float = 0.0
    
    # Source chunks
    chunk_ids: List[str] = field(default_factory=list)
    
    # Quality
    confidence: float = 0.5
    is_noisy: bool = False
    
    @property
    def dominant_difficulty(self) -> str:
        if not self.difficulties:
            return "intermediate"
        counts = defaultdict(int)
        for d in self.difficulties:
            counts[d] += 1
        return max(counts, key=counts.get)
    
    @property
    def primary_role(self) -> str:
        if not self.pedagogy_roles:
            return "explanation"
        return max(self.pedagogy_roles, key=self.pedagogy_roles.get)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "display_name": self.display_name,
            "concept_type": self.concept_type.value,
            "frequency": self.frequency,
            "first_page": self.first_page,
            "difficulty": self.dominant_difficulty,
            "primary_role": self.primary_role,
            "definition": self.definition_text[:500] if self.definition_text else "",
            "confidence": self.confidence,
            "is_noisy": self.is_noisy,
        }


@dataclass
class RelationshipData:
    """A relationship to be stored in the graph."""
    source: str
    target: str
    relation_type: RelationType
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    source_book: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type.value,
            "confidence": self.confidence,
            "source_book": self.source_book,
        }


@dataclass
class BookGraphData:
    """Aggregated graph data for a single book."""
    resource_id: str
    book_id: str
    
    concepts: Dict[str, ConceptNode] = field(default_factory=dict)
    chunks: List[EnrichedChunk] = field(default_factory=list)
    
    occurs_in: List[Tuple[str, str, str]] = field(default_factory=list)
    teaches: List[Tuple[str, str]] = field(default_factory=list)
    prerequisites: List[RelationshipData] = field(default_factory=list)
    is_a: List[RelationshipData] = field(default_factory=list)
    part_of: List[RelationshipData] = field(default_factory=list)
    related_to: List[Tuple[str, str, float]] = field(default_factory=list)


# =============================================================================
# Per-Book Graph Builder
# =============================================================================

class BookGraphBuilder:
    """Builds pedagogical knowledge graph for a single book."""
    
    def __init__(
        self,
        resource_id: str,
        book_id: str = "",
        min_cooccurrence: int = 3,
        noise_frequency_threshold: int = 1,
    ):
        self.resource_id = resource_id
        self.book_id = book_id or resource_id
        self.min_cooccurrence = min_cooccurrence
        self.noise_frequency_threshold = noise_frequency_threshold
        
        self.graph_data = BookGraphData(
            resource_id=resource_id,
            book_id=self.book_id,
        )
        self._cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
    
    def add_chunk(self, chunk: EnrichedChunk):
        """Add a chunk and its concepts to the graph."""
        self.graph_data.chunks.append(chunk)
        
        all_concepts = chunk.all_concepts
        chunk_concept_names = []
        
        for concept in all_concepts:
            canonical = concept.canonical_name
            chunk_concept_names.append(canonical)
            
            if canonical not in self.graph_data.concepts:
                self.graph_data.concepts[canonical] = ConceptNode(
                    canonical_name=canonical,
                    display_name=concept.name,
                    concept_type=concept.concept_type,
                    first_page=chunk.page_number,
                )
            
            node = self.graph_data.concepts[canonical]
            node.frequency += 1
            node.chunk_ids.append(chunk.id)
            node.pedagogy_roles[chunk.pedagogy_role] += 1
            node.difficulties.append(chunk.difficulty)
            
            if chunk.chapter_title:
                node.chapters_seen.add(chunk.chapter_title)
            
            if concept.is_main_concept:
                node.confidence = max(node.confidence, concept.confidence)
            
            if concept.definition_text and concept.confidence > node.definition_confidence:
                node.definition_text = concept.definition_text
                node.definition_confidence = concept.confidence
            
            self.graph_data.occurs_in.append((canonical, chunk.id, chunk.pedagogy_role))
            
            if concept.is_main_concept and chunk.pedagogy_role in ("definition", "explanation"):
                self.graph_data.teaches.append((chunk.id, canonical))
        
        for i, c1 in enumerate(chunk_concept_names):
            for c2 in chunk_concept_names[i + 1:]:
                key = (min(c1, c2), max(c1, c2))
                self._cooccurrence[key] += 1
        
        for rel in chunk.local_relations:
            rel_data = RelationshipData(
                source=rel.source_concept,
                target=rel.target_concept,
                relation_type=rel.relation_type,
                confidence=rel.confidence,
                evidence=[chunk.id],
                source_book=self.book_id,
            )
            
            if rel.relation_type == RelationType.PREREQUISITE_OF:
                self.graph_data.prerequisites.append(rel_data)
            elif rel.relation_type == RelationType.IS_A:
                self.graph_data.is_a.append(rel_data)
            elif rel.relation_type == RelationType.PART_OF:
                self.graph_data.part_of.append(rel_data)
    
    def finalize(self) -> BookGraphData:
        """Finalize graph building and compute derived data."""
        for (c1, c2), count in self._cooccurrence.items():
            if count >= self.min_cooccurrence:
                weight = min(1.0, count / 10.0)
                self.graph_data.related_to.append((c1, c2, weight))
        
        for node in self.graph_data.concepts.values():
            if node.frequency <= self.noise_frequency_threshold and node.confidence < 0.6:
                node.is_noisy = True
        
        return self.graph_data
    
    def build_prerequisites_llm(self, use_lite_model: bool = True) -> List[RelationshipData]:
        """Use LLM to infer prerequisite relationships."""
        concepts = self.graph_data.concepts
        if len(concepts) < 2:
            return []
        
        concept_list = []
        for canonical, node in concepts.items():
            if not node.is_noisy:
                concept_list.append(
                    f"- {node.display_name} (type: {node.concept_type.value}, "
                    f"difficulty: {node.dominant_difficulty}, first_page: {node.first_page})"
                )
        
        if len(concept_list) < 2:
            return []
        
        try:
            import yaml
            prompts_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "prompts", "ingest_v2.yaml"
            )
            with open(prompts_path) as f:
                prompts = yaml.safe_load(f)
            template = prompts.get("prerequisites", {}).get("build_from_concepts", "")
        except Exception:
            template = ""
        
        if not template:
            logger.warning("Prerequisites prompt not found")
            return []
        
        domain = self._infer_domain()
        prompt = template.replace("{{domain}}", domain)
        prompt = prompt.replace("{{concepts_list}}", "\n".join(concept_list[:100]))
        
        try:
            from llm.common import call_json_chat
            model = "google/gemini-2.0-flash-lite-001" if use_lite_model else None
            result = call_json_chat(prompt, default={"prerequisites": []}, model_hint=model)
        except Exception as e:
            logger.warning(f"LLM prerequisite inference failed: {e}")
            return []
        
        new_prereqs = []
        for item in result.get("prerequisites", []):
            from_c = item.get("from", "").lower().strip()
            to_c = item.get("to", "").lower().strip()
            conf = item.get("confidence", 0.5)
            
            if conf < 0.6 or from_c not in concepts or to_c not in concepts:
                continue
            
            new_prereqs.append(RelationshipData(
                source=from_c, target=to_c,
                relation_type=RelationType.PREREQUISITE_OF,
                confidence=conf, evidence=[item.get("reason", "")],
                source_book=self.book_id,
            ))
        
        self.graph_data.prerequisites.extend(new_prereqs)
        logger.info(f"LLM inferred {len(new_prereqs)} prerequisites")
        return new_prereqs
    
    def _infer_domain(self) -> str:
        concept_text = " ".join(n.display_name.lower() for n in self.graph_data.concepts.values())
        domain_keywords = {
            "Physics": ["heat", "thermal", "energy", "force", "motion", "wave", "electric"],
            "Mathematics": ["equation", "derivative", "integral", "function", "matrix", "theorem"],
            "Chemistry": ["reaction", "bond", "molecule", "element", "compound"],
            "Biology": ["cell", "organism", "gene", "protein", "enzyme"],
        }
        for domain, keywords in domain_keywords.items():
            if any(k in concept_text for k in keywords):
                return domain
        return "STEM"


# =============================================================================
# Neo4j Storage
# =============================================================================

class Neo4jGraphWriter:
    """Writes graph data to Neo4j."""
    
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
    
    def write_book_graph(self, graph_data: BookGraphData):
        """Write a book's graph data to Neo4j."""
        driver = self._get_driver()
        
        with driver.session() as session:
            self._create_concepts(session, graph_data)
            self._create_chunks(session, graph_data)
            self._create_relationships(session, graph_data)
        
        logger.info(f"Wrote graph: {len(graph_data.concepts)} concepts, {len(graph_data.chunks)} chunks")
    
    def _create_concepts(self, session, graph_data: BookGraphData):
        concepts = [
            node.to_dict() | {"book_id": graph_data.book_id}
            for node in graph_data.concepts.values() if not node.is_noisy
        ]
        if not concepts:
            return
        
        session.execute_write(lambda tx: tx.run("""
            UNWIND $concepts AS c
            MERGE (concept:Concept {canonical_name: c.canonical_name})
            ON CREATE SET concept.created_at = datetime()
            SET concept.display_name = c.display_name,
                concept.name_lower = c.canonical_name,
                concept.concept_type = c.concept_type,
                concept.difficulty = c.difficulty,
                concept.frequency = c.frequency,
                concept.first_page = c.first_page,
                concept.primary_role = c.primary_role,
                concept.definition = c.definition,
                concept.confidence = c.confidence,
                concept.book_id = c.book_id
        """, concepts=concepts))
    
    def _create_chunks(self, session, graph_data: BookGraphData):
        chunks = [{
            "id": c.id, "resource_id": graph_data.resource_id,
            "book_id": graph_data.book_id, "page_number": c.page_number,
            "pedagogy_role": c.pedagogy_role, "difficulty": c.difficulty,
            "snippet": c.text_snippet[:200],
        } for c in graph_data.chunks]
        
        if not chunks:
            return
        
        session.execute_write(lambda tx: tx.run("""
            UNWIND $chunks AS c
            MERGE (chunk:Chunk {id: c.id})
            ON CREATE SET chunk.created_at = datetime()
            SET chunk.resource_id = c.resource_id, chunk.book_id = c.book_id,
                chunk.page_number = c.page_number, chunk.pedagogy_role = c.pedagogy_role,
                chunk.difficulty = c.difficulty, chunk.snippet = c.snippet
        """, chunks=chunks))
    
    def _create_relationships(self, session, graph_data: BookGraphData):
        # OCCURS_IN
        if graph_data.occurs_in:
            rels = [{"concept": r[0], "chunk_id": r[1], "role": r[2]} for r in graph_data.occurs_in]
            session.execute_write(lambda tx: tx.run("""
                UNWIND $rels AS r
                MATCH (concept:Concept {canonical_name: r.concept})
                MATCH (chunk:Chunk {id: r.chunk_id})
                MERGE (concept)-[rel:OCCURS_IN]->(chunk)
                SET rel.pedagogy_role = r.role
            """, rels=rels))
        
        # TEACHES
        if graph_data.teaches:
            rels = [{"chunk_id": r[0], "concept": r[1]} for r in graph_data.teaches]
            session.execute_write(lambda tx: tx.run("""
                UNWIND $rels AS r
                MATCH (chunk:Chunk {id: r.chunk_id})
                MATCH (concept:Concept {canonical_name: r.concept})
                MERGE (chunk)-[:TEACHES]->(concept)
            """, rels=rels))
        
        # PREREQUISITE_OF
        if graph_data.prerequisites:
            rels = [r.to_dict() for r in graph_data.prerequisites]
            session.execute_write(lambda tx: tx.run("""
                UNWIND $rels AS r
                MATCH (prereq:Concept {canonical_name: r.source})
                MATCH (target:Concept {canonical_name: r.target})
                MERGE (prereq)-[rel:PREREQUISITE_OF]->(target)
                SET rel.confidence = r.confidence, rel.source_book = r.source_book
            """, rels=rels))
        
        # RELATED_TO
        if graph_data.related_to:
            rels = [{"c1": r[0], "c2": r[1], "weight": r[2]} for r in graph_data.related_to]
            session.execute_write(lambda tx: tx.run("""
                UNWIND $rels AS r
                MATCH (a:Concept {canonical_name: r.c1})
                MATCH (b:Concept {canonical_name: r.c2})
                MERGE (a)-[rel:RELATED_TO]-(b)
                SET rel.weight = r.weight
            """, rels=rels))
    
    def delete_noisy_concepts(self, concept_names: List[str]):
        """Delete noisy concepts from the graph."""
        driver = self._get_driver()
        with driver.session() as session:
            session.execute_write(lambda tx: tx.run("""
                UNWIND $names AS name
                MATCH (c:Concept {canonical_name: name})
                DETACH DELETE c
            """, names=concept_names))
        logger.info(f"Deleted {len(concept_names)} noisy concepts")
