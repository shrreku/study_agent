"""
Data models for the v2 ingestion pipeline.

Defines structured representations for:
- Document layout (pages, blocks)
- Enriched chunks with ontology-aware metadata
- Concept extraction results
- Domain ontology entities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib


# =============================================================================
# Block & Page Models (Document Understanding Layer)
# =============================================================================

class BlockType(str, Enum):
    """Types of content blocks detected in a page."""
    HEADING_1 = "heading_1"
    HEADING_2 = "heading_2"
    HEADING_3 = "heading_3"
    HEADING_4 = "heading_4"
    PARAGRAPH = "paragraph"
    EQUATION = "equation"
    EQUATION_INLINE = "equation_inline"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    LIST_ITEM = "list_item"
    CODE = "code"
    FOOTNOTE = "footnote"
    HEADER = "header"  # Page header
    FOOTER = "footer"  # Page footer
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box for a block on a page."""
    x0: float  # Left
    y0: float  # Top
    x1: float  # Right
    y1: float  # Bottom
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_dict(self) -> Dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class PageBlock:
    """A single content block within a page."""
    block_type: BlockType
    text: str
    bbox: Optional[BoundingBox] = None
    
    # For equations
    latex: Optional[str] = None
    
    # For tables
    html: Optional[str] = None
    markdown: Optional[str] = None
    
    # For figures
    image_path: Optional[str] = None
    alt_text: Optional[str] = None
    
    # Style metadata
    font_size: Optional[float] = None
    is_bold: bool = False
    is_italic: bool = False
    
    # Reading order index within the page
    order_index: int = 0
    
    # Confidence score from OCR (0-1)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_type": self.block_type.value,
            "text": self.text,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "latex": self.latex,
            "html": self.html,
            "markdown": self.markdown,
            "order_index": self.order_index,
            "confidence": self.confidence,
        }


@dataclass
class StructuredPage:
    """A page with structured content blocks."""
    page_number: int
    blocks: List[PageBlock] = field(default_factory=list)
    
    # Page dimensions
    width: float = 0.0
    height: float = 0.0
    
    # Raw text fallback
    raw_text: str = ""
    
    # OCR backend used
    ocr_backend: str = "unknown"
    
    # Processing metadata
    is_scanned: bool = False
    language: str = "en"
    
    @property
    def headings(self) -> List[PageBlock]:
        """Get all heading blocks."""
        return [b for b in self.blocks if b.block_type.value.startswith("heading")]
    
    @property
    def equations(self) -> List[PageBlock]:
        """Get all equation blocks."""
        return [b for b in self.blocks if b.block_type in (BlockType.EQUATION, BlockType.EQUATION_INLINE)]
    
    @property
    def tables(self) -> List[PageBlock]:
        """Get all table blocks."""
        return [b for b in self.blocks if b.block_type == BlockType.TABLE]
    
    @property
    def figures(self) -> List[PageBlock]:
        """Get all figure blocks."""
        return [b for b in self.blocks if b.block_type == BlockType.FIGURE]
    
    def get_text(self) -> str:
        """Get concatenated text from all blocks in reading order."""
        sorted_blocks = sorted(self.blocks, key=lambda b: b.order_index)
        return "\n".join(b.text for b in sorted_blocks if b.text)
    
    @property
    def full_text(self) -> str:
        """Alias for get_text() for consistency with EnrichedChunk."""
        return self.get_text()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "blocks": [b.to_dict() for b in self.blocks],
            "width": self.width,
            "height": self.height,
            "ocr_backend": self.ocr_backend,
            "is_scanned": self.is_scanned,
        }


# =============================================================================
# Concept & Relation Models (Extraction Layer)
# =============================================================================

class ConceptType(str, Enum):
    """Ontology-aware concept types for STEM domains."""
    # Core scientific entities
    CORE_CONCEPT = "core_concept"           # Fundamental ideas (e.g., "energy", "force")
    PHENOMENON = "phenomenon"                # Observable events (e.g., "convection", "diffraction")
    LAW = "law"                              # Scientific laws (e.g., "Newton's Second Law")
    PRINCIPLE = "principle"                  # Guiding principles (e.g., "conservation of energy")
    THEOREM = "theorem"                      # Mathematical theorems
    
    # Quantitative entities
    PHYSICAL_QUANTITY = "physical_quantity"  # Measurable quantities (e.g., "temperature", "pressure")
    EQUATION = "equation"                    # Named equations (e.g., "Fourier's equation")
    CONSTANT = "constant"                    # Physical/math constants (e.g., "Planck's constant")
    UNIT = "unit"                            # Units of measurement
    
    # Methods & techniques
    METHOD = "method"                        # Analysis/solution methods
    TECHNIQUE = "technique"                  # Experimental/practical techniques
    ALGORITHM = "algorithm"                  # Computational algorithms
    MODEL = "model"                          # Mathematical/physical models
    
    # Structural entities
    SYSTEM = "system"                        # Physical/abstract systems
    COMPONENT = "component"                  # Parts of systems
    MATERIAL = "material"                    # Substances, materials
    DEVICE = "device"                        # Instruments, devices
    
    # Learning aids
    EXAMPLE = "example"                      # Illustrative examples
    APPLICATION = "application"              # Real-world applications
    DEFINITION = "definition"                # Formal definitions
    
    # Catch-all
    UNKNOWN = "unknown"


class RelationType(str, Enum):
    """Types of relationships between concepts."""
    PREREQUISITE_OF = "PREREQUISITE_OF"      # A is needed to understand B
    IS_A = "IS_A"                             # A is a type/subclass of B
    PART_OF = "PART_OF"                       # A is a component of B
    CAUSES = "CAUSES"                         # A causes/leads to B
    DERIVES_FROM = "DERIVES_FROM"             # A is derived from B (mathematical)
    EQUIVALENT_TO = "EQUIVALENT_TO"           # A and B are equivalent/synonymous
    APPLIES_TO = "APPLIES_TO"                 # A applies to / governs B
    MEASURES = "MEASURES"                     # A measures/quantifies B
    USES = "USES"                             # A uses/employs B
    CONTRASTS_WITH = "CONTRASTS_WITH"         # A contrasts with B


@dataclass
class ExtractedConcept:
    """A concept extracted from text with ontology metadata."""
    name: str
    canonical_name: str = ""  # Lowercased, normalized
    concept_type: ConceptType = ConceptType.UNKNOWN
    
    # Extraction metadata
    is_main_concept: bool = False  # One of the primary concepts in this chunk
    confidence: float = 0.5
    
    # Evidence
    source_text: str = ""  # Text snippet where it was found
    definition_text: str = ""  # If a definition was found
    
    # Disambiguation
    aliases: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.canonical_name:
            self.canonical_name = self._canonicalize(self.name)
    
    @staticmethod
    def _canonicalize(name: str) -> str:
        """Create canonical form of concept name."""
        return name.lower().strip().replace("  ", " ")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "concept_type": self.concept_type.value,
            "is_main_concept": self.is_main_concept,
            "confidence": self.confidence,
            "aliases": self.aliases,
        }


@dataclass
class LocalRelation:
    """A relationship between concepts found within a chunk or nearby context."""
    source_concept: str  # Canonical name
    target_concept: str  # Canonical name
    relation_type: RelationType
    confidence: float = 0.5
    evidence: str = ""  # Text snippet supporting this relation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_concept,
            "target": self.target_concept,
            "type": self.relation_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence[:200] if self.evidence else "",
        }


@dataclass
class FormulaEntity:
    """A formula/equation extracted from text."""
    latex: str
    formula_type: str = "equation"  # equation, definition, derivation
    
    # Links to concepts
    defines_concept: str = ""  # The concept this formula defines
    uses_concepts: List[str] = field(default_factory=list)  # Concepts used in formula
    
    # Variables
    variables: List[Dict[str, str]] = field(default_factory=list)  # [{symbol, meaning, units}]
    
    # Source
    label: str = ""  # e.g., "Eq. 3.1"
    source_page: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "latex": self.latex,
            "formula_type": self.formula_type,
            "defines_concept": self.defines_concept,
            "uses_concepts": self.uses_concepts,
            "variables": self.variables,
            "label": self.label,
        }


# =============================================================================
# Enriched Chunk Model (combines ChunkData + extraction results)
# =============================================================================

@dataclass
class EnrichedChunk:
    """A text chunk with rich extraction metadata."""
    # Identity
    id: str = ""
    resource_id: str = ""
    book_id: str = ""  # For multi-book tracking
    
    # Position
    page_number: int = 0
    page_start: int = 0
    page_end: int = 0
    source_offset: int = 0
    
    # Content
    full_text: str = ""
    text_snippet: str = ""
    
    # Structure (from OCR)
    chapter_number: str = ""
    chapter_title: str = ""
    section_number: str = ""
    section_title: str = ""
    section_level: int = 0
    section_path: List[str] = field(default_factory=list)
    
    # Block composition
    block_types: List[str] = field(default_factory=list)  # Summary of block types in chunk
    has_equation: bool = False
    has_table: bool = False
    has_figure: bool = False
    
    # Extracted concepts (ontology-aware)
    main_concepts: List[ExtractedConcept] = field(default_factory=list)  # Max 3
    mentioned_concepts: List[ExtractedConcept] = field(default_factory=list)
    
    # Local relations
    local_relations: List[LocalRelation] = field(default_factory=list)
    
    # Formulas
    formulas: List[FormulaEntity] = field(default_factory=list)
    
    # Classification
    pedagogy_role: str = "explanation"  # definition, example, derivation, proof, problem, summary
    difficulty: str = "intermediate"
    cognitive_level: str = "understand"
    
    # Hierarchical taxonomy
    domain: str = ""
    topic: str = ""
    subtopic: str = ""
    
    # Computed
    token_count: int = 0
    embedding: Optional[List[float]] = None
    
    # Quality flags
    is_noisy: bool = False
    noise_reason: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
        if not self.text_snippet:
            self.text_snippet = self.full_text[:300]
    
    def _generate_id(self) -> str:
        """Generate deterministic ID from content."""
        content = f"{self.resource_id}:{self.page_number}:{self.source_offset}:{self.full_text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    @property
    def all_concepts(self) -> List[ExtractedConcept]:
        """All concepts (main + mentioned)."""
        return self.main_concepts + self.mentioned_concepts
    
    @property
    def concept_names(self) -> List[str]:
        """Get all concept canonical names."""
        return [c.canonical_name for c in self.all_concepts]
    
    def to_tags_json(self) -> Dict[str, Any]:
        """Convert to JSONB tags format for Postgres."""
        return {
            "pedagogy_role": self.pedagogy_role,
            "difficulty": self.difficulty,
            "cognitive_level": self.cognitive_level,
            "domain": self.domain,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "main_concepts": [c.to_dict() for c in self.main_concepts[:3]],
            "mentioned_concepts": [c.canonical_name for c in self.mentioned_concepts[:10]],
            "local_relations": [r.to_dict() for r in self.local_relations[:5]],
            "formulas": [f.to_dict() for f in self.formulas[:5]],
            "chapter_title": self.chapter_title,
            "section_path": self.section_path,
            "block_types": self.block_types,
            "is_noisy": self.is_noisy,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Full serialization."""
        return {
            "id": self.id,
            "resource_id": self.resource_id,
            "book_id": self.book_id,
            "page_number": self.page_number,
            "full_text": self.full_text,
            "text_snippet": self.text_snippet,
            "chapter_title": self.chapter_title,
            "section_title": self.section_title,
            "section_path": self.section_path,
            "main_concepts": [c.to_dict() for c in self.main_concepts],
            "mentioned_concepts": [c.to_dict() for c in self.mentioned_concepts],
            "local_relations": [r.to_dict() for r in self.local_relations],
            "formulas": [f.to_dict() for f in self.formulas],
            "pedagogy_role": self.pedagogy_role,
            "difficulty": self.difficulty,
            "domain": self.domain,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "is_noisy": self.is_noisy,
        }


# =============================================================================
# Domain Ontology Models (Cross-book Layer)
# =============================================================================

@dataclass
class DomainConcept:
    """A canonical concept in the domain ontology (aggregated across books)."""
    id: str  # Slug like "heat_transfer.convection"
    display_name: str
    canonical_name: str = ""
    
    # Classification
    domain: str = ""  # e.g., "thermodynamics", "mechanics"
    concept_type: ConceptType = ConceptType.CORE_CONCEPT
    
    # Canonical definition (LLM-generated, human-reviewed)
    definition: str = ""
    
    # Aliases from different books
    aliases: List[str] = field(default_factory=list)
    
    # Source tracking
    source_books: List[str] = field(default_factory=list)  # resource_ids
    source_count: int = 0  # How many books mention this
    total_frequency: int = 0  # Total occurrences across all books
    
    # Quality
    is_curated: bool = False  # Human-reviewed
    confidence: float = 0.5
    
    # External links (for future)
    external_ids: Dict[str, str] = field(default_factory=dict)  # e.g., {"wikidata": "Q..."}
    
    def __post_init__(self):
        if not self.canonical_name:
            self.canonical_name = self.display_name.lower().strip()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "canonical_name": self.canonical_name,
            "domain": self.domain,
            "concept_type": self.concept_type.value,
            "definition": self.definition,
            "aliases": self.aliases,
            "source_books": self.source_books,
            "source_count": self.source_count,
            "is_curated": self.is_curated,
            "confidence": self.confidence,
        }


@dataclass
class OntologyEdge:
    """An edge in the domain ontology graph."""
    source: str  # DomainConcept canonical name
    target: str  # DomainConcept canonical name
    relation_type: str  # RelationType value string
    
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)  # Supporting chunk IDs or text
    
    # Source tracking
    source_books: List[str] = field(default_factory=list)
    
    # Quality
    is_curated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type if isinstance(self.relation_type, str) else self.relation_type.value,
            "confidence": self.confidence,
            "source_books": self.source_books,
            "is_curated": self.is_curated,
        }
