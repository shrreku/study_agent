"""
Ontology-aware concept extraction for the v2 ingestion pipeline.

This module handles:
- Unified extraction from chunks (concepts, relations, formulas)
- Context-aware extraction using neighboring chunks
- Integration with known ontology concepts
- Noise detection and filtering
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import (
    ConceptType,
    EnrichedChunk,
    ExtractedConcept,
    FormulaEntity,
    LocalRelation,
    RelationType,
    StructuredPage,
    PageBlock,
    BlockType,
)

logger = logging.getLogger("backend.ingestion.v2.extraction")


# =============================================================================
# Extraction Configuration
# =============================================================================

@dataclass
class ExtractionConfig:
    """Configuration for concept extraction."""
    # Limits
    max_main_concepts: int = 3
    max_mentioned_concepts: int = 10
    max_local_relations: int = 5
    max_formulas: int = 5
    
    # Confidence thresholds
    min_concept_confidence: float = 0.5
    min_relation_confidence: float = 0.6
    
    # Context
    use_neighbor_context: bool = True
    neighbor_window: int = 1  # How many chunks before/after to include
    
    # LLM settings
    use_llm: bool = True
    llm_model: Optional[str] = None  # None = use default
    
    # Known concepts for mapping
    known_concepts: List[str] = None
    
    # Domain hint
    domain: str = "STEM"


# =============================================================================
# Heuristic Extraction (Fast, No LLM)
# =============================================================================

class HeuristicExtractor:
    """Fast, rule-based concept extraction without LLM calls."""
    
    # Stop words that are never concepts
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "this", "that", "these",
        "those", "it", "its", "we", "our", "you", "your", "they", "their",
        "chapter", "section", "figure", "table", "equation", "example", "problem",
        "solution", "note", "see", "given", "find", "determine", "calculate",
        "using", "applying", "substituting", "therefore", "hence", "thus",
        "introduction", "summary", "conclusion", "discussion", "results",
    }
    
    # Patterns for different concept types
    CONCEPT_PATTERNS = {
        ConceptType.LAW: [
            r"([A-Z][a-z]+(?:'s)?)\s+(Law|Laws)",
            r"(Law|Laws)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ],
        ConceptType.PRINCIPLE: [
            r"([A-Z][a-z]+(?:'s)?)\s+Principle",
            r"(Principle)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ],
        ConceptType.THEOREM: [
            r"([A-Z][a-z]+(?:'s)?)\s+Theorem",
        ],
        ConceptType.EQUATION: [
            r"([A-Z][a-z]+(?:'s)?)\s+Equation",
        ],
        ConceptType.PHYSICAL_QUANTITY: [
            r"\b(temperature|pressure|velocity|acceleration|force|energy|"
            r"mass|volume|density|viscosity|conductivity|diffusivity|"
            r"heat\s+flux|heat\s+transfer|momentum|entropy|enthalpy)\b",
        ],
        ConceptType.METHOD: [
            r"([A-Z][a-z]+(?:'s)?)\s+Method",
            r"(Method)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ],
        ConceptType.CONSTANT: [
            r"([A-Z][a-z]+(?:'s)?)\s+(Constant|Number)",
            r"(Boltzmann|Planck|Avogadro|Stefan-Boltzmann)\s+(constant|number)",
        ],
    }
    
    # Technical term suffixes
    TECHNICAL_SUFFIXES = [
        "tion", "sion", "ance", "ence", "ity", "ment",
        "ivity", "ivity", "ysis", "osis",
    ]
    
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
    
    def extract(self, chunk: EnrichedChunk) -> EnrichedChunk:
        """Extract concepts using heuristics."""
        text = chunk.full_text
        text_lower = text.lower()
        
        # Extract concepts
        main_concepts = []
        mentioned_concepts = []
        
        # Pattern-based extraction
        for concept_type, patterns in self.CONCEPT_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    name = match.group(0).strip()
                    if self._is_valid_concept(name):
                        concept = ExtractedConcept(
                            name=name,
                            concept_type=concept_type,
                            confidence=0.8,
                            source_text=self._get_context(text, match.start(), match.end()),
                        )
                        if len(main_concepts) < self.config.max_main_concepts:
                            concept.is_main_concept = True
                            main_concepts.append(concept)
                        else:
                            mentioned_concepts.append(concept)
        
        # Multi-word capitalized phrases (e.g., "Heat Transfer")
        cap_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
        for match in cap_pattern.finditer(text):
            phrase = match.group(1)
            if self._is_valid_concept(phrase) and not self._already_extracted(phrase, main_concepts + mentioned_concepts):
                mentioned_concepts.append(ExtractedConcept(
                    name=phrase,
                    concept_type=ConceptType.CORE_CONCEPT,
                    confidence=0.6,
                ))
        
        # Technical terms by suffix
        for suffix in self.TECHNICAL_SUFFIXES:
            pattern = re.compile(rf'\b([a-z]+{suffix})\b', re.IGNORECASE)
            for match in pattern.finditer(text_lower):
                term = match.group(1)
                if len(term) > 6 and self._is_valid_concept(term):
                    if not self._already_extracted(term, main_concepts + mentioned_concepts):
                        mentioned_concepts.append(ExtractedConcept(
                            name=term.title(),
                            concept_type=ConceptType.PHENOMENON,
                            confidence=0.5,
                        ))
        
        # Detect pedagogy role
        chunk.pedagogy_role = self._detect_pedagogy_role(text_lower)
        
        # Detect difficulty
        chunk.difficulty = self._detect_difficulty(text_lower)
        
        # Check for equations
        chunk.has_equation = bool(re.search(r'[=∫∑∏]|\$[^$]+\$|\\begin\{equation\}', text))
        
        # Limit results
        chunk.main_concepts = main_concepts[:self.config.max_main_concepts]
        chunk.mentioned_concepts = mentioned_concepts[:self.config.max_mentioned_concepts]
        
        return chunk
    
    def _is_valid_concept(self, name: str) -> bool:
        """Check if a name is a valid concept."""
        name_lower = name.lower().strip()
        
        # Too short
        if len(name_lower) < 3:
            return False
        
        # All stop words
        words = name_lower.split()
        if all(w in self.STOP_WORDS for w in words):
            return False
        
        # Mostly numbers
        if sum(c.isdigit() for c in name) > len(name) * 0.5:
            return False
        
        return True
    
    def _already_extracted(
        self,
        name: str,
        existing: List[ExtractedConcept],
    ) -> bool:
        """Check if concept already extracted."""
        name_lower = name.lower()
        for c in existing:
            if c.canonical_name == name_lower:
                return True
        return False
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get surrounding context for a match."""
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        return text[ctx_start:ctx_end]
    
    def _detect_pedagogy_role(self, text_lower: str) -> str:
        """Detect pedagogy role from text patterns."""
        if any(p in text_lower for p in ["is defined as", "we define", "definition:"]):
            return "definition"
        elif any(p in text_lower for p in ["for example", "for instance", "consider the case"]):
            return "example"
        elif any(p in text_lower for p in ["proof:", "to prove", "q.e.d.", "we prove"]):
            return "proof"
        elif any(p in text_lower for p in ["deriving", "derivation", "we derive"]):
            return "derivation"
        elif any(p in text_lower for p in ["problem:", "exercise:", "find the", "calculate"]):
            return "problem"
        elif any(p in text_lower for p in ["in summary", "to summarize", "in conclusion"]):
            return "summary"
        elif any(p in text_lower for p in ["introduction", "this chapter", "we will study"]):
            return "introduction"
        else:
            return "explanation"
    
    def _detect_difficulty(self, text_lower: str) -> str:
        """Detect difficulty level from text patterns."""
        advanced_terms = [
            "differential equation", "partial derivative", "laplacian",
            "fourier transform", "bessel function", "eigenvalue",
            "perturbation", "asymptotic", "tensor", "manifold",
        ]
        intermediate_terms = [
            "integral", "derivative", "gradient", "divergence",
            "matrix", "vector", "coefficient", "boundary condition",
        ]
        
        if any(t in text_lower for t in advanced_terms):
            return "advanced"
        elif any(t in text_lower for t in intermediate_terms):
            return "intermediate"
        else:
            return "introductory"


# =============================================================================
# LLM-Based Extraction
# =============================================================================

class LLMExtractor:
    """LLM-based ontology-aware concept extraction."""
    
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        self._prompts = None
    
    def _load_prompts(self):
        """Load extraction prompts."""
        if self._prompts is not None:
            return
        
        try:
            from backend.prompts import get as prompt_get
            self._prompts = {
                "extract_unified": prompt_get("ingest_v2.concepts.extract_unified"),
                "extract_minimal": prompt_get("ingest_v2.concepts.extract_minimal"),
            }
        except Exception:
            # Fallback to loading directly
            import yaml
            prompts_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "prompts", "ingest_v2.yaml"
            )
            if os.path.exists(prompts_path):
                with open(prompts_path) as f:
                    data = yaml.safe_load(f)
                self._prompts = {
                    "extract_unified": data.get("concepts", {}).get("extract_unified", ""),
                    "extract_minimal": data.get("concepts", {}).get("extract_minimal", ""),
                }
            else:
                self._prompts = {}
    
    def extract(
        self,
        chunk: EnrichedChunk,
        prev_chunk: Optional[EnrichedChunk] = None,
        next_chunk: Optional[EnrichedChunk] = None,
    ) -> EnrichedChunk:
        """Extract concepts using LLM."""
        self._load_prompts()
        
        # Build context
        prev_summary = ""
        if prev_chunk and self.config.use_neighbor_context:
            prev_summary = f"Previous: {prev_chunk.text_snippet[:200]}"
        
        known_concepts_str = ""
        if self.config.known_concepts:
            known_concepts_str = ", ".join(self.config.known_concepts[:50])
        
        # Render prompt
        prompt_template = self._prompts.get("extract_unified", "")
        if not prompt_template:
            logger.warning("No extraction prompt found, falling back to heuristics")
            return HeuristicExtractor(self.config).extract(chunk)
        
        prompt = prompt_template.replace("{{text}}", chunk.full_text[:3000])
        prompt = prompt.replace("{{domain}}", self.config.domain)
        prompt = prompt.replace("{{chapter_title}}", chunk.chapter_title or "")
        prompt = prompt.replace("{{section_title}}", chunk.section_title or "")
        prompt = prompt.replace("{{section_path}}", " > ".join(chunk.section_path) if chunk.section_path else "")
        prompt = prompt.replace("{{prev_chunk_summary}}", prev_summary)
        prompt = prompt.replace("{{known_concepts}}", known_concepts_str)
        
        # Call LLM
        try:
            result = self._call_llm(prompt)
            self._apply_result(chunk, result)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, falling back to heuristics")
            return HeuristicExtractor(self.config).extract(chunk)
        
        return chunk
    
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM for extraction."""
        try:
            from backend.llm import call_llm_json
            return call_llm_json(prompt, default={})
        except ImportError:
            # Try alternative
            try:
                from llm.common import call_json_chat
                return call_json_chat(
                    prompt,
                    default={},
                    model_hint=self.config.llm_model,
                )
            except ImportError:
                raise ImportError("No LLM module available")
    
    def _apply_result(self, chunk: EnrichedChunk, result: Dict[str, Any]):
        """Apply LLM extraction result to chunk."""
        # Main concepts
        main_concepts = []
        for c in result.get("main_concepts", [])[:self.config.max_main_concepts]:
            try:
                concept_type = ConceptType(c.get("concept_type", "unknown"))
            except ValueError:
                concept_type = ConceptType.UNKNOWN
            
            main_concepts.append(ExtractedConcept(
                name=c.get("name", ""),
                concept_type=concept_type,
                is_main_concept=True,
                confidence=c.get("confidence", 0.5),
                definition_text=c.get("definition_text", ""),
            ))
        chunk.main_concepts = main_concepts
        
        # Mentioned concepts
        mentioned_concepts = []
        for c in result.get("mentioned_concepts", [])[:self.config.max_mentioned_concepts]:
            try:
                concept_type = ConceptType(c.get("concept_type", "unknown"))
            except ValueError:
                concept_type = ConceptType.UNKNOWN
            
            mentioned_concepts.append(ExtractedConcept(
                name=c.get("name", ""),
                concept_type=concept_type,
                is_main_concept=False,
                confidence=c.get("confidence", 0.5),
            ))
        chunk.mentioned_concepts = mentioned_concepts
        
        # Local relations
        local_relations = []
        for r in result.get("local_relations", [])[:self.config.max_local_relations]:
            try:
                rel_type = RelationType(r.get("type", "PREREQUISITE_OF"))
            except ValueError:
                continue
            
            confidence = r.get("confidence", 0.5)
            if confidence >= self.config.min_relation_confidence:
                local_relations.append(LocalRelation(
                    source_concept=r.get("source", "").lower(),
                    target_concept=r.get("target", "").lower(),
                    relation_type=rel_type,
                    confidence=confidence,
                    evidence=r.get("evidence", ""),
                ))
        chunk.local_relations = local_relations
        
        # Formulas
        formulas = []
        for f in result.get("formulas", [])[:self.config.max_formulas]:
            formulas.append(FormulaEntity(
                latex=f.get("latex", ""),
                formula_type=f.get("formula_type", "equation"),
                defines_concept=f.get("defines_concept", ""),
                uses_concepts=f.get("uses_concepts", []),
                variables=f.get("variables", []),
                label=f.get("label", ""),
                source_page=chunk.page_number,
            ))
        chunk.formulas = formulas
        
        # Classification
        chunk.pedagogy_role = result.get("pedagogy_role", "explanation")
        chunk.difficulty = result.get("difficulty", "intermediate")
        chunk.cognitive_level = result.get("cognitive_level", "understand")
        chunk.domain = result.get("domain", "")
        chunk.topic = result.get("topic", "")
        chunk.subtopic = result.get("subtopic", "")


# =============================================================================
# Unified Extractor
# =============================================================================

class ConceptExtractor:
    """
    Unified concept extractor that combines heuristics and LLM.
    
    Strategy:
    - Always run heuristics first (fast baseline)
    - For FULL mode, enhance with LLM
    - Apply noise filtering at the end
    """
    
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        self.heuristic = HeuristicExtractor(self.config)
        self.llm = LLMExtractor(self.config) if self.config.use_llm else None
    
    def extract(
        self,
        chunk: EnrichedChunk,
        prev_chunk: Optional[EnrichedChunk] = None,
        next_chunk: Optional[EnrichedChunk] = None,
    ) -> EnrichedChunk:
        """Extract concepts from chunk."""
        # Always start with heuristics
        chunk = self.heuristic.extract(chunk)
        
        # Enhance with LLM if enabled and chunk is substantial
        if self.llm and len(chunk.full_text) > 100:
            chunk = self.llm.extract(chunk, prev_chunk, next_chunk)
        
        # Apply noise filtering
        chunk = self._filter_noise(chunk)
        
        return chunk
    
    def _filter_noise(self, chunk: EnrichedChunk) -> EnrichedChunk:
        """Filter out noisy concepts."""
        # Filter by confidence
        chunk.main_concepts = [
            c for c in chunk.main_concepts
            if c.confidence >= self.config.min_concept_confidence
        ]
        chunk.mentioned_concepts = [
            c for c in chunk.mentioned_concepts
            if c.confidence >= self.config.min_concept_confidence
        ]
        
        # Remove duplicates (by canonical name)
        seen = set()
        unique_main = []
        for c in chunk.main_concepts:
            if c.canonical_name not in seen:
                seen.add(c.canonical_name)
                unique_main.append(c)
        chunk.main_concepts = unique_main
        
        unique_mentioned = []
        for c in chunk.mentioned_concepts:
            if c.canonical_name not in seen:
                seen.add(c.canonical_name)
                unique_mentioned.append(c)
        chunk.mentioned_concepts = unique_mentioned
        
        return chunk
    
    def extract_batch(
        self,
        chunks: List[EnrichedChunk],
    ) -> List[EnrichedChunk]:
        """Extract concepts from a batch of chunks with context."""
        results = []
        for i, chunk in enumerate(chunks):
            prev_chunk = chunks[i - 1] if i > 0 else None
            next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None
            results.append(self.extract(chunk, prev_chunk, next_chunk))
        return results


# =============================================================================
# Chunk Builder (from structured pages)
# =============================================================================

class ChunkBuilder:
    """Build EnrichedChunks from StructuredPages."""
    
    def __init__(
        self,
        resource_id: str,
        book_id: str = "",
        max_chunk_tokens: int = 300,
        min_chunk_tokens: int = 50,
    ):
        self.resource_id = resource_id
        self.book_id = book_id or resource_id
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        
        # Track section hierarchy
        self.current_chapter = ""
        self.current_chapter_num = ""
        self.current_section = ""
        self.current_section_num = ""
        self.section_path: List[str] = []
    
    def build_chunks(
        self,
        pages: List[StructuredPage],
    ) -> List[EnrichedChunk]:
        """Build chunks from structured pages."""
        chunks = []
        
        for page in pages:
            page_chunks = self._process_page(page)
            chunks.extend(page_chunks)
        
        return chunks
    
    def _process_page(self, page: StructuredPage) -> List[EnrichedChunk]:
        """Process a single page into chunks."""
        chunks = []
        
        # Update section tracking from headings
        for block in page.blocks:
            if block.block_type == BlockType.HEADING_1:
                self.current_chapter = block.text
                self.current_chapter_num = self._extract_number(block.text)
                self.section_path = [block.text]
            elif block.block_type == BlockType.HEADING_2:
                self.current_section = block.text
                self.current_section_num = self._extract_number(block.text)
                if len(self.section_path) > 0:
                    self.section_path = [self.section_path[0], block.text]
                else:
                    self.section_path = [block.text]
            elif block.block_type == BlockType.HEADING_3:
                if len(self.section_path) >= 2:
                    self.section_path = self.section_path[:2] + [block.text]
                else:
                    self.section_path.append(block.text)
        
        # Group content blocks into chunks
        current_blocks: List[PageBlock] = []
        current_tokens = 0
        source_offset = 0
        
        for block in page.blocks:
            # Skip headers/footers
            if block.block_type in (BlockType.HEADER, BlockType.FOOTER):
                continue
            
            block_tokens = len(block.text.split())
            
            # Start new chunk if needed
            if current_tokens + block_tokens > self.max_chunk_tokens and current_blocks:
                chunk = self._create_chunk(
                    page, current_blocks, source_offset,
                )
                if chunk:
                    chunks.append(chunk)
                source_offset += sum(len(b.text) for b in current_blocks)
                current_blocks = []
                current_tokens = 0
            
            current_blocks.append(block)
            current_tokens += block_tokens
        
        # Flush remaining
        if current_blocks:
            chunk = self._create_chunk(
                page, current_blocks, source_offset,
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self,
        page: StructuredPage,
        blocks: List[PageBlock],
        source_offset: int,
    ) -> Optional[EnrichedChunk]:
        """Create an EnrichedChunk from blocks."""
        if not blocks:
            return None
        
        # Combine text
        full_text = "\n".join(b.text for b in blocks if b.text)
        if len(full_text.split()) < self.min_chunk_tokens:
            return None
        
        # Collect block type summary
        block_types = list(set(b.block_type.value for b in blocks))
        
        # Check for special content
        has_equation = any(
            b.block_type in (BlockType.EQUATION, BlockType.EQUATION_INLINE)
            for b in blocks
        )
        has_table = any(b.block_type == BlockType.TABLE for b in blocks)
        has_figure = any(b.block_type == BlockType.FIGURE for b in blocks)
        
        return EnrichedChunk(
            resource_id=self.resource_id,
            book_id=self.book_id,
            page_number=page.page_number,
            page_start=page.page_number,
            page_end=page.page_number,
            source_offset=source_offset,
            full_text=full_text,
            text_snippet=full_text[:300],
            chapter_number=self.current_chapter_num,
            chapter_title=self.current_chapter,
            section_number=self.current_section_num,
            section_title=self.current_section,
            section_path=list(self.section_path),
            block_types=block_types,
            has_equation=has_equation,
            has_table=has_table,
            has_figure=has_figure,
            token_count=len(full_text.split()),
        )
    
    def _extract_number(self, text: str) -> str:
        """Extract section/chapter number from heading text."""
        match = re.match(r'^([\d.]+)\s', text)
        if match:
            return match.group(1)
        match = re.match(r'^(?:Chapter|Section)\s+(\d+)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""
