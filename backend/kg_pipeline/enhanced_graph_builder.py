"""Enhanced Educational Knowledge Graph construction with better extraction.

Improvements over basic LangChain GraphTransformer:
- Educational-specific prompts
- Noise filtering (page numbers, temperatures, etc.)
- Section-level aggregation for context
- Multi-stage extraction (concepts -> relationships)
- Formula extraction with LaTeX support
- Confidence scoring
"""
from __future__ import annotations

import logging
import re
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .base import canonicalize_concept, managed_driver
from .canonicalization import ConceptCanonicalizer
from .validation import RelationshipValidator

logger = logging.getLogger(__name__)


# Noise patterns to filter out
NOISE_PATTERNS = [
    r'^[\d\.\s]+$',  # Pure numbers/decimals
    r'^page\s+\d+',  # Page references
    r'^chapter\s+\d+',  # Chapter references
    r'^\d+°[CF]$',  # Temperatures
    r'^\(cid:\d+\)',  # PDF artifacts
    r'^figure\s+\d+',  # Figure references
    r'^table\s+\d+',  # Table references
    r'^eq\.\s*\d+',  # Equation numbers
    r'^section\s+\d+',  # Section references
    r'^example\s+\d+',  # Example headings
    r'^\d{4}$',  # Years
    r'^[a-z]$',  # Single letters
    r'^(i{1,3}|iv|v|vi{1,3}|ix|x)$',  # Roman numerals
]

# Generic terms that are too broad to be useful as concepts
GENERIC_TERMS = {
    "mathematics", "physics", "chemistry", "biology",
    "science", "engineering", "theory", "concept",
    "method", "approach", "technique", "process",
    "system", "model", "analysis", "study",
}


def _is_noise_concept(text: str) -> bool:
    """Check if a concept is noise and should be filtered."""
    if not text or len(text.strip()) < 3:
        return True

    text_lower = text.lower().strip()

    # Pattern-based noise
    for pattern in NOISE_PATTERNS:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return True

    # Generic terms
    if text_lower in GENERIC_TERMS:
        return True

    # Too long to be a concept label
    if len(text) > 100:
        return True

    # Excess special characters
    special_chars = sum(1 for c in text if not c.isalnum() and c not in " -,")
    if special_chars > len(text) // 3:
        return True

    return False


# Adaptive confidence thresholds per relationship type (env-configurable)
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


RELATIONSHIP_CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    "DEFINES": _env_float("KG_MIN_CONFIDENCE_DEFINES", 0.80),
    "PREREQUISITE_OF": _env_float("KG_MIN_CONFIDENCE_PREREQUISITE", 0.75),
    "DERIVES": _env_float("KG_MIN_CONFIDENCE_DERIVES", 0.75),
    "EXPLAINS": _env_float("KG_MIN_CONFIDENCE_EXPLAINS", 0.65),
    "APPLIES_TO": _env_float("KG_MIN_CONFIDENCE_APPLIES_TO", 0.65),
    "EXEMPLIFIES": _env_float("KG_MIN_CONFIDENCE_EXEMPLIFIES", 0.60),
    "RELATED_TO": _env_float("KG_MIN_CONFIDENCE_DEFAULT", 0.65),
}


def _get_enhanced_llm():
    """Create LLM with enhanced settings for educational extraction."""
    import os
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AIMLAPI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("AIMLAPI_BASE_URL")
    model = os.getenv("LLM_MODEL_MINI", "gpt-4o-mini")
    
    if not api_key or not base_url:
        raise ValueError("OPENAI_API_KEY and OPENAI_API_BASE must be set")
    
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.0,
        max_tokens=3000,
    )


EDUCATIONAL_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting educational knowledge graphs from textbook content.

Extract the following from the text:

**Concepts**: Core ideas, principles, phenomena, methods, or terms that are defined or explained.
- Focus on technical terms and domain concepts
- Exclude: page numbers, figure/table references, chapter numbers, pure measurements

**Relationships**:
- DEFINES: A defines B (explicit definitions)
- EXPLAINS: A explains/elaborates on B
- PREREQUISITE_OF: A is required to understand B
- DERIVES: A mathematically derives B
- APPLIES_TO: A is applied in context B
- EXEMPLIFIES: A is an example of B

**Important**: 
- Only extract meaningful educational concepts
- Avoid noise like "page 5", "200°C", "figure 1"
- Focus on conceptual relationships, not spatial/layout relationships
- Be specific and precise

Return a JSON object with:
{{
  "concepts": [
    {{"name": "concept name", "type": "concept|theorem|principle|method", "importance": "high|medium|low"}}
  ],
  "relationships": [
    {{"source": "concept A", "target": "concept B", "type": "DEFINES|EXPLAINS|PREREQUISITE_OF|DERIVES|APPLIES_TO|EXEMPLIFIES", "confidence": 0.0-1.0}}
  ],
  "formulas": [
    {{"equation": "LaTeX or text", "description": "what it represents", "variables": ["var1", "var2"]}}
  ]
}}

Only include high-confidence extractions. Quality over quantity."""),
    ("user", """Title: {title}
Section: {section}

Text:
{text}

Extract educational knowledge graph:""")
])


def extract_enhanced_educational_graph(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Node], List[Relationship]]:
    """Extract educational knowledge graph using enhanced prompts and filtering.
    
    Args:
        text: Educational text content
        metadata: Optional metadata (title, section, etc.)
    
    Returns:
        Tuple of (nodes, relationships) with noise filtered
    """
    if not text or not text.strip() or len(text.strip()) < 50:
        return [], []
    
    metadata = metadata or {}
    
    try:
        llm = _get_enhanced_llm()
        
        # Build prompt
        prompt = EDUCATIONAL_EXTRACTION_PROMPT.format_messages(
            title=metadata.get("title", ""),
            section=metadata.get("section_title", ""),
            text=text[:4000]  # Limit text length
        )
        
        # Get LLM response
        response = llm.invoke(prompt)
        content = response.content
        
        # Parse JSON
        import json
        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        data = json.loads(content.strip())
        
        # Extract nodes
        nodes = []
        concepts_data = data.get("concepts", [])
        
        for concept_data in concepts_data:
            if isinstance(concept_data, str):
                name = concept_data
                node_type = "Concept"
                importance = "medium"
            else:
                name = concept_data.get("name", "")
                node_type = concept_data.get("type", "concept").title()
                importance = concept_data.get("importance", "medium")
            
            # Filter noise
            if _is_noise_concept(name):
                continue
            
            # Map types
            type_mapping = {
                "concept": "Concept",
                "theorem": "Theorem",
                "principle": "Principle",
                "method": "Method",
            }
            node_type = type_mapping.get(node_type.lower(), "Concept")
            
            node = Node(
                id=name,
                type=node_type,
                properties={"importance": importance}
            )
            nodes.append(node)
        
        # Extract formulas as nodes
        formulas_data = data.get("formulas", [])
        for formula_data in formulas_data:
            if isinstance(formula_data, dict):
                equation = formula_data.get("equation", "")
                desc = formula_data.get("description", "")
                if equation:
                    node = Node(
                        id=desc if desc else equation[:50],
                        type="Formula",
                        properties={
                            "equation": equation,
                            "description": desc,
                            "variables": formula_data.get("variables", [])
                        }
                    )
                    nodes.append(node)
        
        # Extract relationships
        relationships: List[Relationship] = []
        rels_data = data.get("relationships", [])

        for rel_data in rels_data:
            if not isinstance(rel_data, dict):
                continue

            source = rel_data.get("source", "")
            target = rel_data.get("target", "")
            rel_type = rel_data.get("type", "RELATED_TO")
            confidence = float(rel_data.get("confidence", 0.8))

            # Filter noise concepts
            if _is_noise_concept(source) or _is_noise_concept(target):
                continue

            # Apply adaptive confidence threshold
            min_conf = RELATIONSHIP_CONFIDENCE_THRESHOLDS.get(rel_type, RELATIONSHIP_CONFIDENCE_THRESHOLDS.get("RELATED_TO", 0.65))
            if confidence < min_conf:
                continue

            # Find corresponding nodes
            source_node = None
            target_node = None

            for node in nodes:
                nid = (node.id or "")
                if nid.lower() == (source or "").lower():
                    source_node = node
                if nid.lower() == (target or "").lower():
                    target_node = node

            # Create nodes if they don't exist
            if not source_node:
                source_node = Node(id=source, type="Concept")
                nodes.append(source_node)

            if not target_node:
                target_node = Node(id=target, type="Concept")
                nodes.append(target_node)

            rel = Relationship(
                source=source_node,
                target=target_node,
                type=rel_type,
                properties={"confidence": confidence},
            )
            relationships.append(rel)

        # Canonicalize concept names and deduplicate nodes
        canonicalizer = ConceptCanonicalizer()
        id_map: Dict[str, Node] = {}
        for node in nodes:
            canon = canonicalizer.canonicalize(node.id)
            node.id = canon
            if canon not in id_map:
                id_map[canon] = node
            else:
                # Merge properties into existing
                try:
                    id_map[canon].properties.update(node.properties or {})
                except Exception:
                    pass
        nodes = list(id_map.values())

        # Re-point relationships to canonical nodes
        for rel in relationships:
            s_id = canonicalizer.canonicalize(rel.source.id)
            t_id = canonicalizer.canonicalize(rel.target.id)
            rel.source = id_map.get(s_id, rel.source)
            rel.target = id_map.get(t_id, rel.target)

        # Optional semantic validation of relationships
        validate_enabled = os.getenv("KG_SEMANTIC_VALIDATION_ENABLED", "true").lower() in ("1", "true", "yes")
        if validate_enabled:
            validator = RelationshipValidator()
            validated: List[Relationship] = []
            for rel in relationships:
                rtype = rel.type or "RELATED_TO"
                conf = float((rel.properties or {}).get("confidence", 0.8))
                if rtype == "PREREQUISITE_OF":
                    ok, adj, reason = validator.validate_prerequisite(rel.source.id, rel.target.id, conf)
                    if ok:
                        rel.properties = dict(rel.properties or {})
                        rel.properties["confidence"] = adj
                        rel.properties["validation"] = reason
                        validated.append(rel)
                elif rtype == "APPLIES_TO":
                    ok, adj, reason = validator.validate_applies_to(rel.source.id, rel.target.id, conf)
                    if ok:
                        rel.properties = dict(rel.properties or {})
                        rel.properties["confidence"] = adj
                        validated.append(rel)
                else:
                    validated.append(rel)
            relationships = validated

        logger.info(
            f"Enhanced extraction: {len(nodes)} nodes, {len(relationships)} relationships "
            f"from {len(concepts_data)} raw concepts, {len(rels_data)} raw relationships"
        )

        return nodes, relationships
        
    except Exception as e:
        logger.error(f"Enhanced extraction failed: {e}", exc_info=True)
        return [], []


def build_enhanced_educational_kg(
    text: str,
    chunk_id: str,
    resource_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Complete enhanced pipeline: extract graph and merge to Neo4j.
    
    Args:
        text: Educational text content
        chunk_id: Source chunk ID
        resource_id: Source resource ID
        metadata: Optional metadata
    
    Returns:
        Result dict with extraction stats
    """
    logger.info(f"Building enhanced educational KG for chunk {chunk_id}")
    
    # Extract graph with enhanced logic
    nodes, relationships = extract_enhanced_educational_graph(text, metadata)
    
    result = {
        "nodes_extracted": len(nodes),
        "relationships_extracted": len(relationships),
        "nodes_merged": 0,
        "relationships_merged": 0,
        "errors": 0,
    }
    
    if not nodes and not relationships:
        logger.warning(f"No graph elements extracted from chunk {chunk_id}")
        return result
    
    # Merge to Neo4j (reuse existing merge logic)
    from .graph_builder import merge_graph_to_neo4j
    
    stats = merge_graph_to_neo4j(
        nodes, relationships, chunk_id, resource_id, method="enhanced_educational"
    )
    
    result.update({
        "nodes_merged": stats["nodes"],
        "relationships_merged": stats["relationships"],
        "errors": stats["errors"],
    })
    
    logger.info(
        f"Enhanced KG built for chunk {chunk_id}: "
        f"{result['nodes_merged']} nodes, "
        f"{result['relationships_merged']} relationships"
    )
    
    return result
