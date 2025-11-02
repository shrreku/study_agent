"""Educational Knowledge Graph construction using LangChain GraphTransformer.

This module provides a robust, LLM-powered approach to extracting entities
and relationships from educational content, specifically optimized for:
- Concepts and their definitions
- Mathematical formulas and derivations
- Prerequisite relationships
- Pedagogical relations (explains, exemplifies, etc.)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from .base import canonicalize_concept, managed_driver

logger = logging.getLogger(__name__)


def _get_llm_instance():
    """Create LLM instance compatible with your AIMLAPI setup."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AIMLAPI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("AIMLAPI_BASE_URL")
    if base_url:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
    model = os.getenv("LLM_MODEL_MINI") or os.getenv("LLM_MODEL_NANO") or "gpt-4o-mini"
    
    if not api_key or not base_url:
        raise ValueError("OPENAI_API_KEY and OPENAI_API_BASE must be set")
    
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.0,
        max_tokens=2000,
    )


def _create_educational_transformer() -> LLMGraphTransformer:
    """Create LLMGraphTransformer configured for educational content.
    
    Extracts:
    - Concept nodes
    - Relationship types: DEFINES, EXPLAINS, REQUIRES, DERIVES, EXEMPLIFIES
    """
    llm = _get_llm_instance()
    
    # Define allowed node and relationship types for educational content
    allowed_nodes = ["Concept", "Formula", "Example", "Theorem", "Definition"]
    allowed_relationships = [
        "DEFINES",
        "EXPLAINS", 
        "REQUIRES",  # Prerequisites
        "DERIVES",   # Mathematical derivations
        "EXEMPLIFIES",
        "RELATED_TO",
        "PROVES",
    ]
    
    return LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        node_properties=["description", "importance"],
        relationship_properties=["context", "confidence"],
        strict_mode=False,  # Allow flexible extraction
    )


def extract_educational_graph(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Node], List[Relationship]]:
    """Extract educational knowledge graph from text using LangChain.
    
    Args:
        text: Educational text content
        metadata: Optional metadata (title, section, chunk_type, etc.)
    
    Returns:
        Tuple of (nodes, relationships) extracted from text
    """
    if not text or not text.strip():
        return [], []
    
    metadata = metadata or {}
    
    # Create document with metadata
    doc = Document(
        page_content=text,
        metadata={
            "title": metadata.get("title", ""),
            "section": metadata.get("section_title", ""),
            "chunk_type": metadata.get("chunk_type", ""),
            "source": metadata.get("resource_id", ""),
        }
    )
    
    try:
        transformer = _create_educational_transformer()
        graph_documents: List[GraphDocument] = transformer.convert_to_graph_documents([doc])
        
        if not graph_documents:
            return [], []
        
        # Extract nodes and relationships from first graph document
        graph_doc = graph_documents[0]
        return graph_doc.nodes, graph_doc.relationships
        
    except Exception as e:
        logger.exception("Failed to extract graph from text", extra={"error": str(e)})
        return [], []


def merge_graph_to_neo4j(
    nodes: List[Node],
    relationships: List[Relationship],
    chunk_id: str,
    resource_id: str,
    method: str = "langchain_graph_transformer",
) -> Dict[str, int]:
    """Merge extracted graph into Neo4j.
    
    Args:
        nodes: Extracted nodes
        relationships: Extracted relationships
        chunk_id: Source chunk ID
        resource_id: Source resource ID
        method: Extraction method tag
    
    Returns:
        Stats dict with counts of nodes/relationships created
    """
    stats = {"nodes": 0, "relationships": 0, "errors": 0}
    
    if not nodes and not relationships:
        return stats
    
    with managed_driver() as driver:
        if driver is None:
            logger.error("Neo4j driver unavailable")
            return stats
        
        try:
            with driver.session() as session:
                # Merge nodes
                for node in nodes:
                    try:
                        node_type = node.type or "Concept"
                        node_id = node.id
                        
                        # Canonicalize if it's a Concept
                        if node_type == "Concept":
                            canonical, display = canonicalize_concept(node_id)
                            if not canonical:
                                continue
                        else:
                            canonical = node_id.lower().strip()
                            display = node_id
                        
                        properties = node.properties or {}
                        
                        session.execute_write(
                            _merge_node_tx,
                            node_type=node_type,
                            canonical=canonical,
                            display=display,
                            properties=properties,
                        )
                        stats["nodes"] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to merge node {node.id}: {e}")
                        stats["errors"] += 1
                
                # Merge relationships
                for rel in relationships:
                    try:
                        source_node = rel.source
                        target_node = rel.target
                        rel_type = rel.type or "RELATED_TO"
                        
                        # Canonicalize concept names
                        source_canonical, source_display = canonicalize_concept(source_node.id)
                        target_canonical, target_display = canonicalize_concept(target_node.id)
                        
                        if not source_canonical or not target_canonical:
                            continue
                        
                        properties = rel.properties or {}
                        properties["chunk_id"] = chunk_id
                        properties["resource_id"] = resource_id
                        properties["method"] = method
                        
                        session.execute_write(
                            _merge_relationship_tx,
                            source_type=source_node.type or "Concept",
                            source_canonical=source_canonical,
                            source_display=source_display,
                            target_type=target_node.type or "Concept",
                            target_canonical=target_canonical,
                            target_display=target_display,
                            rel_type=rel_type,
                            properties=properties,
                        )
                        stats["relationships"] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to merge relationship: {e}")
                        stats["errors"] += 1
                
        except Exception as e:
            logger.exception("Failed to merge graph to Neo4j", extra={"error": str(e)})
            stats["errors"] += 1
    
    return stats


def _merge_node_tx(tx, node_type: str, canonical: str, display: str, properties: Dict[str, Any]):
    """Transaction function to merge a node."""
    query = f"""
    MERGE (n:{node_type} {{canonical_name: $canonical}})
    ON CREATE SET 
        n.display_name = $display,
        n.name_lower = $canonical,
        n.created_at = datetime()
    SET 
        n.display_name = coalesce(n.display_name, $display),
        n.last_seen = datetime(),
        n.name_lower = $canonical
    SET n += $properties
    """
    tx.run(query, canonical=canonical, display=display, properties=properties)


def _merge_relationship_tx(
    tx,
    source_type: str,
    source_canonical: str,
    source_display: str,
    target_type: str,
    target_canonical: str,
    target_display: str,
    rel_type: str,
    properties: Dict[str, Any],
):
    """Transaction function to merge a relationship."""
    # First ensure both nodes exist
    tx.run(
        f"""
        MERGE (s:{source_type} {{canonical_name: $source_canonical}})
        ON CREATE SET s.display_name = $source_display, s.name_lower = $source_canonical, s.created_at = datetime()
        SET s.last_seen = datetime()
        """,
        source_canonical=source_canonical,
        source_display=source_display,
    )
    
    tx.run(
        f"""
        MERGE (t:{target_type} {{canonical_name: $target_canonical}})
        ON CREATE SET t.display_name = $target_display, t.name_lower = $target_canonical, t.created_at = datetime()
        SET t.last_seen = datetime()
        """,
        target_canonical=target_canonical,
        target_display=target_display,
    )
    
    # Create relationship
    query = f"""
    MATCH (s:{source_type} {{canonical_name: $source_canonical}})
    MATCH (t:{target_type} {{canonical_name: $target_canonical}})
    MERGE (s)-[r:{rel_type}]->(t)
    SET r.updated_at = datetime()
    SET r += $properties
    """
    tx.run(
        query,
        source_canonical=source_canonical,
        target_canonical=target_canonical,
        properties=properties,
    )


def build_educational_kg(
    text: str,
    chunk_id: str,
    resource_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Complete pipeline: extract graph and merge to Neo4j.
    
    Args:
        text: Educational text content
        chunk_id: Source chunk ID
        resource_id: Source resource ID
        metadata: Optional metadata
    
    Returns:
        Result dict with extraction stats
    """
    logger.info(f"Building educational KG for chunk {chunk_id}")
    
    # Extract graph
    nodes, relationships = extract_educational_graph(text, metadata)
    
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
    
    # Merge to Neo4j
    stats = merge_graph_to_neo4j(nodes, relationships, chunk_id, resource_id)
    result.update({
        "nodes_merged": stats["nodes"],
        "relationships_merged": stats["relationships"],
        "errors": stats["errors"],
    })
    
    logger.info(
        f"KG built for chunk {chunk_id}: "
        f"{result['nodes_merged']} nodes, "
        f"{result['relationships_merged']} relationships"
    )
    
    return result
