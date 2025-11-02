"""Knowledge Graph query API endpoints."""
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from core.auth import require_auth
from kg_pipeline.base import managed_driver

router = APIRouter()
logger = logging.getLogger(__name__)


class ConceptSearchResult(BaseModel):
    """Result for concept search."""
    canonical_name: str
    display_name: str
    node_type: str
    aliases: List[str] = []
    last_seen: Optional[str] = None


class Node(BaseModel):
    """Knowledge graph node."""
    id: str
    type: str
    properties: Dict[str, Any] = {}


class Edge(BaseModel):
    """Knowledge graph edge."""
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = {}


class SubgraphResult(BaseModel):
    """Result for subgraph query."""
    center: Node
    nodes: List[Node] = []
    edges: List[Edge] = []
    node_count: int = 0
    edge_count: int = 0


@router.get("/api/kg/concepts", response_model=List[ConceptSearchResult])
async def search_concepts(
    q: str = Query(..., description="Search query for concept names"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    token: str = Depends(require_auth),
):
    """Search for concepts by name, alias, or canonical name.
    
    Args:
        q: Search query string
        limit: Maximum number of results (default: 20, max: 100)
        token: Auth token
        
    Returns:
        List of matching concepts
    """
    logger.info(f"Searching concepts: q={q}, limit={limit}")
    
    results = []
    
    with managed_driver() as driver:
        if driver is None:
            raise HTTPException(status_code=503, detail="Neo4j connection unavailable")
        
        try:
            with driver.session() as session:
                # Search across canonical_name, display_name, and name_lower
                query = """
                MATCH (c:Concept)
                WHERE c.canonical_name CONTAINS toLower($query)
                   OR c.display_name CONTAINS $query
                   OR c.name_lower CONTAINS toLower($query)
                
                // Get aliases
                OPTIONAL MATCH (c)-[:ALIAS_OF]-(alias:Concept)
                
                RETURN c.canonical_name as canonical_name,
                       c.display_name as display_name,
                       labels(c)[0] as node_type,
                       c.last_seen as last_seen,
                       collect(DISTINCT alias.display_name) as aliases
                ORDER BY 
                    CASE 
                        WHEN c.canonical_name = toLower($query) THEN 0
                        WHEN c.display_name = $query THEN 1
                        WHEN c.canonical_name STARTS WITH toLower($query) THEN 2
                        WHEN c.display_name STARTS WITH $query THEN 3
                        ELSE 4
                    END,
                    c.display_name
                LIMIT $limit
                """
                
                result = session.run(query, query=q, limit=limit)
                
                for record in result:
                    results.append(ConceptSearchResult(
                        canonical_name=record["canonical_name"] or "",
                        display_name=record["display_name"] or "",
                        node_type=record["node_type"] or "Concept",
                        aliases=[a for a in record["aliases"] if a],
                        last_seen=str(record["last_seen"]) if record["last_seen"] else None,
                    ))
                
        except Exception as e:
            logger.exception(f"Concept search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    logger.info(f"Found {len(results)} concepts")
    return results


@router.get("/api/kg/subgraph", response_model=SubgraphResult)
async def get_concept_subgraph(
    center: str = Query(..., description="Central concept name (canonical or display)"),
    depth: int = Query(1, ge=1, le=3, description="Depth of subgraph traversal"),
    relations: Optional[str] = Query(None, description="Comma-separated relation types to include (optional)"),
    token: str = Depends(require_auth),
):
    """Get subgraph centered on a concept.
    
    Args:
        center: Central concept name
        depth: How many hops to traverse (1-3)
        relations: Optional comma-separated list of relation types to filter
        token: Auth token
        
    Returns:
        Subgraph with nodes and edges
    """
    logger.info(f"Getting subgraph: center={center}, depth={depth}, relations={relations}")
    
    # Parse relation types if provided
    relation_types = []
    if relations:
        relation_types = [r.strip().upper() for r in relations.split(",") if r.strip()]
    
    with managed_driver() as driver:
        if driver is None:
            raise HTTPException(status_code=503, detail="Neo4j connection unavailable")
        
        try:
            with driver.session() as session:
                # First find the center node
                find_center_query = """
                MATCH (c:Concept)
                WHERE c.canonical_name = toLower($center)
                   OR c.display_name = $center
                RETURN c
                LIMIT 1
                """
                
                center_result = session.run(find_center_query, center=center)
                center_record = center_result.single()
                
                if not center_record:
                    raise HTTPException(status_code=404, detail=f"Concept '{center}' not found")
                
                center_node_raw = center_record["c"]
                center_node = Node(
                    id=center_node_raw.get("canonical_name", ""),
                    type=list(center_node_raw.labels)[0] if center_node_raw.labels else "Concept",
                    properties=dict(center_node_raw),
                )
                
                # Build subgraph query
                if relation_types:
                    # Filter by specific relation types
                    rel_pattern = "|".join(relation_types)
                    path_pattern = f"(c)-[r:{rel_pattern}*1..{depth}]-(related)"
                else:
                    # Include all relation types
                    path_pattern = f"(c)-[r*1..{depth}]-(related)"
                
                subgraph_query = f"""
                MATCH (c:Concept)
                WHERE c.canonical_name = toLower($center)
                   OR c.display_name = $center
                
                OPTIONAL MATCH path = {path_pattern}
                
                WITH c, 
                     collect(DISTINCT related) as related_nodes,
                     collect(DISTINCT relationships(path)) as path_rels
                
                // Flatten relationships
                UNWIND path_rels as rel_list
                UNWIND rel_list as rel
                
                WITH c, related_nodes, collect(DISTINCT rel) as all_rels
                
                RETURN c as center,
                       related_nodes,
                       all_rels
                """
                
                result = session.run(subgraph_query, center=center, depth=depth)
                record = result.single()
                
                if not record:
                    return SubgraphResult(
                        center=center_node,
                        nodes=[],
                        edges=[],
                        node_count=0,
                        edge_count=0,
                    )
                
                # Process related nodes
                nodes = []
                related_nodes_raw = record["related_nodes"] or []
                
                for node_raw in related_nodes_raw:
                    if node_raw is None:
                        continue
                    
                    nodes.append(Node(
                        id=node_raw.get("canonical_name") or node_raw.get("display_name", ""),
                        type=list(node_raw.labels)[0] if node_raw.labels else "Node",
                        properties=dict(node_raw),
                    ))
                
                # Process edges
                edges = []
                rels_raw = record["all_rels"] or []
                
                for rel_raw in rels_raw:
                    if rel_raw is None:
                        continue
                    
                    # Get source and target from relationship
                    source_node = rel_raw.start_node
                    target_node = rel_raw.end_node
                    
                    edges.append(Edge(
                        source=source_node.get("canonical_name") or source_node.get("display_name", ""),
                        target=target_node.get("canonical_name") or target_node.get("display_name", ""),
                        type=rel_raw.type,
                        properties=dict(rel_raw),
                    ))
                
                return SubgraphResult(
                    center=center_node,
                    nodes=nodes,
                    edges=edges,
                    node_count=len(nodes),
                    edge_count=len(edges),
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Subgraph query failed: {e}")
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
