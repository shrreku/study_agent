#!/usr/bin/env python3
"""Enhanced KG backfill with section-level aggregation and better extraction.

Improvements:
- Section-level processing (aggregate chunks by section for context)
- Enhanced educational prompts
- Noise filtering
- Multi-stage extraction
"""
import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Add backend to path
ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))
APP_DIR = Path("/app")
if APP_DIR.exists() and str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from dotenv import find_dotenv, load_dotenv

if (APP_DIR / "core").exists():
    from core.db import get_db_conn
    from kg_pipeline.enhanced_graph_builder import build_enhanced_educational_kg
else:
    from backend.core.db import get_db_conn
    from backend.kg_pipeline.enhanced_graph_builder import build_enhanced_educational_kg


def _get_chunks(resource_id: str | None, limit: int | None) -> List[Dict[str, Any]]:
    """Fetch chunks from database."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    id::text as chunk_id, 
                    resource_id::text,
                    full_text,
                    chunk_type,
                    section_title,
                    section_number,
                    section_path,
                    section_level,
                    page_number,
                    source_offset
                FROM chunk
                WHERE full_text IS NOT NULL AND full_text != ''
            """
            params = []
            
            if resource_id:
                query += " AND resource_id = %s::uuid"
                params.append(resource_id)
            
            query += " ORDER BY resource_id, page_number, source_offset"
            
            if limit:
                query += f" LIMIT {int(limit)}"
            
            cur.execute(query, params)
            
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


def _aggregate_by_section(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate chunks by section for richer context.
    
    Returns list of aggregated section dicts with combined text.
    """
    sections = defaultdict(lambda: {
        "chunks": [],
        "text_parts": [],
        "chunk_ids": [],
    })
    
    for chunk in chunks:
        section_key = (
            chunk.get("section_title") or "unknown",
            chunk.get("section_number") or "0"
        )
        
        sections[section_key]["chunks"].append(chunk)
        sections[section_key]["text_parts"].append(chunk["full_text"])
        sections[section_key]["chunk_ids"].append(chunk["chunk_id"])
    
    aggregated = []
    for (section_title, section_number), data in sections.items():
        # Combine text from all chunks in section
        combined_text = "\n\n".join(data["text_parts"])
        
        # Use first chunk's metadata
        first_chunk = data["chunks"][0]
        
        aggregated.append({
            "section_title": section_title,
            "section_number": section_number,
            "combined_text": combined_text,
            "chunk_ids": data["chunk_ids"],
            "resource_id": first_chunk["resource_id"],
            "chunk_count": len(data["chunks"]),
            "metadata": {
                "title": section_title,
                "section_title": section_title,
                "section_number": section_number,
                "section_path": first_chunk.get("section_path", []),
                "section_level": first_chunk.get("section_level"),
                "chunk_type": "section_aggregate",
            }
        })
    
    return aggregated


def run_enhanced_backfill(
    resource_id: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    use_sections: bool = True,
) -> Dict[str, int]:
    """Run enhanced KG backfill with section-level aggregation.
    
    Args:
        resource_id: Optional resource UUID to limit processing
        limit: Optional max number of chunks/sections to process
        dry_run: If True, extract but don't write to Neo4j
        use_sections: If True, aggregate chunks by section for richer context
    
    Returns:
        Stats dict with processing counts
    """
    stats = {
        "chunks_fetched": 0,
        "sections_processed": 0,
        "processed": 0,
        "skipped_empty_text": 0,
        "nodes_extracted": 0,
        "relationships_extracted": 0,
        "nodes_merged": 0,
        "relationships_merged": 0,
        "errors": 0,
    }
    
    # Fetch chunks
    logging.info(f"Fetching chunks (resource={resource_id}, limit={limit})...")
    chunks = _get_chunks(resource_id, limit)
    stats["chunks_fetched"] = len(chunks)
    logging.info(f"Fetched {len(chunks)} chunks")
    
    if use_sections:
        # Aggregate by section
        sections = _aggregate_by_section(chunks)
        logging.info(f"Aggregated into {len(sections)} sections")
        
        # Process sections
        for section in sections[:limit] if limit else sections:
            text = section["combined_text"]
            
            if not text or not text.strip():
                stats["skipped_empty_text"] += 1
                continue
            
            # Use first chunk ID as representative
            chunk_id = section["chunk_ids"][0]
            resource = section["resource_id"]
            metadata = section["metadata"]
            
            try:
                if dry_run:
                    from kg_pipeline.enhanced_graph_builder import extract_enhanced_educational_graph
                    nodes, rels = extract_enhanced_educational_graph(text, metadata)
                    stats["nodes_extracted"] += len(nodes)
                    stats["relationships_extracted"] += len(rels)
                    logging.info(
                        f"[DRY RUN] Section '{section['section_title']}': "
                        f"{len(nodes)} nodes, {len(rels)} relationships "
                        f"(from {section['chunk_count']} chunks)"
                    )
                else:
                    result = build_enhanced_educational_kg(text, chunk_id, resource, metadata)
                    stats["nodes_extracted"] += result["nodes_extracted"]
                    stats["relationships_extracted"] += result["relationships_extracted"]
                    stats["nodes_merged"] += result["nodes_merged"]
                    stats["relationships_merged"] += result["relationships_merged"]
                    stats["errors"] += result["errors"]
                
                stats["processed"] += 1
                stats["sections_processed"] += 1
                
                if stats["sections_processed"] % 3 == 0:
                    logging.info(
                        f"Progress: {stats['sections_processed']} sections processed, "
                        f"{stats['nodes_merged']} nodes, "
                        f"{stats['relationships_merged']} relationships merged"
                    )
                
            except Exception as e:
                stats["errors"] += 1
                logging.exception(
                    f"Failed to process section '{section.get('section_title')}'",
                    extra={"error": str(e)}
                )
    else:
        # Process individual chunks
        logging.info(f"Processing {len(chunks)} individual chunks with enhanced extraction")
        
        for chunk in chunks:
            text = chunk["full_text"]
            
            if not text or not text.strip():
                stats["skipped_empty_text"] += 1
                continue
            
            chunk_id = chunk["chunk_id"]
            resource = chunk["resource_id"]
            
            metadata = {
                "chunk_type": chunk.get("chunk_type"),
                "section_title": chunk.get("section_title"),
                "section_number": chunk.get("section_number"),
                "section_path": chunk.get("section_path", []),
                "section_level": chunk.get("section_level"),
                "page_number": chunk.get("page_number"),
                "title": chunk.get("section_title"),
            }
            
            try:
                if dry_run:
                    from kg_pipeline.enhanced_graph_builder import extract_enhanced_educational_graph
                    nodes, rels = extract_enhanced_educational_graph(text, metadata)
                    stats["nodes_extracted"] += len(nodes)
                    stats["relationships_extracted"] += len(rels)
                    logging.info(
                        f"[DRY RUN] Chunk {chunk_id}: {len(nodes)} nodes, {len(rels)} relationships"
                    )
                else:
                    result = build_enhanced_educational_kg(text, chunk_id, resource, metadata)
                    stats["nodes_extracted"] += result["nodes_extracted"]
                    stats["relationships_extracted"] += result["relationships_extracted"]
                    stats["nodes_merged"] += result["nodes_merged"]
                    stats["relationships_merged"] += result["relationships_merged"]
                    stats["errors"] += result["errors"]
                
                stats["processed"] += 1
                
                if stats["processed"] % 5 == 0:
                    logging.info(
                        f"Progress: {stats['processed']} chunks processed, "
                        f"{stats['nodes_merged']} nodes, "
                        f"{stats['relationships_merged']} relationships merged"
                    )
                
            except Exception as e:
                stats["errors"] += 1
                logging.exception(
                    f"Failed to process chunk {chunk_id}",
                    extra={"chunk_id": chunk_id, "error": str(e)}
                )
    
    return stats


def main():
    """Main entry point."""
    load_dotenv(find_dotenv(), override=False)
    
    parser = argparse.ArgumentParser(
        description="Enhanced KG backfill with section-level aggregation"
    )
    parser.add_argument(
        "--resource-id",
        help="Limit to a single resource UUID",
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N items",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract but don't write to Neo4j",
    )
    parser.add_argument(
        "--no-sections",
        action="store_true",
        help="Process individual chunks instead of aggregating by section",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (default INFO)",
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    
    logging.info(
        f"Starting enhanced KG backfill: "
        f"resource={args.resource_id} limit={args.limit} "
        f"dry_run={args.dry_run} use_sections={not args.no_sections}"
    )
    
    stats = run_enhanced_backfill(
        resource_id=args.resource_id,
        limit=args.limit,
        dry_run=args.dry_run,
        use_sections=not args.no_sections,
    )
    
    logging.info(f"Enhanced KG backfill complete: {stats}")
    
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
