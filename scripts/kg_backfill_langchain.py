#!/usr/bin/env python3
"""Backfill knowledge graph using LangChain GraphTransformer.

This script uses LangChain's LLMGraphTransformer to extract rich knowledge
graphs from educational content and merge them into Neo4j.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add backend to path for imports
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
    from kg_pipeline.graph_builder import build_educational_kg
else:
    from backend.core.db import get_db_conn
    from backend.kg_pipeline.graph_builder import build_educational_kg


def _chunk_meta_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from chunk row."""
    section_level = row.get("section_level")
    if section_level is not None:
        try:
            section_level = int(section_level)
        except (TypeError, ValueError):
            section_level = None
    
    return {
        "chunk_type": row.get("chunk_type"),
        "section_title": row.get("section_title"),
        "section_number": row.get("section_number"),
        "section_path": row.get("section_path") or [],
        "section_level": section_level,
        "page_number": row.get("page_number"),
        "title": row.get("section_title"),
        "resource_id": row.get("resource_id"),
    }


def run_backfill(
    resource_id: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Run LangChain-based KG backfill.
    
    Args:
        resource_id: Optional resource UUID to limit processing
        limit: Optional max number of chunks to process
        dry_run: If True, extract but don't write to Neo4j
    
    Returns:
        Stats dict with processing counts
    """
    stats = {
        "processed": 0,
        "skipped_empty_text": 0,
        "nodes_extracted": 0,
        "relationships_extracted": 0,
        "nodes_merged": 0,
        "relationships_merged": 0,
        "errors": 0,
    }
    
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # Build query
            query = """
                SELECT id::text as chunk_id, resource_id::text, full_text, chunk_type,
                       section_title, section_number, section_path, section_level, page_number
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
            rows = cur.fetchall()
            
            logging.info(f"Processing {len(rows)} chunks with LangChain GraphTransformer")
            
            for row in rows:
                chunk_id = row[0]
                resource = row[1]
                text = row[2]
                
                if not text or not text.strip():
                    stats["skipped_empty_text"] += 1
                    continue
                
                # Build metadata
                chunk_meta = _chunk_meta_from_row({
                    "chunk_type": row[3],
                    "section_title": row[4],
                    "section_number": row[5],
                    "section_path": row[6],
                    "section_level": row[7],
                    "page_number": row[8],
                    "resource_id": resource,
                })
                
                try:
                    if dry_run:
                        from kg_pipeline.graph_builder import extract_educational_graph
                        nodes, rels = extract_educational_graph(text, chunk_meta)
                        stats["nodes_extracted"] += len(nodes)
                        stats["relationships_extracted"] += len(rels)
                        logging.info(
                            f"[DRY RUN] Chunk {chunk_id}: {len(nodes)} nodes, {len(rels)} relationships"
                        )
                    else:
                        result = build_educational_kg(text, chunk_id, resource, chunk_meta)
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
    finally:
        conn.close()
    
    return stats


def main():
    """Main entry point."""
    load_dotenv(find_dotenv(), override=False)
    
    parser = argparse.ArgumentParser(
        description="Backfill KG using LangChain GraphTransformer"
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
        help="Process only the first N chunks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract but don't write to Neo4j",
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
        f"Starting LangChain KG backfill: "
        f"resource={args.resource_id} limit={args.limit} dry_run={args.dry_run}"
    )
    
    stats = run_backfill(
        resource_id=args.resource_id,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    
    logging.info(f"LangChain KG backfill complete: {stats}")
    
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
