#!/usr/bin/env python3
"""
Backfill pedagogy_role tags for existing chunks.

Usage:
    python scripts/backfill_pedagogy_roles.py --batch-size 100 --mode hybrid
    python scripts/backfill_pedagogy_roles.py --dry-run --limit 10

Modes:
    heuristic: Fast, pattern-based classification
    llm: Accurate, LLM-based classification (expensive)
    hybrid: Heuristic first, LLM for uncertain cases (recommended)
"""

import argparse
import sys
import os
import time
import json
from typing import Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
if os.path.exists(backend_path):
    sys.path.insert(0, backend_path)
else:
    # Running inside Docker container where backend is /app
    sys.path.insert(0, '/app')

from ingestion.hierarchical_tagger import classify_pedagogy_role
from core.db import get_db_conn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backfill_pedagogy_roles(
    batch_size: int = 100,
    mode: str = "hybrid",
    dry_run: bool = False,
    limit: Optional[int] = None,
):
    """
    Backfill pedagogy_role for chunks missing tags.
    
    Args:
        batch_size: Number of chunks to process per batch
        mode: Classification mode (heuristic/llm/hybrid)
        dry_run: If True, don't write to DB
        limit: Process only N chunks (for testing)
    """
    # Set environment variable for LLM classification if needed
    if mode in ["llm", "hybrid"]:
        os.environ["PEDAGOGY_LLM_CLASSIFICATION"] = "true"
    else:
        os.environ["PEDAGOGY_LLM_CLASSIFICATION"] = "false"
    
    conn = get_db_conn()
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Count total chunks needing backfill
            cur.execute(
                """
                SELECT COUNT(*) as total 
                FROM chunk 
                WHERE tags IS NULL OR tags = '{}'::jsonb OR tags->>'pedagogy_role' IS NULL
                """
            )
            total = cur.fetchone()["total"]
            logger.info(f"Total chunks needing backfill: {total}")
            
            if limit:
                total = min(total, limit)
                logger.info(f"Limited to {limit} chunks")
            
            processed = 0
            updated = 0
            errors = 0
            role_counts = {}
            
            offset = 0
            while processed < total:
                # Fetch batch
                query = """
                    SELECT id, full_text, tags, page_number, section_title, section_number, section_level
                    FROM chunk
                    WHERE tags IS NULL OR tags = '{}'::jsonb OR tags->>'pedagogy_role' IS NULL
                    ORDER BY id
                    LIMIT %s OFFSET %s
                """
                cur.execute(query, (batch_size, offset))
                chunks = cur.fetchall()
                
                if not chunks:
                    break
                
                logger.info(f"Processing batch {offset//batch_size + 1}: {len(chunks)} chunks")
                
                for chunk in chunks:
                    try:
                        # Classify pedagogy role
                        full_text = chunk["full_text"] or ""
                        
                        # Build section context
                        section_context = None
                        if chunk.get("section_title"):
                            section_context = {
                                'title': chunk.get("section_title"),
                                'number': chunk.get("section_number"),
                                'level': chunk.get("section_level")
                            }
                        
                        role = classify_pedagogy_role(full_text, section_context)
                        
                        # Track role distribution
                        role_counts[role] = role_counts.get(role, 0) + 1
                        
                        # Update tags
                        existing_tags = chunk.get("tags") or {}
                        if isinstance(existing_tags, str):
                            existing_tags = json.loads(existing_tags)
                        elif not isinstance(existing_tags, dict):
                            existing_tags = {}
                        
                        existing_tags["pedagogy_role"] = role
                        
                        # Write to DB
                        if not dry_run:
                            cur.execute(
                                "UPDATE chunk SET tags = %s, updated_at = NOW() WHERE id = %s",
                                (json.dumps(existing_tags), chunk["id"])
                            )
                            updated += 1
                        
                        processed += 1
                        
                        # Progress logging
                        if processed % 100 == 0:
                            logger.info(f"Progress: {processed}/{total} ({100*processed/total:.1f}%)")
                            logger.info(f"Role distribution so far: {role_counts}")
                    
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk['id']}: {e}")
                        errors += 1
                
                # Commit batch
                if not dry_run:
                    conn.commit()
                    logger.info(f"Committed batch {offset//batch_size + 1}")
                
                offset += batch_size
                
                # Rate limiting for LLM mode
                if mode in ["llm", "hybrid"]:
                    time.sleep(0.5)  # Avoid rate limits
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Backfill complete:")
            logger.info(f"  Processed: {processed}")
            logger.info(f"  Updated: {updated}")
            logger.info(f"  Errors: {errors}")
            logger.info(f"  Dry run: {dry_run}")
            logger.info(f"\nRole distribution:")
            for role, count in sorted(role_counts.items(), key=lambda x: x[1], reverse=True):
                pct = 100 * count / max(1, processed)
                logger.info(f"  {role:15} {count:6} ({pct:5.1f}%)")
            logger.info(f"{'='*60}\n")
    
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Backfill pedagogy_role tags for existing chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run on first 10 chunks
  python scripts/backfill_pedagogy_roles.py --dry-run --limit 10
  
  # Backfill all chunks with heuristic mode (fast)
  python scripts/backfill_pedagogy_roles.py --mode heuristic
  
  # Backfill with hybrid mode (recommended)
  python scripts/backfill_pedagogy_roles.py --mode hybrid --batch-size 50
  
  # Backfill with LLM mode (slow but accurate)
  python scripts/backfill_pedagogy_roles.py --mode llm --batch-size 20
        """
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Chunks per batch (default: 100)")
    parser.add_argument("--mode", choices=["heuristic", "llm", "hybrid"], default="hybrid", 
                       help="Classification mode (default: hybrid)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB, just preview")
    parser.add_argument("--limit", type=int, help="Process only N chunks (for testing)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting backfill with:")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info(f"  Limit: {args.limit or 'None'}")
    logger.info("")
    
    backfill_pedagogy_roles(
        batch_size=args.batch_size,
        mode=args.mode,
        dry_run=args.dry_run,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
