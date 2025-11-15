#!/usr/bin/env python3
"""
Validate pedagogy_role tagging quality.

Usage:
    python scripts/validate_pedagogy_tags.py
    python scripts/validate_pedagogy_tags.py --sample-size 20
"""

import argparse
import sys
import os
from typing import Dict, List
from psycopg2.extras import RealDictCursor
import logging

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
if os.path.exists(backend_path):
    sys.path.insert(0, backend_path)
else:
    # Running inside Docker container where backend is /app
    sys.path.insert(0, '/app')

from core.db import get_db_conn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_pedagogy_tags(sample_size: int = 10):
    """Generate validation report for pedagogy tagging.
    
    Args:
        sample_size: Number of random samples to show for manual review
    """
    conn = get_db_conn()
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            print("\n" + "="*80)
            print("PEDAGOGY ROLE TAGGING VALIDATION REPORT")
            print("="*80 + "\n")
            
            # 1. Coverage statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(CASE WHEN tags->>'pedagogy_role' IS NOT NULL THEN 1 END) as tagged_chunks,
                    ROUND(100.0 * COUNT(CASE WHEN tags->>'pedagogy_role' IS NOT NULL THEN 1 END) / NULLIF(COUNT(*), 0), 2) as coverage_pct
                FROM chunk
            """)
            coverage = cur.fetchone()
            
            print("=== Coverage Statistics ===")
            print(f"Total chunks: {coverage['total_chunks']:,}")
            print(f"Tagged chunks: {coverage['tagged_chunks']:,}")
            print(f"Coverage: {coverage['coverage_pct']}%")
            
            if coverage['coverage_pct'] < 95:
                print(f"⚠️  WARNING: Coverage is below 95% target")
            else:
                print(f"✓ Coverage meets 95% target")
            print()
            
            # 2. Role distribution
            cur.execute("""
                SELECT 
                    tags->>'pedagogy_role' as role,
                    COUNT(*) as count,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage
                FROM chunk
                WHERE tags->>'pedagogy_role' IS NOT NULL
                GROUP BY tags->>'pedagogy_role'
                ORDER BY count DESC
            """)
            
            print("=== Role Distribution ===")
            print(f"{'Role':<20} {'Count':>10} {'Percentage':>12}")
            print("-" * 44)
            
            roles = cur.fetchall()
            for row in roles:
                print(f"{row['role']:<20} {row['count']:>10,} {row['percentage']:>11.1f}%")
            
            # Check for imbalance
            if roles:
                max_pct = max(r['percentage'] for r in roles)
                if max_pct > 50:
                    print(f"\n⚠️  WARNING: Role '{roles[0]['role']}' dominates with {max_pct}%")
                else:
                    print(f"\n✓ Role distribution is balanced (no role > 50%)")
            print()
            
            # 3. Tags structure validation
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN jsonb_typeof(tags) = 'object' THEN 1 END) as valid_json,
                    COUNT(CASE WHEN tags = '{}'::jsonb THEN 1 END) as empty_tags
                FROM chunk
                WHERE tags IS NOT NULL
            """)
            structure = cur.fetchone()
            
            print("=== Tags Structure ===")
            print(f"Total chunks with tags: {structure['total']:,}")
            print(f"Valid JSONB objects: {structure['valid_json']:,}")
            print(f"Empty tags: {structure['empty_tags']:,}")
            
            if structure['valid_json'] == structure['total']:
                print("✓ All tags are valid JSONB objects")
            else:
                print(f"⚠️  WARNING: {structure['total'] - structure['valid_json']} chunks have invalid tags")
            print()
            
            # 4. Sample chunks for manual review
            cur.execute(f"""
                SELECT 
                    tags->>'pedagogy_role' as role,
                    LEFT(full_text, 200) as snippet,
                    page_number,
                    section_title
                FROM chunk
                WHERE tags->>'pedagogy_role' IS NOT NULL
                ORDER BY RANDOM()
                LIMIT {sample_size}
            """)
            
            print(f"=== Sample Tagged Chunks (for manual validation, n={sample_size}) ===")
            samples = cur.fetchall()
            for idx, row in enumerate(samples, 1):
                print(f"\n--- Sample {idx} ---")
                print(f"Role: {row['role']}")
                print(f"Page: {row['page_number']}, Section: {row['section_title'] or 'N/A'}")
                print(f"Text: {row['snippet']}...")
            print()
            
            # 5. Chunks without pedagogy_role (for investigation)
            cur.execute("""
                SELECT COUNT(*) as untagged
                FROM chunk
                WHERE tags IS NULL OR tags = '{}'::jsonb OR tags->>'pedagogy_role' IS NULL
            """)
            untagged = cur.fetchone()['untagged']
            
            if untagged > 0:
                print(f"=== Untagged Chunks ===")
                print(f"Chunks without pedagogy_role: {untagged:,}")
                
                # Show a few examples
                cur.execute("""
                    SELECT id, LEFT(full_text, 100) as snippet, page_number
                    FROM chunk
                    WHERE tags IS NULL OR tags = '{}'::jsonb OR tags->>'pedagogy_role' IS NULL
                    LIMIT 5
                """)
                print("\nExamples of untagged chunks:")
                for row in cur.fetchall():
                    print(f"  ID: {row['id']}, Page: {row['page_number']}")
                    print(f"  Text: {row['snippet']}...")
                    print()
            else:
                print("✓ All chunks have pedagogy_role tags")
            
            print("\n" + "="*80)
            print("VALIDATION COMPLETE")
            print("="*80 + "\n")
    
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Validate pedagogy_role tagging quality"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=10, 
        help="Number of random samples to show for manual review (default: 10)"
    )
    
    args = parser.parse_args()
    
    validate_pedagogy_tags(sample_size=args.sample_size)


if __name__ == "__main__":
    main()
