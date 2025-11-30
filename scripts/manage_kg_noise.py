#!/usr/bin/env python3
"""
Knowledge Graph Noise Management CLI.

Commands:
    # List potentially noisy concepts
    python scripts/manage_kg_noise.py list --book-id my_book --threshold 0.5
    
    # Delete specific concepts
    python scripts/manage_kg_noise.py delete concept1 concept2 concept3
    
    # Auto-detect and delete noise
    python scripts/manage_kg_noise.py auto-clean --book-id my_book --dry-run
    
    # Merge concepts
    python scripts/manage_kg_noise.py merge --from "thermal conductivity" --to "conductivity"
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("manage_kg_noise")


def get_neo4j_driver():
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
    return GraphDatabase.driver(uri, auth=(user, password))


def cmd_list(args):
    """List potentially noisy concepts."""
    driver = get_neo4j_driver()
    
    with driver.session() as session:
        # Query for low-confidence, low-frequency concepts
        query = """
            MATCH (c:Concept)
            WHERE ($book_id IS NULL OR c.book_id = $book_id)
            OPTIONAL MATCH (c)-[r]-()
            WITH c, count(r) as rel_count
            WHERE c.confidence < $threshold OR c.frequency <= $min_freq
            RETURN c.canonical_name AS name, 
                   c.display_name AS display,
                   c.confidence AS confidence,
                   c.frequency AS frequency,
                   rel_count AS relationships,
                   c.book_id AS book
            ORDER BY c.confidence ASC, c.frequency ASC
            LIMIT 100
        """
        
        result = session.run(
            query,
            book_id=args.book_id,
            threshold=args.threshold,
            min_freq=args.min_frequency,
        )
        
        concepts = list(result)
        
        if not concepts:
            print("No potentially noisy concepts found.")
            return
        
        print(f"\nPotentially noisy concepts ({len(concepts)}):\n")
        print(f"{'Name':<40} {'Confidence':>10} {'Frequency':>10} {'Rels':>6} {'Book':<20}")
        print("-" * 90)
        
        for c in concepts:
            print(
                f"{c['display'][:40]:<40} "
                f"{c['confidence'] or 0:>10.2f} "
                f"{c['frequency'] or 0:>10} "
                f"{c['relationships']:>6} "
                f"{(c['book'] or '')[:20]:<20}"
            )
    
    driver.close()


def cmd_delete(args):
    """Delete specified concepts."""
    if not args.concepts:
        logger.error("No concepts specified")
        return
    
    driver = get_neo4j_driver()
    
    with driver.session() as session:
        if args.dry_run:
            # Just show what would be deleted
            result = session.run("""
                UNWIND $names AS name
                MATCH (c:Concept)
                WHERE c.canonical_name = name OR c.display_name = name
                OPTIONAL MATCH (c)-[r]-()
                RETURN c.canonical_name AS name, count(r) AS rels
            """, names=args.concepts)
            
            found = list(result)
            print(f"\nWould delete {len(found)} concepts:")
            for c in found:
                print(f"  - {c['name']} ({c['rels']} relationships)")
        else:
            # Actually delete
            result = session.execute_write(lambda tx: tx.run("""
                UNWIND $names AS name
                MATCH (c:Concept)
                WHERE c.canonical_name = name OR c.display_name = name
                DETACH DELETE c
                RETURN count(c) AS deleted
            """, names=args.concepts).single())
            
            deleted = result["deleted"] if result else 0
            print(f"Deleted {deleted} concepts")
    
    driver.close()


def cmd_auto_clean(args):
    """Auto-detect and clean noisy concepts."""
    from backend.ingestion.v2 import NoiseDetector
    
    detector = NoiseDetector(
        min_frequency=args.min_frequency,
        min_confidence=args.threshold,
    )
    
    driver = get_neo4j_driver()
    
    with driver.session() as session:
        # Get all concepts
        result = session.run("""
            MATCH (c:Concept)
            WHERE $book_id IS NULL OR c.book_id = $book_id
            OPTIONAL MATCH (c)-[:RELATED_TO]-(other:Concept)
            RETURN c.canonical_name AS name,
                   c.display_name AS display,
                   c.frequency AS frequency,
                   c.confidence AS confidence,
                   collect(other.canonical_name) AS cooccurring
        """, book_id=args.book_id)
        
        noisy = []
        for record in result:
            is_noise, reason = detector.is_noisy(
                record["display"] or record["name"],
                record["frequency"] or 1,
                record["confidence"] or 0.5,
                record["cooccurring"],
            )
            if is_noise:
                noisy.append({
                    "name": record["name"],
                    "display": record["display"],
                    "reason": reason,
                })
        
        if not noisy:
            print("No noisy concepts detected.")
            driver.close()
            return
        
        print(f"\nDetected {len(noisy)} noisy concepts:\n")
        for c in noisy[:50]:
            print(f"  - {c['display']}: {c['reason']}")
        
        if len(noisy) > 50:
            print(f"  ... and {len(noisy) - 50} more")
        
        if args.dry_run:
            print("\n[DRY RUN] No changes made.")
        else:
            confirm = input(f"\nDelete {len(noisy)} concepts? [y/N]: ")
            if confirm.lower() == "y":
                names = [c["name"] for c in noisy]
                session.execute_write(lambda tx: tx.run("""
                    UNWIND $names AS name
                    MATCH (c:Concept {canonical_name: name})
                    DETACH DELETE c
                """, names=names))
                print(f"Deleted {len(noisy)} noisy concepts")
            else:
                print("Cancelled.")
    
    driver.close()


def cmd_merge(args):
    """Merge one concept into another."""
    driver = get_neo4j_driver()
    
    with driver.session() as session:
        if args.dry_run:
            # Show what would happen
            from_result = session.run("""
                MATCH (c:Concept)
                WHERE c.canonical_name = $name OR c.display_name = $name
                OPTIONAL MATCH (c)-[r]-()
                RETURN c.canonical_name AS name, count(r) AS rels
            """, name=args.from_concept).single()
            
            to_result = session.run("""
                MATCH (c:Concept)
                WHERE c.canonical_name = $name OR c.display_name = $name
                RETURN c.canonical_name AS name
            """, name=args.to_concept).single()
            
            if not from_result:
                print(f"Source concept not found: {args.from_concept}")
                return
            if not to_result:
                print(f"Target concept not found: {args.to_concept}")
                return
            
            print(f"\nWould merge:")
            print(f"  FROM: {from_result['name']} ({from_result['rels']} relationships)")
            print(f"  INTO: {to_result['name']}")
        else:
            # Perform merge
            session.execute_write(lambda tx: tx.run("""
                MATCH (from:Concept)
                WHERE from.canonical_name = $from_name OR from.display_name = $from_name
                MATCH (to:Concept)
                WHERE to.canonical_name = $to_name OR to.display_name = $to_name
                
                // Transfer OCCURS_IN
                OPTIONAL MATCH (from)-[r1:OCCURS_IN]->(chunk:Chunk)
                MERGE (to)-[:OCCURS_IN]->(chunk)
                
                // Transfer PREREQUISITE_OF (outgoing)
                OPTIONAL MATCH (from)-[r2:PREREQUISITE_OF]->(other:Concept)
                WHERE other <> to
                MERGE (to)-[:PREREQUISITE_OF]->(other)
                
                // Transfer PREREQUISITE_OF (incoming)  
                OPTIONAL MATCH (other2:Concept)-[r3:PREREQUISITE_OF]->(from)
                WHERE other2 <> to
                MERGE (other2)-[:PREREQUISITE_OF]->(to)
                
                // Update aliases
                SET to.aliases = coalesce(to.aliases, []) + [from.canonical_name]
                
                // Delete source
                DETACH DELETE from
            """, from_name=args.from_concept, to_name=args.to_concept))
            
            print(f"Merged '{args.from_concept}' into '{args.to_concept}'")
    
    driver.close()


def cmd_stats(args):
    """Show knowledge graph statistics."""
    driver = get_neo4j_driver()
    
    with driver.session() as session:
        # Basic counts
        counts = session.run("""
            MATCH (c:Concept) WITH count(c) AS concepts
            MATCH (ch:Chunk) WITH concepts, count(ch) AS chunks
            MATCH ()-[r:PREREQUISITE_OF]->() WITH concepts, chunks, count(r) AS prereqs
            MATCH ()-[r2:RELATED_TO]-() WITH concepts, chunks, prereqs, count(r2)/2 AS related
            MATCH ()-[r3:TEACHES]->() WITH concepts, chunks, prereqs, related, count(r3) AS teaches
            MATCH ()-[r4:OCCURS_IN]->() 
            RETURN concepts, chunks, prereqs, related, teaches, count(r4) AS occurs
        """).single()
        
        print("\n" + "=" * 50)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("=" * 50)
        print(f"\nNodes:")
        print(f"  Concepts: {counts['concepts']}")
        print(f"  Chunks: {counts['chunks']}")
        print(f"\nRelationships:")
        print(f"  PREREQUISITE_OF: {counts['prereqs']}")
        print(f"  RELATED_TO: {counts['related']}")
        print(f"  TEACHES: {counts['teaches']}")
        print(f"  OCCURS_IN: {counts['occurs']}")
        
        # By book
        books = session.run("""
            MATCH (c:Concept)
            WHERE c.book_id IS NOT NULL
            RETURN c.book_id AS book, count(c) AS concepts
            ORDER BY concepts DESC
            LIMIT 10
        """)
        
        book_list = list(books)
        if book_list:
            print(f"\nBy Book:")
            for b in book_list:
                print(f"  {b['book']}: {b['concepts']} concepts")
        
        # Quality distribution
        quality = session.run("""
            MATCH (c:Concept)
            RETURN 
                sum(CASE WHEN c.confidence >= 0.8 THEN 1 ELSE 0 END) AS high,
                sum(CASE WHEN c.confidence >= 0.5 AND c.confidence < 0.8 THEN 1 ELSE 0 END) AS medium,
                sum(CASE WHEN c.confidence < 0.5 THEN 1 ELSE 0 END) AS low
        """).single()
        
        print(f"\nConfidence Distribution:")
        print(f"  High (≥0.8): {quality['high']}")
        print(f"  Medium (0.5-0.8): {quality['medium']}")
        print(f"  Low (<0.5): {quality['low']}")
        print("=" * 50)
    
    driver.close()


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Noise Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # list command
    list_parser = subparsers.add_parser("list", help="List potentially noisy concepts")
    list_parser.add_argument("--book-id", help="Filter by book ID")
    list_parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    list_parser.add_argument("--min-frequency", type=int, default=1, help="Min frequency")
    
    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete specific concepts")
    delete_parser.add_argument("concepts", nargs="+", help="Concept names to delete")
    delete_parser.add_argument("--dry-run", action="store_true")
    
    # auto-clean command
    auto_parser = subparsers.add_parser("auto-clean", help="Auto-detect and clean noise")
    auto_parser.add_argument("--book-id", help="Filter by book ID")
    auto_parser.add_argument("--threshold", type=float, default=0.5)
    auto_parser.add_argument("--min-frequency", type=int, default=1)
    auto_parser.add_argument("--dry-run", action="store_true")
    
    # merge command
    merge_parser = subparsers.add_parser("merge", help="Merge one concept into another")
    merge_parser.add_argument("--from", dest="from_concept", required=True)
    merge_parser.add_argument("--to", dest="to_concept", required=True)
    merge_parser.add_argument("--dry-run", action="store_true")
    
    # stats command
    subparsers.add_parser("stats", help="Show graph statistics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "list":
        cmd_list(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "auto-clean":
        cmd_auto_clean(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
