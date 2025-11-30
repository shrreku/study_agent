#!/usr/bin/env python3
"""
CLI for Ingestion Pipeline v2 - Multi-book, Ontology-aware ingestion.

Usage:
    # Single book ingestion
    python scripts/ingest_v2.py sample/hcv11th_part1.pdf --mode standard --domain Physics
    
    # Multi-book ingestion with ontology building
    python scripts/ingest_v2.py book1.pdf book2.pdf book3.pdf --domain "Heat Transfer" --build-ontology
    
    # Fast mode (no LLM, heuristics only)
    python scripts/ingest_v2.py book.pdf --mode fast
    
    # Custom page range
    python scripts/ingest_v2.py book.pdf --page-range 1-50
    
    # Dry run (parse only, no storage)
    python scripts/ingest_v2.py book.pdf --dry-run
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.ingestion.v2 import (
    IngestMode,
    IngestConfig,
    IngestPipelineV2,
    MultiBookIngestor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_v2")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest textbooks into ontology-aware knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "files",
        nargs="+",
        help="PDF files to ingest",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "standard", "full"],
        default="standard",
        help="Ingestion mode (default: standard)",
    )
    parser.add_argument(
        "--domain",
        default="STEM",
        help="Domain hint for extraction (default: STEM)",
    )
    parser.add_argument(
        "--ocr",
        default="auto",
        choices=["auto", "pymupdf", "gemini", "openai", "hybrid"],
        help="OCR backend (default: auto)",
    )
    parser.add_argument(
        "--page-range",
        type=str,
        help="Page range to process (e.g., 1-50)",
    )
    parser.add_argument(
        "--build-ontology",
        action="store_true",
        help="Build cross-book ontology after ingestion",
    )
    parser.add_argument(
        "--no-neo4j",
        action="store_true",
        help="Skip Neo4j storage",
    )
    parser.add_argument(
        "--no-postgres",
        action="store_true",
        help="Skip Postgres storage",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding generation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and extract only, no storage",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate files
    for f in args.files:
        if not os.path.exists(f):
            logger.error(f"File not found: {f}")
            sys.exit(1)
        if not f.lower().endswith(".pdf"):
            logger.warning(f"Non-PDF file: {f}")
    
    # Build config
    config = IngestConfig(
        mode=IngestMode(args.mode),
        domain=args.domain,
        ocr_backend=args.ocr,
        store_neo4j=not (args.no_neo4j or args.dry_run),
        store_postgres=not (args.no_postgres or args.dry_run),
        embed_chunks=not args.no_embed,
    )
    
    if args.page_range:
        try:
            start, end = map(int, args.page_range.split("-"))
            config.page_range = (start, end)
        except ValueError:
            logger.error(f"Invalid page range: {args.page_range}")
            sys.exit(1)
    
    # Run ingestion
    results = []
    
    if len(args.files) == 1:
        logger.info(f"Single-book ingestion: {args.files[0]}")
        pipeline = IngestPipelineV2(config)
        result = pipeline.ingest(args.files[0])
        results.append(result)
    else:
        logger.info(f"Multi-book ingestion: {len(args.files)} files")
        ingestor = MultiBookIngestor(config)
        results = ingestor.ingest_books(
            args.files,
            domain=args.domain,
            build_ontology=args.build_ontology,
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    
    total_chunks = 0
    total_concepts = 0
    total_rels = 0
    
    for r in results:
        status = "✓" if not r.errors else "⚠"
        print(f"\n{status} {r.book_id}")
        print(f"   Pages: {r.pages_processed}")
        print(f"   Chunks: {r.chunks_created}")
        print(f"   Concepts: {r.concepts_extracted}")
        print(f"   Relationships: {r.relationships_created}")
        print(f"   Duration: {r.duration_seconds:.1f}s")
        if r.errors:
            for e in r.errors:
                print(f"   ERROR: {e}")
        
        total_chunks += r.chunks_created
        total_concepts += r.concepts_extracted
        total_rels += r.relationships_created
    
    print("\n" + "-" * 60)
    print(f"TOTAL: {total_chunks} chunks, {total_concepts} concepts, {total_rels} relationships")
    print("=" * 60)
    
    # Save results to JSON
    if args.output:
        output_data = {
            "mode": args.mode,
            "domain": args.domain,
            "results": [
                {
                    "resource_id": r.resource_id,
                    "book_id": r.book_id,
                    "pages_processed": r.pages_processed,
                    "chunks_created": r.chunks_created,
                    "concepts_extracted": r.concepts_extracted,
                    "relationships_created": r.relationships_created,
                    "duration_seconds": r.duration_seconds,
                    "errors": r.errors,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
