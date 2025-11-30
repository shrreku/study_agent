#!/usr/bin/env python3
"""Test the new ingestion pipeline with a sample PDF."""

import os
import sys
import uuid
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def test_pipeline(file_path: str, mode: str = "fast"):
    """Test the ingestion pipeline with a PDF file."""
    from ingestion.pipeline import IngestPipeline, IngestMode, DatabasePool
    
    print(f"\n{'='*60}")
    print(f"Testing Ingestion Pipeline")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    print(f"Mode: {mode}")
    print(f"{'='*60}\n")
    
    # Check file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return
    
    # Generate a test resource ID
    resource_id = str(uuid.uuid4())
    print(f"Resource ID: {resource_id}")
    
    # Create pipeline
    try:
        ingest_mode = IngestMode(mode.lower())
    except ValueError:
        ingest_mode = IngestMode.FAST
    
    pipeline = IngestPipeline(
        resource_id=resource_id,
        file_path=file_path,
        mode=ingest_mode,
    )
    
    # Phase 1: Parse
    print("\n[Phase 1] Parsing document...")
    t0 = time.time()
    try:
        pipeline._phase_parse()
        print(f"  ✓ Created {len(pipeline.chunks)} chunks in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  ✗ Parse failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Show sample chunks
    print("\n  Sample chunks:")
    for i, chunk in enumerate(pipeline.chunks[:3]):
        preview = chunk.full_text[:100].replace('\n', ' ')
        print(f"    [{i+1}] Page {chunk.page_number}: {preview}...")
    
    # Phase 2: Extract
    print("\n[Phase 2] Extracting metadata...")
    t0 = time.time()
    try:
        pipeline._phase_extract()
        print(f"  ✓ Extracted metadata in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  ✗ Extract failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Show sample metadata
    print("\n  Sample metadata:")
    for i, chunk in enumerate(pipeline.chunks[:3]):
        print(f"    [{i+1}] Role: {chunk.pedagogy_role}, Difficulty: {chunk.difficulty}")
        if chunk.concepts:
            print(f"        Concepts: {', '.join(chunk.concepts[:5])}")
    
    # Phase 3: Embed
    print("\n[Phase 3] Computing embeddings...")
    t0 = time.time()
    try:
        pipeline._phase_embed()
        embedded_count = sum(1 for c in pipeline.chunks if c.embedding)
        print(f"  ✓ Embedded {embedded_count}/{len(pipeline.chunks)} chunks in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  ✗ Embed failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Show embedding stats
    if pipeline.chunks and pipeline.chunks[0].embedding:
        dim = len(pipeline.chunks[0].embedding)
        print(f"  Embedding dimension: {dim}")
    
    # Phase 4: Store (skip actual DB write for test)
    print("\n[Phase 4] Store (skipped for test)")
    print("  Set INGEST_TEST_STORE=true to enable database writes")
    
    if os.getenv("INGEST_TEST_STORE", "false").lower() in ("true", "1"):
        print("\n  Storing to databases...")
        t0 = time.time()
        try:
            pipeline._phase_store()
            print(f"  ✓ Stored in {time.time()-t0:.2f}s")
        except Exception as e:
            print(f"  ✗ Store failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total chunks: {len(pipeline.chunks)}")
    
    # Pedagogy role distribution
    roles = {}
    for chunk in pipeline.chunks:
        role = chunk.pedagogy_role
        roles[role] = roles.get(role, 0) + 1
    print(f"\nPedagogy roles:")
    for role, count in sorted(roles.items(), key=lambda x: -x[1]):
        print(f"  {role}: {count}")
    
    # Difficulty distribution
    difficulties = {}
    for chunk in pipeline.chunks:
        diff = chunk.difficulty
        difficulties[diff] = difficulties.get(diff, 0) + 1
    print(f"\nDifficulty levels:")
    for diff, count in sorted(difficulties.items(), key=lambda x: -x[1]):
        print(f"  {diff}: {count}")
    
    # Concept extraction
    all_concepts = set()
    for chunk in pipeline.chunks:
        all_concepts.update(chunk.concepts)
    print(f"\nUnique concepts extracted: {len(all_concepts)}")
    if all_concepts:
        sorted_concepts = sorted(list(all_concepts))
        print(f"Concepts: {', '.join(sorted_concepts[:15])}")
    
    # Show detailed view of first few chunks
    print("\n" + "-"*60)
    print("DETAILED CHUNK VIEW (first 3)")
    print("-"*60)
    for i, chunk in enumerate(pipeline.chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  Page: {chunk.page_number}")
        print(f"  Role: {chunk.pedagogy_role}")
        print(f"  Difficulty: {chunk.difficulty}")
        print(f"  Concepts: {chunk.concepts}")
        print(f"  Text preview: {chunk.full_text[:200].replace(chr(10), ' ')}...")
    
    # Show pedagogical knowledge graph data
    print("\n" + "="*60)
    print("PEDAGOGICAL KNOWLEDGE GRAPH")
    print("="*60)
    
    graph_data = pipeline._build_pedagogical_graph()
    
    # Call LLM to build prerequisites (uses gemini-flash-lite for FAST/STANDARD)
    if graph_data["concepts"]:
        print("\n[Building prerequisites with LLM...]")
        use_lite = mode in ("fast", "standard")
        try:
            llm_prereqs = pipeline._build_prerequisites_llm(graph_data["concepts"], use_lite=use_lite)
            graph_data["prerequisites"].extend(llm_prereqs)
            model_name = "gemini-2.0-flash-lite" if use_lite else "default"
            print(f"  ✓ LLM ({model_name}) extracted {len(llm_prereqs)} prerequisites")
        except Exception as e:
            print(f"  ✗ LLM call failed: {e}")
    
    print(f"\nNodes:")
    print(f"  Concepts: {len(graph_data['concepts'])}")
    print(f"  Chunks: {len(graph_data['chunks'])}")
    
    print(f"\nRelationships:")
    print(f"  PREREQUISITE_OF: {len(graph_data['prerequisites'])}")
    print(f"  RELATED_TO: {len(graph_data['related'])}")
    print(f"  TEACHES: {len(graph_data['teaches'])}")
    print(f"  OCCURS_IN: {len(graph_data['occurrences'])}")
    
    # Show concepts with metadata
    print("\n" + "-"*60)
    print("CONCEPTS WITH PEDAGOGICAL METADATA")
    print("-"*60)
    for canonical, data in list(graph_data['concepts'].items())[:10]:
        print(f"\n  {data['display']}")
        print(f"    Difficulty: {data.get('difficulty', 'unknown')}")
        print(f"    First appears: page {data.get('first_page', '?')}")
        print(f"    Frequency: {data.get('frequency', 0)} mentions")
        print(f"    Primary role: {data.get('primary_role', 'unknown')}")
    
    # Show prerequisite relationships
    if graph_data['prerequisites']:
        print("\n" + "-"*60)
        print("PREREQUISITE RELATIONSHIPS (top 10)")
        print("-"*60)
        for prereq, target, conf, evidence in graph_data['prerequisites'][:10]:
            conf_str = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
            print(f"  {prereq} → {target} (confidence: {conf_str})")
    else:
        print("\n" + "-"*60)
        print("PREREQUISITE RELATIONSHIPS")
        print("-"*60)
        print("  None extracted (LLM call may have failed or no valid prereqs found)")
    
    # Show related concepts
    if graph_data['related']:
        print("\n" + "-"*60)
        print("RELATED CONCEPTS (from co-occurrence)")
        print("-"*60)
        for c1, c2, weight in graph_data['related'][:10]:
            print(f"  {c1} ↔ {c2} (weight: {weight:.2f})")
    
    # Cleanup
    DatabasePool.shutdown()
    
    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ingestion pipeline")
    parser.add_argument("file", nargs="?", help="Path to PDF file")
    parser.add_argument("--mode", default="fast", choices=["fast", "standard", "full"])
    
    args = parser.parse_args()
    
    # Default to sample PDF
    if args.file:
        file_path = args.file
    else:
        file_path = os.path.join(
            os.path.dirname(__file__), "..", 
            "sample", "Fundamentals of Heat Transfer.pdf"
        )
    
    test_pipeline(file_path, args.mode)
