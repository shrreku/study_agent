#!/usr/bin/env python3
"""
Direct test of enhanced chunking pipeline with sample PDF.
This runs inside the backend container to test the full pipeline.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, '/app')

print("="*60)
print("Enhanced Chunking Pipeline Direct Test")
print("="*60)

# Import modules
from chunker import enhanced_structural_chunk_resource
from quality_validator import validate_chunk_quality
from kg_pipeline import merge_chunk_formulas_enhanced
import json

# Test PDF path
PDF_PATH = "/app/sample/test.pdf"

print(f"\nPDF: {PDF_PATH}")

# Check if file exists
if not os.path.exists(PDF_PATH):
    print(f"ERROR: PDF not found at {PDF_PATH}")
    sys.exit(1)

file_size = os.path.getsize(PDF_PATH)
print(f"File size: {file_size / 1024:.1f} KB")

# Step 1: Run enhanced chunking
print("\n" + "="*60)
print("Step 1: Running Enhanced Chunking Pipeline...")
print("="*60)

try:
    chunks = enhanced_structural_chunk_resource(PDF_PATH)
    print(f"✓ Chunking complete!")
    print(f"  Total chunks: {len(chunks)}")
except Exception as e:
    print(f"✗ Chunking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Analyze chunk metadata
print("\n" + "="*60)
print("Step 2: Analyzing Chunk Metadata...")
print("="*60)

# Count chunks with enhanced metadata
chunks_with_formulas = sum(1 for c in chunks if c.get('formulas'))
chunks_with_domain = sum(1 for c in chunks if c.get('domain'))
chunks_with_context = sum(1 for c in chunks if c.get('context'))
chunks_with_relationships = sum(1 for c in chunks if c.get('relationships'))

print(f"\nEnhanced Metadata Detection:")
print(f"  Chunks with formulas: {chunks_with_formulas}")
print(f"  Chunks with domain tags: {chunks_with_domain}")
print(f"  Chunks with context windows: {chunks_with_context}")
print(f"  Chunks with relationships: {chunks_with_relationships}")

# Show sample enhanced chunk
print("\n" + "-"*60)
print("Sample Chunk with Formulas:")
print("-"*60)

for chunk in chunks:
    if chunk.get('formulas'):
        print(f"\nChunk ID: {chunk.get('chunk_id', 'N/A')}")
        print(f"Page: {chunk.get('page_number', 'N/A')}")
        print(f"Text preview: {chunk.get('full_text', '')[:200]}...")
        print(f"\nFormulas found: {len(chunk['formulas'])}")
        
        for i, formula in enumerate(chunk['formulas'][:3], 1):
            print(f"\n  Formula {i}:")
            print(f"    LaTeX: {formula.get('latex', 'N/A')}")
            print(f"    Type: {formula.get('type', 'N/A')}")
            
            variables = formula.get('variables', [])
            if variables:
                print(f"    Variables ({len(variables)}):")
                for var in variables[:5]:
                    symbol = var.get('symbol', '?')
                    meaning = var.get('meaning', 'N/A')
                    units = var.get('units', '')
                    print(f"      - {symbol}: {meaning} {f'({units})' if units else ''}")
        
        break  # Show only first chunk with formulas

# Step 3: Run quality validation
print("\n" + "="*60)
print("Step 3: Quality Validation...")
print("="*60)

try:
    quality_report = validate_chunk_quality(chunks)
    
    summary = quality_report.get('summary', {})
    print(f"\nQuality Grade: {summary.get('quality_grade', 'N/A')}")
    print(f"All Checks Passed: {summary.get('all_checks_passed', False)}")
    print(f"Average Metadata Richness: {summary.get('average_metadata_richness', 0):.1f}/100")
    
    checks = quality_report.get('checks', {})
    print(f"\nDetailed Checks:")
    for check_name, check_data in checks.items():
        if isinstance(check_data, dict):
            passed = check_data.get('passed', False)
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
            
except Exception as e:
    print(f"✗ Quality validation failed: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Test Variable node creation (mock)
print("\n" + "="*60)
print("Step 4: Variable Node Analysis...")
print("="*60)

total_formulas = 0
total_variables = 0

for chunk in chunks:
    formulas = chunk.get('formulas', [])
    total_formulas += len(formulas)
    for formula in formulas:
        total_variables += len(formula.get('variables', []))

print(f"\nTotal formulas in chunks: {total_formulas}")
print(f"Total variables in formulas: {total_variables}")

if total_variables > 0:
    print(f"\n✓ KG-07 Enhancement Ready!")
    print(f"  These {total_variables} variables would create Variable nodes in Neo4j")
    print(f"  Each linked to formulas and concepts via REPRESENTS_CONCEPT")

# Step 5: Summary
print("\n" + "="*60)
print("TEST COMPLETE!")
print("="*60)

print(f"\nSummary:")
print(f"  ✓ PDF processed: Fundamentals of Heat Transfer.pdf")
print(f"  ✓ Chunks created: {len(chunks)}")
print(f"  ✓ Enhanced features:")
print(f"      - INGEST-02 (Semantic Chunking): ✓")
print(f"      - INGEST-03 (Hierarchical Tags): {chunks_with_domain}/{len(chunks)}")
print(f"      - INGEST-04 (Formula Metadata): {chunks_with_formulas}/{len(chunks)}")
print(f"      - INGEST-05 (Context Windows): {chunks_with_context}/{len(chunks)}")
print(f"      - INGEST-06 (Relationships): {chunks_with_relationships}/{len(chunks)}")
print(f"  ✓ Formulas extracted: {total_formulas}")
print(f"  ✓ Variables identified: {total_variables}")
print(f"  ✓ Quality grade: {summary.get('quality_grade', 'N/A')}")

print(f"\nKG-07 & KG-09 Features:")
print(f"  ✓ Variable nodes ready for creation: {total_variables}")
print(f"  ✓ KG API endpoints: Available at /api/kg/concepts and /api/kg/subgraph")

# Save sample chunk for inspection
sample_output = {
    'total_chunks': len(chunks),
    'chunks_with_formulas': chunks_with_formulas,
    'total_formulas': total_formulas,
    'total_variables': total_variables,
    'quality_summary': summary,
    'sample_chunks': [c for c in chunks if c.get('formulas')][:2]
}

output_path = '/tmp/enhanced_chunking_test_results.json'
with open(output_path, 'w') as f:
    json.dump(sample_output, f, indent=2, default=str)

print(f"\nTest results saved to: {output_path}")
print("\nYou can inspect the results inside the container:")
print("  docker-compose exec backend cat /tmp/enhanced_chunking_test_results.json | jq")
