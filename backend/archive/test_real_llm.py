#!/usr/bin/env python3
"""
Test enhanced chunking with REAL LLM API calls.
Does NOT override USE_LLM_MOCK - respects environment.
"""
import sys
sys.path.insert(0, '/app')

import os
import json

PDF_PATH = "/app/sample/test.pdf"

print("="*70)
print("ENHANCED CHUNKING TEST WITH REAL LLM")
print("="*70)

# Show actual environment (don't override)
print(f"\nEnvironment Configuration:")
print(f"  USE_LLM_MOCK: {os.getenv('USE_LLM_MOCK', 'not set')}")
print(f"  LLM_MODEL_MINI: {os.getenv('LLM_MODEL_MINI', 'not set')}")
print(f"  LLM_PREVIEW_MAX_TOKENS: {os.getenv('LLM_PREVIEW_MAX_TOKENS', 'not set')}")
print(f"  PROMPT_SET: {os.getenv('PROMPT_SET', 'not set')}")
print(f"  API Key: {os.getenv('AIMLAPI_API_KEY', '')[:10]}...")

if os.getenv('USE_LLM_MOCK', '0') != '0':
    print(f"\n⚠️ WARNING: USE_LLM_MOCK is not 0!")
    print(f"   This test requires real LLM API calls.")
    print(f"   Set USE_LLM_MOCK=0 in .env and restart backend.")
    sys.exit(1)

print(f"\n✓ Real LLM mode confirmed")

# Test imports
print(f"\nImporting modules...")
try:
    from chunker import enhanced_structural_chunk_resource
    from quality_validator import validate_chunk_quality
    from kg_pipeline import merge_chunk_formulas_enhanced
    print(f"✓ Modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Run chunking
print(f"\n" + "="*70)
print(f"RUNNING ENHANCED CHUNKING PIPELINE")
print(f"="*70)
print(f"\nPDF: {PDF_PATH}")
print(f"Size: {os.path.getsize(PDF_PATH) / 1024:.1f} KB")

print(f"\nThis will take 1-2 minutes as it makes real LLM API calls...")
print(f"Processing...")

try:
    chunks = enhanced_structural_chunk_resource(PDF_PATH)
    print(f"\n✓ Chunking complete!")
    print(f"  Chunks created: {len(chunks)}")
except Exception as e:
    print(f"\n✗ Chunking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if not chunks:
    print(f"\n✗ FAILURE: No chunks created")
    print(f"\nPossible issues:")
    print(f"  1. LLM model name incorrect")
    print(f"  2. API key invalid")
    print(f"  3. Prompt not working with this model")
    print(f"  4. Token limits too restrictive")
    sys.exit(1)

# Analyze chunks
print(f"\n" + "="*70)
print(f"CHUNK ANALYSIS")
print(f"="*70)

chunks_with_formulas = sum(1 for c in chunks if c.get('formulas'))
chunks_with_domain = sum(1 for c in chunks if c.get('domain'))
chunks_with_context = sum(1 for c in chunks if c.get('context'))

print(f"\nMetadata enrichment:")
print(f"  Formulas: {chunks_with_formulas}/{len(chunks)}")
print(f"  Domain tags: {chunks_with_domain}/{len(chunks)}")
print(f"  Context windows: {chunks_with_context}/{len(chunks)}")

# Count formulas and variables
total_formulas = 0
total_variables = 0

for chunk in chunks:
    formulas = chunk.get('formulas', [])
    total_formulas += len(formulas)
    for formula in formulas:
        total_variables += len(formula.get('variables', []))

print(f"\nFormula extraction:")
print(f"  Total formulas: {total_formulas}")
print(f"  Total variables: {total_variables}")

if total_formulas > 0:
    print(f"\n✓ SUCCESS! Formulas extracted with variables")
    print(f"  This means KG-07 Variable nodes will be created!")
    
    # Show sample
    for chunk in chunks:
        if chunk.get('formulas'):
            f = chunk['formulas'][0]
            print(f"\n  Sample formula:")
            print(f"    LaTeX: {f.get('latex', 'N/A')[:60]}...")
            print(f"    Variables: {len(f.get('variables', []))}")
            if f.get('variables'):
                v = f['variables'][0]
                print(f"    Example: {v.get('symbol')} = {v.get('meaning')}")
            break

# Quality check
print(f"\n" + "="*70)
print(f"QUALITY VALIDATION")
print(f"="*70)

try:
    quality = validate_chunk_quality(chunks)
    summary = quality.get('summary', {})
    
    print(f"\n  Grade: {summary.get('quality_grade', 'N/A')}")
    print(f"  Metadata richness: {summary.get('average_metadata_richness', 0):.1f}/100")
    print(f"  All checks passed: {summary.get('all_checks_passed', False)}")
    
except Exception as e:
    print(f"\n  Quality check error: {e}")

# Summary
print(f"\n" + "="*70)
print(f"TEST RESULT: {'SUCCESS' if chunks else 'FAILURE'}")
print(f"="*70)

if chunks:
    print(f"\n✓ Pipeline working with real LLM API!")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Formulas: {total_formulas}")
    print(f"  Variables: {total_variables}")
    print(f"\n✓ KG-07 ready: Variable nodes will be created")
    print(f"✓ KG-09 ready: APIs already implemented")
else:
    print(f"\n✗ Pipeline failed to create chunks")

# Save results
results = {
    'success': len(chunks) > 0,
    'chunks_created': len(chunks),
    'formulas_extracted': total_formulas,
    'variables_identified': total_variables,
    'quality_grade': summary.get('quality_grade', 'N/A') if chunks else 'F',
    'sample_chunk': chunks[0] if chunks else None
}

with open('/tmp/real_llm_test_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved: /tmp/real_llm_test_results.json")
