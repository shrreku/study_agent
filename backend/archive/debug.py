#!/usr/bin/env python3
"""Debug why semantic chunker returns 0 chunks."""
import sys
sys.path.insert(0, '/app')

import os
os.environ['USE_LLM_MOCK'] = '0'
os.environ['ENHANCED_CHUNKING_ENABLED'] = 'true'

from parse_utils import extract_text_by_type
import semantic_chunker

# Test PDF
PDF_PATH = "/app/sample/test.pdf"

print("="*60)
print("Debugging Semantic Chunker")
print("="*60)

# Step 1: Extract pages
print("\nStep 1: Extract pages...")
pages = extract_text_by_type(PDF_PATH, None)
print(f"Pages extracted: {len(pages)}")

for i, page in enumerate(pages[:3]):
    print(f"\nPage {i}: {type(page)}, length={len(page)}")
    print(f"Preview: {page[:200]}")

# Step 2: Test individual functions
print("\n" + "="*60)
print("Step 2: Testing semantic chunker internals...")
print("="*60)

# Check what create_semantic_chunks does
print("\nCalling create_semantic_chunks...")
print(f"content_aware=True, preserve_formulas=True")

try:
    # Add some debug to understand what's happening
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    chunks = semantic_chunker.create_semantic_chunks(
        pages,
        content_aware=True,
        preserve_formulas=True
    )
    
    print(f"\nResult: {len(chunks)} chunks")
    
    if not chunks:
        print("\nDEBUG: No chunks created!")
        print("Let me try with simpler parameters...")
        
        chunks2 = semantic_chunker.create_semantic_chunks(
            pages,
            content_aware=False,
            preserve_formulas=False
        )
        print(f"With content_aware=False: {len(chunks2)} chunks")
        
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Try basic chunking
print("\n" + "="*60)
print("Step 3: Testing basic chunker...")
print("="*60)

from chunker import structural_chunk_resource
basic_chunks = structural_chunk_resource(PDF_PATH)
print(f"Basic chunker returned: {len(basic_chunks)} chunks")

if basic_chunks:
    print(f"First chunk keys: {list(basic_chunks[0].keys())}")
