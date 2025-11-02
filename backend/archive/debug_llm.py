#!/usr/bin/env python3
"""Debug LLM calls to understand what's happening."""
import sys
sys.path.insert(0, '/app')

import os
os.environ['USE_LLM_MOCK'] = '0'

from parse_utils import extract_text_by_type
from prompts import get as prompt_get, render as prompt_render
from llm import call_llm_json
import json

PDF_PATH = "/app/sample/test.pdf"

print("="*70)
print("Debugging LLM Calls for Semantic Chunker")
print("="*70)

# Get pages
pages = extract_text_by_type(PDF_PATH, None)
print(f"\nExtracted {len(pages)} pages")
print(f"Page 1 length: {len(pages[1])} characters")

# Get the prompt template
tmpl = prompt_get("ingest.page_structure_v2")
if not tmpl:
    print("\n✗ ERROR: prompt not found!")
    sys.exit(1)

print(f"\n✓ Prompt template found")
print(f"Template length: {len(tmpl)} chars")

# Render the prompt
prompt = prompt_render(tmpl, {
    "page_number": 1,
    "page_text": pages[1][:2000]  # First 2000 chars to fit in context
})

print(f"\n✓ Prompt rendered")
print(f"Final prompt length: {len(prompt)} chars")
print(f"\nPrompt preview (first 500 chars):")
print("-"*70)
print(prompt[:500])
print("...")
print("-"*70)

# Check environment variables
print(f"\nEnvironment:")
print(f"  USE_LLM_MOCK: {os.getenv('USE_LLM_MOCK')}")
print(f"  OPENAI_API_BASE: {os.getenv('OPENAI_API_BASE')}")
print(f"  AIMLAPI_BASE_URL: {os.getenv('AIMLAPI_BASE_URL')}")
print(f"  LLM_MODEL_MINI: {os.getenv('LLM_MODEL_MINI')}")
print(f"  LLM_PREVIEW_MAX_TOKENS: {os.getenv('LLM_PREVIEW_MAX_TOKENS', '800')}")
print(f"  API Key: {os.getenv('AIMLAPI_API_KEY', '')[:10]}...")

# Try calling LLM
print(f"\nCalling LLM...")
default = {"sections": [], "chunks": []}

try:
    result = call_llm_json(prompt, default)
    
    print(f"\n✓ LLM responded!")
    print(f"Response type: {type(result)}")
    print(f"Response keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
    
    if result.get("sections") or result.get("chunks"):
        print(f"\n✓ SUCCESS! Got structured response")
        print(f"  Sections: {len(result.get('sections', []))}")
        print(f"  Chunks: {len(result.get('chunks', []))}")
        
        if result.get("chunks"):
            print(f"\n  First chunk:")
            print(json.dumps(result["chunks"][0], indent=4))
    else:
        print(f"\n✗ ISSUE: Got response but not in expected format")
        print(f"\nFull response:")
        print(json.dumps(result, indent=2))
        
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Try with a simpler prompt
print(f"\n" + "="*70)
print("Testing with simplified prompt...")
print("="*70)

simple_prompt = """Analyze this text and return JSON with sections and chunks.
Text: "Chapter 5 discusses heat transfer. It covers conduction, convection, and radiation."

Return JSON in this format:
{
  "sections": [{"number": "5", "title": "Heat Transfer", "start": 0, "end": 50, "level": 1}],
  "chunks": [{"start": 0, "end": 50, "type": "concept_intro", "cognitive_level": "understand", "difficulty": "introductory"}]
}"""

print(f"Simple prompt length: {len(simple_prompt)} chars")

try:
    simple_result = call_llm_json(simple_prompt, default)
    print(f"\n✓ Simple test responded!")
    print(json.dumps(simple_result, indent=2))
    
    if simple_result.get("sections") or simple_result.get("chunks"):
        print(f"\n✓ Simple test SUCCESS!")
    else:
        print(f"\n✗ Simple test also failed - LLM not following instructions")
        
except Exception as e:
    print(f"\n✗ Simple test ERROR: {e}")
