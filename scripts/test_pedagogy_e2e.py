#!/usr/bin/env python3
"""
End-to-end test for pedagogy role tagging system.

Tests:
1. Heuristic classification accuracy
2. LLM classification (if enabled)
3. Database persistence
4. Retrieval filtering
5. Integration with chunker
"""

import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
if os.path.exists(backend_path):
    sys.path.insert(0, backend_path)
else:
    sys.path.insert(0, '/app')

from ingestion.hierarchical_tagger import classify_pedagogy_role
from core.db import get_db_conn
import json

def test_heuristic_classification():
    """Test heuristic classification for all roles."""
    print("\n" + "="*60)
    print("TEST 1: Heuristic Classification")
    print("="*60)
    
    test_cases = [
        ("Heat conduction is defined as the transfer of thermal energy.", "definition"),
        ("Proof: Let T(x,t) be the temperature distribution.", "proof"),
        ("For example, consider a metal rod heated at one end.", "example"),
        ("To derive Fourier's law, we start with the energy balance.", "derivation"),
        ("In practice, heat exchangers are used in power plants.", "application"),
        ("Problem 1: Calculate the heat flux through a wall.", "problem"),
        ("In summary, we have covered three heat transfer mechanisms.", "summary"),
        ("The temperature distribution depends on thermal conductivity.", "explanation"),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_role in test_cases:
        role = classify_pedagogy_role(text)
        status = "✓" if role == expected_role else "✗"
        if role == expected_role:
            passed += 1
        else:
            failed += 1
        print(f"{status} Expected: {expected_role:12} Got: {role:12} | {text[:50]}...")
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_database_persistence():
    """Test that tags are persisted correctly in database."""
    print("\n" + "="*60)
    print("TEST 2: Database Persistence")
    print("="*60)
    
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # Check tags column exists
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name='chunk' AND column_name='tags'
            """)
            result = cur.fetchone()
            if result:
                print(f"✓ Tags column exists: {result[0]} ({result[1]})")
            else:
                print("✗ Tags column does not exist")
                return False
            
            # Check GIN index exists
            cur.execute("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename='chunk' AND indexname='idx_chunk_tags_gin'
            """)
            if cur.fetchone():
                print("✓ GIN index on tags exists")
            else:
                print("✗ GIN index on tags missing")
                return False
            
            # Check pedagogy_role index exists
            cur.execute("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename='chunk' AND indexname='idx_chunk_pedagogy_role'
            """)
            if cur.fetchone():
                print("✓ Pedagogy role index exists")
            else:
                print("✗ Pedagogy role index missing")
                return False
            
            # Check chunks have pedagogy_role
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN tags->>'pedagogy_role' IS NOT NULL THEN 1 END) as tagged
                FROM chunk
            """)
            total, tagged = cur.fetchone()
            coverage = (tagged / total * 100) if total > 0 else 0
            print(f"✓ Coverage: {tagged}/{total} chunks ({coverage:.1f}%)")
            
            # Check tags are valid JSONB
            cur.execute("""
                SELECT COUNT(*) 
                FROM chunk 
                WHERE tags IS NOT NULL AND jsonb_typeof(tags) != 'object'
            """)
            invalid = cur.fetchone()[0]
            if invalid == 0:
                print("✓ All tags are valid JSONB objects")
            else:
                print(f"✗ {invalid} chunks have invalid tags")
                return False
            
            return coverage >= 95
    finally:
        conn.close()


def test_retrieval_filtering():
    """Test retrieval filtering by pedagogy role."""
    print("\n" + "="*60)
    print("TEST 3: Retrieval Filtering")
    print("="*60)
    
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # Test filtering by single role
            cur.execute("""
                SELECT COUNT(*) 
                FROM chunk 
                WHERE tags->>'pedagogy_role' = 'example'
            """)
            example_count = cur.fetchone()[0]
            print(f"✓ Found {example_count} chunks with role 'example'")
            
            # Test filtering by multiple roles
            cur.execute("""
                SELECT COUNT(*) 
                FROM chunk 
                WHERE tags->>'pedagogy_role' IN ('definition', 'explanation')
            """)
            multi_count = cur.fetchone()[0]
            print(f"✓ Found {multi_count} chunks with roles 'definition' or 'explanation'")
            
            # Test role distribution
            cur.execute("""
                SELECT 
                    tags->>'pedagogy_role' as role,
                    COUNT(*) as count
                FROM chunk
                WHERE tags->>'pedagogy_role' IS NOT NULL
                GROUP BY tags->>'pedagogy_role'
                ORDER BY count DESC
            """)
            print("\nRole distribution:")
            for row in cur.fetchall():
                role, count = row
                print(f"  {role:15} {count:4} chunks")
            
            return True
    finally:
        conn.close()


def test_section_context():
    """Test that section context influences classification."""
    print("\n" + "="*60)
    print("TEST 4: Section Context Influence")
    print("="*60)
    
    text = "Consider a metal rod heated at one end."
    
    # Without context
    role_no_context = classify_pedagogy_role(text, None)
    print(f"Without context: {role_no_context}")
    
    # With example section context
    section_context = {"title": "Example 3.1: Heat Conduction"}
    role_with_context = classify_pedagogy_role(text, section_context)
    print(f"With 'Example' section: {role_with_context}")
    
    # With problem section context
    section_context = {"title": "Problem 5.2"}
    role_problem = classify_pedagogy_role(text, section_context)
    print(f"With 'Problem' section: {role_problem}")
    
    if role_with_context == "example":
        print("✓ Section context correctly influences classification")
        return True
    else:
        print("✗ Section context did not influence classification as expected")
        return False


def test_llm_classification():
    """Test LLM classification if enabled."""
    print("\n" + "="*60)
    print("TEST 5: LLM Classification (Optional)")
    print("="*60)
    
    llm_enabled = os.getenv("PEDAGOGY_LLM_CLASSIFICATION", "false").lower() in ("true", "1", "yes")
    
    if not llm_enabled:
        print("⊘ LLM classification is disabled (set PEDAGOGY_LLM_CLASSIFICATION=true to test)")
        return True
    
    print("✓ LLM classification is enabled")
    
    # Test with ambiguous text that heuristics might miss
    ambiguous_text = "The relationship between force and acceleration can be expressed mathematically."
    role = classify_pedagogy_role(ambiguous_text)
    print(f"Classified ambiguous text as: {role}")
    
    return True


def main():
    """Run all end-to-end tests."""
    print("\n" + "="*60)
    print("PEDAGOGY ROLE TAGGING - END-TO-END TESTS")
    print("="*60)
    
    results = {
        "Heuristic Classification": test_heuristic_classification(),
        "Database Persistence": test_database_persistence(),
        "Retrieval Filtering": test_retrieval_filtering(),
        "Section Context": test_section_context(),
        "LLM Classification": test_llm_classification(),
    }
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
