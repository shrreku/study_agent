"""
Integration tests for pedagogy role tagging system.

Tests:
1. Pedagogy role classification accuracy
2. Retrieval filtering by pedagogy role
3. Tags persistence in database
4. Backfill script functionality
"""

import pytest
import json
from typing import Dict, List
from ingestion.hierarchical_tagger import classify_pedagogy_role, extract_hierarchical_tags


class TestPedagogyRoleClassification:
    """Test pedagogy role classification accuracy."""
    
    def test_definition_classification(self):
        """Test that definitions are correctly classified."""
        test_cases = [
            "Heat conduction is defined as the transfer of thermal energy through a material.",
            "We define the Reynolds number as the ratio of inertial forces to viscous forces.",
            "Definition: A vector is a quantity with both magnitude and direction.",
        ]
        
        for text in test_cases:
            role = classify_pedagogy_role(text)
            assert role == "definition", f"Expected 'definition', got '{role}' for: {text[:50]}"
    
    def test_proof_classification(self):
        """Test that proofs are correctly classified."""
        test_cases = [
            "Proof: Let T(x,t) be the temperature distribution. We prove that...",
            "To prove this theorem, we start with the assumption that...",
            "Therefore, by mathematical induction, the statement holds. Q.E.D.",
        ]
        
        for text in test_cases:
            role = classify_pedagogy_role(text)
            assert role == "proof", f"Expected 'proof', got '{role}' for: {text[:50]}"
    
    def test_example_classification(self):
        """Test that examples are correctly classified."""
        test_cases = [
            "For example, consider a metal rod of length L heated at one end.",
            "As an example, let's calculate the heat flux through a copper plate.",
            "For instance, in the case of steady-state conduction...",
        ]
        
        for text in test_cases:
            role = classify_pedagogy_role(text)
            assert role == "example", f"Expected 'example', got '{role}' for: {text[:50]}"
    
    def test_derivation_classification(self):
        """Test that derivations are correctly classified."""
        test_cases = [
            "To derive Fourier's law, we start with the energy balance equation.",
            "Deriving the heat equation from first principles, we obtain...",
            "Starting from Newton's law of cooling, we can show that...",
        ]
        
        for text in test_cases:
            role = classify_pedagogy_role(text)
            assert role == "derivation", f"Expected 'derivation', got '{role}' for: {text[:50]}"
    
    def test_application_classification(self):
        """Test that applications are correctly classified."""
        test_cases = [
            "In practice, heat exchangers are used in power plants to...",
            "Real-world applications of this principle include refrigeration systems.",
            "This method is commonly used in engineering design of thermal systems.",
        ]
        
        for text in test_cases:
            role = classify_pedagogy_role(text)
            assert role == "application", f"Expected 'application', got '{role}' for: {text[:50]}"
    
    def test_problem_classification(self):
        """Test that problems are correctly classified."""
        test_cases = [
            "Problem 1: Calculate the heat flux through a wall of thickness 0.2m.",
            "Exercise: Find the temperature distribution in a cylindrical rod.",
            "Solve for the steady-state temperature profile.",
        ]
        
        for text in test_cases:
            role = classify_pedagogy_role(text)
            assert role == "problem", f"Expected 'problem', got '{role}' for: {text[:50]}"
    
    def test_summary_classification(self):
        """Test that summaries are correctly classified."""
        test_cases = [
            "In summary, we have covered three main heat transfer mechanisms.",
            "To summarize, the key points of this chapter are...",
            "In conclusion, Fourier's law provides a fundamental relationship.",
        ]
        
        for text in test_cases:
            role = classify_pedagogy_role(text)
            assert role == "summary", f"Expected 'summary', got '{role}' for: {text[:50]}"
    
    def test_explanation_default(self):
        """Test that ambiguous text defaults to explanation."""
        text = "The temperature distribution in a solid depends on thermal conductivity."
        role = classify_pedagogy_role(text)
        assert role == "explanation", f"Expected 'explanation', got '{role}'"
    
    def test_section_context_influence(self):
        """Test that section context influences classification."""
        text = "Consider a metal rod heated at one end."
        
        # Without context - might be classified as explanation
        role_no_context = classify_pedagogy_role(text, None)
        
        # With "Example" section context - should be example
        section_context = {"title": "Example 3.1: Heat Conduction in a Rod"}
        role_with_context = classify_pedagogy_role(text, section_context)
        
        assert role_with_context == "example", f"Expected 'example' with section context, got '{role_with_context}'"


class TestHierarchicalTagsIntegration:
    """Test integration of pedagogy_role into hierarchical tags."""
    
    def test_pedagogy_role_in_hierarchical_tags(self):
        """Test that pedagogy_role is included in hierarchical tags."""
        text = "Heat conduction is defined as the transfer of thermal energy."
        tags = extract_hierarchical_tags(text)
        
        assert "pedagogy_role" in tags, "pedagogy_role should be in hierarchical tags"
        # Note: LLM might return different role than heuristic, so just check it's valid
        valid_roles = ["definition", "explanation", "example", "derivation", "proof", "application", "problem", "summary"]
        assert tags["pedagogy_role"] in valid_roles, f"pedagogy_role should be valid, got '{tags['pedagogy_role']}'"
    
    def test_default_pedagogy_role(self):
        """Test that default pedagogy_role is set when unclear."""
        text = "The temperature varies with position and time."
        tags = extract_hierarchical_tags(text)
        
        assert "pedagogy_role" in tags, "pedagogy_role should always be present"
        assert tags["pedagogy_role"] in ["explanation", "definition", "example", "derivation", "proof", "application", "problem", "summary"], \
            f"pedagogy_role should be a valid role, got '{tags['pedagogy_role']}'"


class TestPedagogyRoleRetrieval:
    """Test retrieval filtering by pedagogy role (requires database)."""
    
    @pytest.mark.skip(reason="Requires database setup")
    def test_retrieval_with_pedagogy_filtering(self):
        """Test that retrieval filters by pedagogy role correctly."""
        # This would require:
        # 1. Insert test chunks with known pedagogy roles
        # 2. Query with pedagogy_roles filter
        # 3. Verify only appropriate roles returned
        pass
    
    @pytest.mark.skip(reason="Requires database setup")
    def test_pedagogy_role_boosting(self):
        """Test that matching pedagogy roles are scored higher."""
        # This would require:
        # 1. Insert chunks with different pedagogy roles
        # 2. Query with desired_roles parameter
        # 3. Verify matching roles have higher scores
        pass


class TestTagsPersistence:
    """Test that tags are correctly persisted to database."""
    
    @pytest.mark.skip(reason="Requires database setup")
    def test_tags_column_exists(self):
        """Test that tags column exists in chunk table."""
        # Query database schema to verify tags column exists
        pass
    
    @pytest.mark.skip(reason="Requires database setup")
    def test_tags_jsonb_format(self):
        """Test that tags are stored as valid JSONB."""
        # Insert chunk with tags, retrieve, verify JSONB format
        pass
    
    @pytest.mark.skip(reason="Requires database setup")
    def test_pedagogy_role_index(self):
        """Test that pedagogy_role index exists and is used."""
        # Query with pedagogy_role filter, check query plan uses index
        pass


class TestBackfillScript:
    """Test backfill script functionality."""
    
    @pytest.mark.skip(reason="Requires database setup")
    def test_backfill_dry_run(self):
        """Test that dry run doesn't modify database."""
        # Run backfill with dry_run=True
        # Verify no changes to database
        pass
    
    @pytest.mark.skip(reason="Requires database setup")
    def test_backfill_coverage(self):
        """Test that backfill covers all chunks."""
        # Run backfill
        # Verify all chunks have pedagogy_role
        pass
    
    @pytest.mark.skip(reason="Requires database setup")
    def test_backfill_idempotent(self):
        """Test that backfill can be run multiple times safely."""
        # Run backfill twice
        # Verify same results
        pass


# Run tests with: pytest backend/tests/test_pedagogy_tagging.py -v
