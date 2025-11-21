"""
Comprehensive tests for ResponseGenerator (Phase 3: Response Generation).

Tests verify:
1. No OCR artifacts in response text or citations
2. Clean citation structure with proper metadata
3. Separation of response generation from citation building
4. Both grounding modes (llm_integrated, explicit_citation)
5. All response types (explain, ask, hint, reflect, review, etc.)
6. Fallback handling for missing chunks or LLM failures
"""

import pytest
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock

from backend.agents.tutor.response_generator import (
    ResponseGenerator,
    ResponseContext,
)
from backend.agents.tutor.actions.types import Citation, TutorResponse


# ===== Test Fixtures =====

@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    """Sample retrieved chunks with OCR artifacts for testing."""
    return [
        {
            "id": "chunk-001",
            "snippet": "1.2 (cid:4) Physical Origins and Rate Equations of Heat Conduction",
            "pedagogy_role": "definition",
            "score": 0.92,
        },
        {
            "id": "chunk-002",
            "snippet": "Fourier's Law: q = -k∇T (cid:5) where k is thermal conductivity",
            "pedagogy_role": "formula",
            "score": 0.87,
        },
        {
            "id": "chunk-003",
            "snippet": "Example: (cid:4) In steady-state conduction, the temperature gradient remains constant",
            "pedagogy_role": "example",
            "score": 0.78,
        },
    ]


@pytest.fixture
def empty_chunks() -> List[Dict[str, Any]]:
    """Empty chunk list for testing fallback behavior."""
    return []


@pytest.fixture
def response_generator() -> ResponseGenerator:
    """Create a ResponseGenerator instance."""
    return ResponseGenerator(
        default_grounding_mode="llm_integrated",
        citation_limit=3,
    )


@pytest.fixture
def response_context(sample_chunks) -> ResponseContext:
    """Create a sample ResponseContext."""
    return ResponseContext(
        concept="Heat Conduction",
        level="intermediate",
        chunks=sample_chunks,
        message="Can you explain how heat conducts through materials?",
    )


# ===== Citation Building Tests =====

class TestCitationBuilding:
    """Test citation building with OCR artifact cleaning."""
    
    def test_build_citations_cleans_ocr_artifacts(self, response_generator, sample_chunks):
        """Verify OCR artifacts are removed from citation snippets."""
        citations = response_generator._build_citations(sample_chunks)
        
        assert len(citations) == 3
        # Check first citation
        assert citations[0].chunk_id == "chunk-001"
        assert "(cid:" not in citations[0].snippet_text
        assert "§" in citations[0].snippet_text or "Physical Origins" in citations[0].snippet_text
        
        # Check second citation
        assert citations[1].chunk_id == "chunk-002"
        assert "(cid:" not in citations[1].snippet_text
        
        # Check third citation
        assert citations[2].chunk_id == "chunk-003"
        assert "(cid:" not in citations[2].snippet_text
    
    def test_build_citations_preserves_metadata(self, response_generator, sample_chunks):
        """Verify pedagogy_role and relevance_score are preserved."""
        citations = response_generator._build_citations(sample_chunks)
        
        assert citations[0].pedagogy_role == "definition"
        assert abs(citations[0].relevance_score - 0.92) < 0.01
        
        assert citations[1].pedagogy_role == "formula"
        assert abs(citations[1].relevance_score - 0.87) < 0.01
    
    def test_build_citations_respects_limit(self, response_generator, sample_chunks):
        """Verify citation limit is respected."""
        citations = response_generator._build_citations(sample_chunks, limit=2)
        assert len(citations) == 2
        
        citations = response_generator._build_citations(sample_chunks, limit=1)
        assert len(citations) == 1
    
    def test_build_citations_empty_chunks(self, response_generator):
        """Verify empty chunks return empty citations."""
        citations = response_generator._build_citations([])
        assert citations == []
    
    def test_build_citations_handles_missing_fields(self, response_generator):
        """Verify graceful handling of missing chunk fields."""
        chunks = [
            {"id": "c1", "snippet": "Valid chunk"},
            {"snippet": "Missing ID - should skip"},
            {"id": "c3"},  # Missing snippet - should skip
        ]
        
        citations = response_generator._build_citations(chunks)
        assert len(citations) == 1
        assert citations[0].chunk_id == "c1"
    
    def test_build_source_ids(self, response_generator, sample_chunks):
        """Verify source ID extraction."""
        source_ids = response_generator._build_source_ids(sample_chunks)
        assert source_ids == ["chunk-001", "chunk-002", "chunk-003"]


# ===== Explain Response Tests =====

class TestExplainResponse:
    """Test explanation response generation."""
    
    @patch('backend.agents.tutor.response_generator.call_json_chat')
    def test_explain_response_llm_integrated(self, mock_call_json, response_generator, response_context):
        """Test explain response with llm_integrated grounding."""
        mock_call_json.return_value = {
            "response": "Heat conduction is the transfer of thermal energy through direct contact.",
            "confidence": 0.85,
        }
        
        response = response_generator.generate_explain_response(
            response_context,
            grounding_mode="llm_integrated",
        )
        
        assert response.action_type == "explain"
        assert "Heat" in response.response_text
        assert abs(response.confidence - 0.85) < 0.01
        assert len(response.citations) > 0
        assert all("(cid:" not in c.snippet_text for c in response.citations)
    
    @patch('backend.agents.tutor.response_generator.call_json_chat')
    def test_explain_response_explicit_citation(self, mock_call_json, response_generator, response_context):
        """Test explain response with explicit_citation grounding."""
        mock_call_json.return_value = {
            "response": "Pure explanation without chunk references.",
            "confidence": 0.8,
        }
        
        response = response_generator.generate_explain_response(
            response_context,
            grounding_mode="explicit_citation",
        )
        
        assert response.grounding_mode == "explicit_citation"
        assert response.action_type == "explain"
        assert len(response.citations) > 0
    
    def test_explain_response_no_chunks(self, response_generator):
        """Test explain response fallback with no chunks."""
        context = ResponseContext(
            concept="Heat Transfer",
            level="intermediate",
            chunks=[],
        )
        
        response = response_generator.generate_explain_response(context)
        
        assert response.confidence < 0.3
        assert len(response.citations) == 0
        assert response.response_text != ""
    
    @patch('backend.agents.tutor.response_generator.call_json_chat')
    def test_explain_response_llm_failure_fallback(self, mock_call_json, response_generator, response_context):
        """Test fallback when LLM call fails."""
        mock_call_json.side_effect = Exception("LLM service error")
        
        response = response_generator.generate_explain_response(response_context)
        
        assert response.response_text != ""
        assert response.confidence > 0


# ===== Ask (Question) Response Tests =====

class TestAskResponse:
    """Test assessment question generation."""
    
    @patch('backend.agents.tutor.response_generator.call_json_chat')
    def test_ask_response_generates_question(self, mock_call_json, response_generator, response_context):
        """Test question generation."""
        mock_call_json.return_value = {
            "question": "Based on Fourier's Law, what factors determine heat conduction rate?",
            "confidence": 0.88,
        }
        
        response = response_generator.generate_ask_response(response_context)
        
        assert response.action_type == "ask"
        assert "?" in response.response_text
        assert len(response.citations) > 0
        assert abs(response.confidence - 0.88) < 0.01
    
    def test_ask_response_no_chunks_fallback(self, response_generator):
        """Test question generation with no chunks."""
        context = ResponseContext(
            concept="Heat Transfer",
            level="beginner",
            chunks=[],
        )
        
        response = response_generator.generate_ask_response(context)
        
        assert response.action_type == "ask"
        assert "?" in response.response_text
        assert "Heat Transfer" in response.response_text
    
    @patch('backend.agents.tutor.response_generator.call_json_chat')
    def test_ask_response_explicit_citation(self, mock_call_json, response_generator, response_context):
        """Test question with explicit citations."""
        mock_call_json.return_value = {
            "question": "What is the relationship between temperature gradient and heat flux?",
            "confidence": 0.82,
        }
        
        response = response_generator.generate_ask_response(
            response_context,
            grounding_mode="explicit_citation",
        )
        
        assert response.grounding_mode == "explicit_citation"
        assert len(response.citations) > 0


# ===== Hint Response Tests =====

class TestHintResponse:
    """Test hint response generation."""
    
    @patch('backend.agents.tutor.response_generator.call_json_chat')
    def test_hint_response(self, mock_call_json, response_generator, response_context):
        """Test hint generation."""
        mock_call_json.return_value = {
            "response": "Hint: Think about how temperature differences drive heat flow.",
            "confidence": 0.75,
        }
        
        response = response_generator.generate_hint_response(response_context)
        
        assert response.action_type == "hint"
        assert "Hint" in response.response_text
        assert len(response.citations) > 0
    
    def test_hint_response_no_chunks(self, response_generator):
        """Test hint with no chunks."""
        context = ResponseContext(
            concept="Conduction",
            level="intermediate",
            chunks=[],
        )
        
        response = response_generator.generate_hint_response(context)
        
        assert response.action_type == "hint"
        assert response.response_text != ""


# ===== Reflect Response Tests =====

class TestReflectResponse:
    """Test reflection/assessment response generation."""
    
    @patch('backend.agents.tutor.response_generator.call_json_chat')
    def test_reflect_response(self, mock_call_json, response_generator, response_context):
        """Test reflection prompt generation."""
        mock_call_json.return_value = {
            "response": "Can you walk me through how you would apply Fourier's Law to this problem?",
            "confidence": 0.80,
        }
        
        response = response_generator.generate_reflect_response(response_context)
        
        assert response.action_type == "reflect"
        assert "?" in response.response_text
        assert len(response.citations) > 0
    
    def test_reflect_response_with_plan(self, response_generator, sample_chunks):
        """Test reflection with plan context."""
        context = ResponseContext(
            concept="Thermal Conductivity",
            level="advanced",
            chunks=sample_chunks,
            plan_thinking="Student needs to apply theory to practice",
            plan_rationale="Check deep understanding",
        )
        
        response = response_generator.generate_reflect_response(context)
        
        assert response.action_type == "reflect"
        assert response.response_text != ""


# ===== Review Response Tests =====

class TestReviewResponse:
    """Test review/summary response generation."""
    
    def test_review_response_deterministic(self, response_generator, response_context):
        """Test deterministic review summary."""
        response = response_generator.generate_review_response(response_context)
        
        assert response.action_type == "review"
        assert "review" in response.response_text.lower()
        assert len(response.citations) > 0
        # Check no OCR artifacts
        assert "(cid:" not in response.response_text
        for citation in response.citations:
            assert "(cid:" not in citation.snippet_text
    
    def test_review_response_no_chunks(self, response_generator):
        """Test review with no chunks."""
        context = ResponseContext(
            concept="Heat Transfer",
            level="beginner",
            chunks=[],
        )
        
        response = response_generator.generate_review_response(context)
        
        assert response.action_type == "review"
        assert "review" in response.response_text.lower()


# ===== Prerequisite Review Tests =====

class TestPrerequisiteReview:
    """Test prerequisite review response generation."""
    
    @patch('backend.agents.tutor.response_generator.call_json_chat')
    def test_prerequisite_review_response(self, mock_call_json, response_generator, sample_chunks):
        """Test prerequisite review."""
        mock_call_json.return_value = {
            "response": "Let's first review thermal conductivity before moving to convection.",
            "confidence": 0.8,
        }
        
        response = response_generator.generate_prerequisite_review_response(
            target_concept="Convection",
            missing_prereqs=["Thermal Conductivity", "Temperature Gradient"],
            chunks=sample_chunks,
        )
        
        assert response.action_type == "explain"
        assert "Thermal Conductivity" in response.response_text or "review" in response.response_text.lower()
        assert len(response.citations) > 0
    
    def test_prerequisite_review_no_prereqs(self, response_generator):
        """Test when no prerequisites are missing."""
        response = response_generator.generate_prerequisite_review_response(
            target_concept="Convection",
            missing_prereqs=[],
            chunks=[],
        )
        
        assert response.action_type == "explain"
        assert "ready" in response.response_text.lower()


# ===== Orientation Response Tests =====

class TestOrientationResponse:
    """Test session orientation response generation."""
    
    @patch('backend.agents.tutor.response_generator.call_json_chat')
    def test_orientation_response(self, mock_call_json, response_generator, sample_chunks):
        """Test orientation response."""
        mock_call_json.return_value = {
            "response": "Let's start with the fundamentals of heat transfer.",
            "recommended_concept": "Heat Transfer Basics",
            "confidence": 0.85,
        }
        
        response = response_generator.generate_orientation_response(
            message="Hi, I need help with thermodynamics",
            focus_concept="Heat Transfer",
            learning_targets=["Heat Conduction", "Convection", "Radiation"],
            learning_path=["Basics", "Conduction", "Convection"],
            mastery_map={"Heat Transfer": {"mastery": 0.5}},
            chunks=sample_chunks,
        )
        
        assert response.action_type == "orient"
        assert "Heat Transfer" in response.response_text or "fundamentals" in response.response_text.lower()
        assert response.inference_concept == "Heat Transfer Basics"
        assert len(response.citations) > 0
    
    def test_orientation_response_no_recommendation(self, response_generator):
        """Test orientation with no learning path."""
        response = response_generator.generate_orientation_response(
            message="Help!",
            focus_concept=None,
            learning_targets=[],
            learning_path=[],
            mastery_map={},
            chunks=[],
        )
        
        assert response.action_type == "orient"
        assert response.response_text != ""


# ===== Cold Start Question Tests =====

class TestColdStartQuestion:
    """Test cold-start assessment question generation."""
    
    @patch('backend.agents.tutor.response_generator.call_json_chat')
    def test_cold_start_question(self, mock_call_json, response_generator, sample_chunks):
        """Test cold-start question."""
        mock_call_json.return_value = {
            "question": "What is the fundamental principle behind heat conduction?",
            "confidence": 0.7,
        }
        
        response = response_generator.generate_cold_start_question(
            concept="Heat Conduction",
            chunks=sample_chunks,
            message="I'm ready to start",
        )
        
        assert response.action_type == "ask"
        assert "?" in response.response_text
        assert len(response.citations) > 0
    
    def test_cold_start_question_no_concept(self, response_generator):
        """Test cold-start question without specific concept."""
        response = response_generator.generate_cold_start_question(
            concept=None,
            chunks=[],
        )
        
        assert response.action_type == "ask"
        assert "?" in response.response_text


# ===== Integration Tests =====

class TestResponseGeneratorIntegration:
    """Integration tests for ResponseGenerator."""
    
    def test_multiple_responses_all_clean(self, response_generator, response_context):
        """Test that all response types produce clean text."""
        with patch('backend.agents.tutor.response_generator.call_json_chat') as mock_call:
            mock_call.return_value = {"response": "Clean response", "confidence": 0.8}
            
            responses = [
                response_generator.generate_explain_response(response_context),
                response_generator.generate_ask_response(response_context),
                response_generator.generate_hint_response(response_context),
                response_generator.generate_reflect_response(response_context),
                response_generator.generate_review_response(response_context),
            ]
        
        for response in responses:
            assert "(cid:" not in response.response_text
            for citation in response.citations:
                assert "(cid:" not in citation.snippet_text
    
    def test_citation_consistency(self, response_generator, response_context):
        """Test that citations are consistent across response types."""
        with patch('backend.agents.tutor.response_generator.call_json_chat') as mock_call:
            mock_call.return_value = {"response": "Response", "confidence": 0.8}
            
            responses = [
                response_generator.generate_explain_response(response_context),
                response_generator.generate_ask_response(response_context),
            ]
        
        # Both should have same citations
        assert len(responses[0].citations) == len(responses[1].citations)
        assert responses[0].citations[0].chunk_id == responses[1].citations[0].chunk_id
    
    def test_source_ids_consistency(self, response_generator, response_context):
        """Test that source IDs match citations."""
        with patch('backend.agents.tutor.response_generator.call_json_chat') as mock_call:
            mock_call.return_value = {"response": "Response", "confidence": 0.8}
            
            response = response_generator.generate_explain_response(response_context)
        
        citation_ids = {c.chunk_id for c in response.citations}
        source_ids = set(response.source_chunk_ids)
        assert citation_ids == source_ids


# ===== TutorResponse Dataclass Tests =====

class TestTutorResponseDataclass:
    """Test TutorResponse dataclass structure."""
    
    def test_tutor_response_creation(self, sample_chunks):
        """Test creating TutorResponse instances."""
        citations = [
            Citation(
                chunk_id="c1",
                snippet_text="Clean snippet",
                pedagogy_role="definition",
                relevance_score=0.9,
            )
        ]
        
        response = TutorResponse(
            response_text="Sample response",
            action_type="explain",
            confidence=0.85,
            citations=citations,
            grounding_mode="llm_integrated",
            inference_concept="Test Concept",
            source_chunk_ids=["c1"],
        )
        
        assert response.response_text == "Sample response"
        assert response.action_type == "explain"
        assert len(response.citations) == 1
        assert response.confidence == 0.85
    
    def test_tutor_response_defaults(self):
        """Test TutorResponse default values."""
        response = TutorResponse(
            response_text="Text",
            action_type="ask",
            confidence=0.7,
        )
        
        assert response.citations == []
        assert response.grounding_mode == "llm_integrated"
        assert response.inference_concept is None
        assert response.source_chunk_ids == []


# ===== Edge Cases =====

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_large_chunk_cleanup(self, response_generator):
        """Test handling of large chunks."""
        large_chunk = {
            "id": "large",
            "snippet": "x" * 5000 + " (cid:4) more text",
        }
        
        citations = response_generator._build_citations([large_chunk])
        
        assert len(citations) == 1
        assert len(citations[0].snippet_text) < 5000
        assert "(cid:" not in citations[0].snippet_text
    
    def test_special_characters_preserved(self, response_generator):
        """Test that valid special characters are preserved."""
        chunks = [
            {
                "id": "special",
                "snippet": "Formula: q = -k∇T | Symbols: α β γ",
            }
        ]
        
        citations = response_generator._build_citations(chunks)
        
        assert "∇" in citations[0].snippet_text
        assert "α" in citations[0].snippet_text
    
    def test_unicode_handling(self, response_generator):
        """Test unicode character handling."""
        chunks = [
            {
                "id": "unicode",
                "snippet": "Energy: E = 能量 (energy in Chinese)",
            }
        ]
        
        citations = response_generator._build_citations(chunks)
        
        assert "能量" in citations[0].snippet_text
    
    def test_empty_response_text_fallback(self, response_generator, response_context):
        """Test fallback when LLM returns empty response."""
        with patch('backend.agents.tutor.response_generator.call_json_chat') as mock_call:
            mock_call.return_value = {
                "response": "",
                "confidence": 0.0,
            }
            
            response = response_generator.generate_explain_response(response_context)
        
        assert response.response_text != ""
        assert response.confidence > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

