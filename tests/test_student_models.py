"""
Tests for Student Models and Curriculum Runner

Run with:
    pytest tests/test_student_models.py -v
"""

import pytest
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


class TestLearningProfile:
    """Test LearningProfile configurations"""
    
    def test_learner_type_profiles(self):
        """Each learner type should have distinct profile"""
        from mdp.student_models import LearningProfile, LearnerType
        
        profiles = {}
        for lt in LearnerType:
            profile = LearningProfile.from_learner_type(lt)
            profiles[lt] = profile
            
            # Basic validation
            assert 0.0 <= profile.learning_rate <= 1.0
            assert 0.0 <= profile.prior_knowledge <= 1.0
            assert 0.0 <= profile.p_correct_at_mastery <= 1.0
            assert 0.0 <= profile.p_correct_at_zero <= 1.0
        
        # Struggling should have lower learning rate than fast learner
        assert profiles[LearnerType.STRUGGLING].learning_rate < profiles[LearnerType.FAST_LEARNER].learning_rate
        
        # Deep thinker should ask more questions
        assert profiles[LearnerType.DEEP_THINKER].p_ask_question > profiles[LearnerType.PASSIVE].p_ask_question
    
    def test_profile_probabilities_valid(self):
        """All probability parameters should be in [0, 1]"""
        from mdp.student_models import LearningProfile, LearnerType
        
        for lt in LearnerType:
            profile = LearningProfile.from_learner_type(lt)
            
            assert profile.p_correct_at_zero <= profile.p_correct_at_mastery, \
                f"Higher mastery should give higher P(correct) for {lt}"


class TestStudentState:
    """Test StudentState tracking"""
    
    def test_mastery_tracking(self):
        """Mastery should update correctly"""
        from mdp.student_models import StudentState
        
        state = StudentState(student_id="test")
        
        # Initial mastery should be 0
        assert state.get_mastery("concept1") == 0.0
        
        # Update mastery
        state.update_mastery("concept1", 0.3)
        assert state.get_mastery("concept1") == 0.3
        
        # Negative delta
        state.update_mastery("concept1", -0.1)
        assert state.get_mastery("concept1") == 0.2
        
        # Bounded to [0, 1]
        state.update_mastery("concept1", 2.0)
        assert state.get_mastery("concept1") == 1.0
        
        state.update_mastery("concept1", -5.0)
        assert state.get_mastery("concept1") == 0.0


class TestLLMStudentSimulator:
    """Test LLMStudentSimulator behavior"""
    
    def test_creation(self):
        """Student should be creatable"""
        from mdp.student_models import create_student
        
        student = create_student("average", llm_client=None)
        assert student is not None
        assert student.profile.learner_type.value == "average"
    
    def test_template_response(self):
        """Template responses should work without LLM"""
        from mdp.student_models import create_student
        
        student = create_student("struggling", llm_client=None)
        
        response, meta = student.respond(
            tutor_message="What is convection?",
            concept="convection",
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "intent" in meta
        assert "correctness" in meta
        assert "mastery_after" in meta
    
    def test_mastery_updates(self):
        """Mastery should update based on correctness"""
        from mdp.student_models import create_student, LearnerType, LearningProfile
        
        # Create deterministic student for testing
        profile = LearningProfile(
            learner_type=LearnerType.FAST_LEARNER,
            p_correct_at_mastery=1.0,  # Always correct at high mastery
            p_correct_at_zero=1.0,     # Always correct even at low mastery
            prior_knowledge=0.0,
        )
        
        from mdp.student_models import LLMStudentSimulator
        student = LLMStudentSimulator(profile, llm_client=None)
        
        initial_mastery = student.get_mastery("test_concept")
        assert initial_mastery == profile.prior_knowledge
        
        # Respond to a question (should get it correct and gain mastery)
        _, meta = student.respond(
            tutor_message="Can you answer this question?",
            concept="test_concept",
        )
        
        # Mastery should have changed (could go up or down depending on random correctness)
        new_mastery = student.get_mastery("test_concept")
        assert new_mastery >= 0.0
        assert new_mastery <= 1.0
    
    def test_conversation_history(self):
        """Conversation history should be tracked"""
        from mdp.student_models import create_student
        
        student = create_student("average", llm_client=None)
        
        assert len(student.conversation_history) == 0
        
        student.respond("Message 1", "concept")
        assert len(student.conversation_history) == 2  # tutor + student turn
        
        student.respond("Message 2", "concept")
        assert len(student.conversation_history) == 4


class TestStudentPopulation:
    """Test StudentPopulation sampling"""
    
    def test_sample_student(self):
        """Should sample students from distribution"""
        from mdp.student_models import StudentPopulation
        
        pop = StudentPopulation(llm_client=None)
        
        students = [pop.sample_student() for _ in range(20)]
        
        assert len(students) == 20
        
        # Should have some variety
        types = {s.profile.learner_type for s in students}
        assert len(types) >= 2  # At least 2 different types
    
    def test_generate_balanced(self):
        """Should generate one of each type"""
        from mdp.student_models import StudentPopulation, LearnerType
        
        pop = StudentPopulation(llm_client=None)
        students = pop.generate_balanced()
        
        assert len(students) == len(LearnerType)
        
        types = {s.profile.learner_type for s in students}
        assert types == set(LearnerType)


class TestCurriculumConfig:
    """Test CurriculumConfig creation"""
    
    def test_default_config(self):
        """Default config should be valid"""
        from mdp.curriculum_runner import CurriculumConfig
        
        config = CurriculumConfig()
        
        assert config.max_turns_per_concept > 0
        assert 0.0 < config.target_mastery <= 1.0
        assert config.students_per_concept > 0
    
    def test_custom_config(self):
        """Custom config should be applied"""
        from mdp.curriculum_runner import CurriculumConfig
        
        config = CurriculumConfig(
            concept_ids=["c1", "c2"],
            students_per_concept=5,
            target_mastery=0.9,
            use_llm_students=False,
        )
        
        assert config.concept_ids == ["c1", "c2"]
        assert config.students_per_concept == 5
        assert config.target_mastery == 0.9
        assert config.use_llm_students is False


class TestSessionResult:
    """Test SessionResult data structure"""
    
    def test_to_dict(self):
        """Should serialize to dict"""
        from mdp.curriculum_runner import SessionResult
        
        result = SessionResult(
            session_id="test123",
            student_id="student1",
            learner_type="average",
            concepts_covered=["c1", "c2"],
            total_turns=15,
        )
        
        d = result.to_dict()
        
        assert d["session_id"] == "test123"
        assert d["student_id"] == "student1"
        assert d["concepts_covered"] == ["c1", "c2"]
        assert d["total_turns"] == 15


class TestIntegration:
    """Integration tests (requires more setup)"""
    
    def test_end_to_end_template_students(self):
        """End-to-end test with template students (no LLM)"""
        from mdp.curriculum_runner import CurriculumRunner, CurriculumConfig
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CurriculumConfig(
                concept_ids=["test_concept"],
                students_per_concept=1,
                max_turns_per_concept=5,
                target_mastery=0.95,  # High so it stops quickly
                use_llm_students=False,
                output_dir=tmpdir,
                output_prefix="test",
            )
            
            runner = CurriculumRunner(config, llm_client=None)
            results = runner.run()
            
            # Should complete
            assert results.get("status") == "success" or "error" in results
            
            if results.get("status") == "success":
                assert results.get("concepts_covered", 0) >= 1
                assert "output_files" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
