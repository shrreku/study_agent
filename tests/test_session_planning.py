import unittest
from unittest.mock import MagicMock
from mdp.session_planning import SessionPlanner
from mdp.schemas import StudentProfile

class TestSessionPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = SessionPlanner()
        self.planner.rag = MagicMock()
        self.profile = StudentProfile(student_id="test_user")

    def test_chronology_only(self):
        # A(1), B(2), C(3) - No Deps
        concepts = [
            {'id': 'A', 'name': 'Concept A', 'first_chunk_id': 'chunk_1'},
            {'id': 'B', 'name': 'Concept B', 'first_chunk_id': 'chunk_2'},
            {'id': 'C', 'name': 'Concept C', 'first_chunk_id': 'chunk_3'},
        ]
        deps = []
        
        self.planner.rag.get_concepts_for_resources.return_value = concepts
        self.planner.rag.get_concept_dependencies.return_value = deps

        plan = self.planner.generate_plan(self.profile, resource_ids=['res1'])
        
        self.assertEqual([s.concept_id for s in plan.steps], ['A', 'B', 'C'])

    def test_prerequisite_override(self):
        # B(1), A(2) - Chronologically B is first.
        # But A -> B (A is prereq of B).
        # Expected: A, B
        concepts = [
            {'id': 'B', 'name': 'Concept B', 'first_chunk_id': 'chunk_1'},
            {'id': 'A', 'name': 'Concept A', 'first_chunk_id': 'chunk_2'},
        ]
        deps = [{'from': 'A', 'to': 'B'}]
        
        self.planner.rag.get_concepts_for_resources.return_value = concepts
        self.planner.rag.get_concept_dependencies.return_value = deps

        plan = self.planner.generate_plan(self.profile, resource_ids=['res1'])
        
        self.assertEqual([s.concept_id for s in plan.steps], ['A', 'B'])

    def test_mixed_logic(self):
        # A(1), B(2), C(3). C -> B.
        # Chronologically: A, B, C.
        # Dependencies: C must come before B.
        # Execution:
        # In-degrees: A=0, B=1 (from C), C=0.
        # Queue: [A, C] (sorted by time).
        # 1. Pop A.
        # 2. Pop C. Frees B.
        # 3. Pop B.
        # Result: A, C, B.
        concepts = [
            {'id': 'A', 'name': 'Concept A', 'first_chunk_id': 'chunk_1'},
            {'id': 'B', 'name': 'Concept B', 'first_chunk_id': 'chunk_2'},
            {'id': 'C', 'name': 'Concept C', 'first_chunk_id': 'chunk_3'},
        ]
        deps = [{'from': 'C', 'to': 'B'}]
        
        self.planner.rag.get_concepts_for_resources.return_value = concepts
        self.planner.rag.get_concept_dependencies.return_value = deps
        
        plan = self.planner.generate_plan(self.profile, ['res1'])
        
        self.assertEqual([s.concept_id for s in plan.steps], ['A', 'C', 'B'])

if __name__ == '__main__':
    unittest.main()
