from typing import List, Dict, Any
import uuid
import time
import logging
from collections import deque, defaultdict
from mdp.schemas import SessionPlan, SessionStep, StudentProfile
from mdp.rag import RAGTools

logger = logging.getLogger(__name__)

class SessionPlanner:
    """
    Responsible for generating a sequence of concepts (Session Plan)
    based on selected resources, respecting prerequisites and chronology.
    """
    def __init__(self):
        self.rag = RAGTools()

    def generate_plan(self, student_profile: StudentProfile, resource_ids: List[str] = None, concept_ids: List[str] = None) -> SessionPlan:
        """
        Generates a session plan from the given resources and/or specific concepts.

        Behaviour:
        - If only resource_ids are provided: use all concepts that appear in those
          resources (via the KG) and order them by prerequisites + chronology.
        - If only concept_ids are provided: use those concepts (looked up in the KG)
          and order them by prerequisites (chronology may be undefined).
        - If both are provided: restrict to concepts that appear in the resources AND
          are present in concept_ids. If the intersection is empty, fall back to using
          concept_ids only.
        """
        # 1. Fetch Concepts
        concepts: List[Dict[str, Any]] = []
        resource_ids = resource_ids or []
        explicit_concepts = concept_ids or []

        if resource_ids:
            logger.info(f"Generating session plan for resources: {resource_ids}")
            concepts = self.rag.get_concepts_for_resources(resource_ids)
            if explicit_concepts:
                allowed = set(explicit_concepts)
                concepts = [c for c in concepts if c.get("id") in allowed]
                if not concepts:
                    logger.info(
                        "No overlapping concepts for resources/explicit list; "
                        "falling back to explicit concepts only",
                    )
                    concepts = self.rag.get_concept_details(explicit_concepts)
        elif explicit_concepts:
            logger.info(f"Generating session plan for concepts: {explicit_concepts}")
            concepts = self.rag.get_concept_details(explicit_concepts)

        if not concepts:
            logger.warning("No concepts found for inputs.")
            return SessionPlan(
                session_id=str(uuid.uuid4()),
                steps=[],
                resource_ids=resource_ids,
                created_at=time.time(),
            )
        
        # Create a map for easy access and deduplication
        # concept_map: id -> {id, name, first_chunk_id}
        concept_map = {c['id']: c for c in concepts}
        concept_ids = list(concept_map.keys())
        logger.info(f"Found {len(concept_ids)} unique concepts.")

        # 2. Fetch Dependencies between these concepts
        deps = self.rag.get_concept_dependencies(concept_ids)
        logger.info(f"Found {len(deps)} dependency edges.")
        
        # Build Adjacency List and In-Degree map
        adj = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Initialize in-degree for all concepts
        for c in concept_ids:
            in_degree[c] = 0
            
        for d in deps:
            u, v = d['from'], d['to']
            # Only consider dependencies within the candidate set
            if u in concept_map and v in concept_map:
                adj[u].append(v)
                in_degree[v] += 1
            
        # 3. Topological Sort with Chronology Tie-Breaking
        # Queue contains nodes with in_degree = 0 (available to be studied)
        queue = [c for c in concept_ids if in_degree[c] == 0]
        
        # Sort queue initially by first_chunk_id (Chronology)
        # Concepts appearing earlier in the document should come first if no prereqs block them
        queue.sort(key=lambda x: str(concept_map[x].get('first_chunk_id', '')))
        
        sorted_concepts = []
        
        while queue:
            # Always pick the earliest available concept
            # The queue is already sorted by chronology when initialized
            # We pop the first one
            u = queue.pop(0)
            sorted_concepts.append(u)
            
            # Process neighbors
            neighbors_freed = []
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    neighbors_freed.append(v)
            
            # Sort freed neighbors by chronology before adding to queue
            neighbors_freed.sort(key=lambda x: str(concept_map[x].get('first_chunk_id', '')))
            
            # Merge neighbors into queue. 
            # Note: Ideally we want to maintain the queue sorted by chronology.
            # Since queue is already sorted, and neighbors are sorted, we can merge or just append and re-sort.
            # Re-sorting is safest and simplest given N is small (<100 usually).
            queue.extend(neighbors_freed)
            queue.sort(key=lambda x: str(concept_map[x].get('first_chunk_id', '')))
        
        # Handle Cycles or Disconnected Components that weren't reached
        if len(sorted_concepts) < len(concept_ids):
            logger.warning("Cycle detected or unreachable nodes in dependency graph. Appending remaining concepts by chronology.")
            remaining = [c for c in concept_ids if c not in sorted_concepts]
            remaining.sort(key=lambda x: str(concept_map[x].get('first_chunk_id', '')))
            sorted_concepts.extend(remaining)
            
        # 4. Create Plan Steps
        final_steps = []
        step_counter = 1
        
        for cid in sorted_concepts:
            c_data = concept_map[cid]
            
            # Placeholder for Mastery Logic
            # if student_profile.is_mastered(cid): continue
            
            step = SessionStep(
                step_id=step_counter,
                concept_id=cid,
                concept_name=c_data.get('name', cid),
                reason="Prerequisite & Chronology",
                status="pending"
            )
            final_steps.append(step)
            step_counter += 1
            
        logger.info(f"Generated session plan with {len(final_steps)} steps.")
        
        return SessionPlan(
            session_id=str(uuid.uuid4()),
            steps=final_steps,
            resource_ids=resource_ids,
            created_at=time.time()
        )
