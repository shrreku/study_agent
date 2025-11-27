import logging
import time
import os
from typing import Optional, Dict, List, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
from backend.core.db import get_db_conn

logger = logging.getLogger(__name__)

class MasteryModel:
    def __init__(self):
        self.alpha = 0.3  # Learning rate for EMA
        self.logger = logger

    def get_mastery(self, user_id: str, concepts: List[str]) -> Dict[str, float]:
        """
        Retrieve current mastery levels for a list of concepts for a user.
        Returns a dictionary {concept: mastery_score}.
        Defaults to 0.0 if no record exists.
        """
        if not user_id or not concepts:
            return {}

        try:
            conn = get_db_conn()
            mastery_map = {c: 0.0 for c in concepts}
            
            with conn.cursor() as cur:
                # Use ANY for array matching in SQL
                query = """
                    SELECT concept, mastery 
                    FROM user_concept_mastery 
                    WHERE user_id = %s AND concept = ANY(%s)
                """
                cur.execute(query, (user_id, concepts))
                rows = cur.fetchall()
                
                for row in rows:
                    # row is likely a tuple (concept, mastery) or RealDictRow depending on cursor factory
                    # But get_db_conn doesn't seem to set RealDictCursor by default unless specified
                    # Let's assume tuple for standard cursor
                    concept = row[0]
                    score = float(row[1]) if row[1] is not None else 0.0
                    mastery_map[concept] = score
                    
            conn.close()
            return mastery_map
        except Exception as e:
            self.logger.error(f"Failed to get mastery for user {user_id}: {e}")
            # Return default 0.0s on error to be safe
            return {c: 0.0 for c in concepts}

    def update_mastery(self, user_id: str, concept: str, correctness_score: float, hints_used: int = 0) -> float:
        """
        Update mastery for a single concept based on an interaction.
        
        Args:
            user_id: UUID of the user
            concept: Concept name/ID
            correctness_score: 0.0 to 1.0 (1.0 = fully correct)
            hints_used: Number of hints used (could dampen the score)
            
        Returns:
            The NEW mastery score.
        """
        if not user_id or not concept:
            return 0.0

        conn = None
        try:
            conn = get_db_conn()
            
            # 1. Get current state
            current_mastery = 0.0
            attempts = 0
            correct_count = 0
            
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT mastery, attempts, correct FROM user_concept_mastery WHERE user_id = %s AND concept = %s",
                    (user_id, concept)
                )
                row = cur.fetchone()
                if row:
                    current_mastery = float(row[0]) if row[0] is not None else 0.0
                    attempts = int(row[1]) if row[1] is not None else 0
                    correct_count = int(row[2]) if row[2] is not None else 0
            
            # 2. Calculate effective score
            # Heuristic: Reduce score if hints were used. 
            # e.g., if correct (1.0) but 1 hint, score -> 0.8. 2 hints -> 0.6.
            effective_score = max(0.0, correctness_score - (0.2 * hints_used))
            
            # 3. Update rule (EMA)
            # We can make alpha adaptive (e.g., higher for first few attempts)
            # For now, fixed alpha
            new_mastery = (1 - self.alpha) * current_mastery + (self.alpha * effective_score)
            
            # Update counters
            new_attempts = attempts + 1
            new_correct = correct_count + (1 if correctness_score > 0.8 else 0) # Threshold for "correct" stat
            
            self.logger.info(
                f"Updating mastery for user={user_id}, concept='{concept}': "
                f"old={current_mastery:.3f}, obs={effective_score:.3f} (raw={correctness_score}, hints={hints_used}), "
                f"new={new_mastery:.3f}"
            )
            
            # 4. Write back
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_concept_mastery (user_id, concept, mastery, last_seen, attempts, correct)
                    VALUES (%s, %s, %s, now(), %s, %s)
                    ON CONFLICT (user_id, concept) 
                    DO UPDATE SET
                        mastery = EXCLUDED.mastery,
                        last_seen = EXCLUDED.last_seen,
                        attempts = EXCLUDED.attempts,
                        correct = EXCLUDED.correct
                    """,
                    (user_id, concept, new_mastery, new_attempts, new_correct)
                )
            conn.commit()
            
            return new_mastery
            
        except Exception as e:
            self.logger.error(f"Failed to update mastery for {user_id}/{concept}: {e}")
            if conn:
                conn.rollback()
            return 0.0 # Fallback
        finally:
            if conn:
                conn.close()
