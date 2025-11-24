from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from psycopg2.extras import Json

from .context_model import TutorContext
from .state import TutorSessionPolicy

logger = logging.getLogger(__name__)


class TutorStateManager:
    """
    Manages persistence of the Tutor Agent's state.
    Adapts the internal MDP state to the existing database schema.
    """

    def __init__(self, cur: Any):
        self.cur = cur

    def load_context(self, session_id: str, user_id: str) -> TutorContext:
        """
        Load the TutorContext from the database.
        """
        logger.info(f"Loading context for session {session_id}")

        # 1. Fetch session data (policy, target_concepts)
        self.cur.execute(
            """
            SELECT policy, target_concepts
            FROM tutor_session
            WHERE id = %s::uuid
            LIMIT 1
            """,
            (session_id,),
        )
        row = self.cur.fetchone()

        if not row:
            # Should not happen if session exists, but handle gracefully
            return TutorContext(
                session_id=session_id,
                user_id=user_id,
                policy_state=TutorSessionPolicy(),
                mastery_map={},
            )

        policy_data = row[0] or {}
        target_concepts = row[1] or []

        # 2. Fetch mastery map
        self.cur.execute(
            """
            SELECT concept, mastery
            FROM user_concept_mastery
            WHERE user_id = %s::uuid
            """,
            (user_id,),
        )
        mastery_rows = self.cur.fetchall() or []
        mastery_map = {
            row[0]: {"mastery": float(row[1])}
            for row in mastery_rows
            if row[0]
        }

        # 3. Reconstruct TutorSessionPolicy
        # Primary source: mdp_state (new environment orchestrator path)
        mdp_state = policy_data.get("mdp_state", {}) or {}

        # Backwards-compat: adapt legacy policy.session_plan from older
        # step-engine sessions (including /api/test/init-tutor-session)
        # into an mdp_state shape that SessionState builder understands.
        if not mdp_state:
            legacy_session_plan = policy_data.get("session_plan")
            if isinstance(legacy_session_plan, dict):
                try:
                    legacy_strategy = (
                        str(legacy_session_plan.get("strategy") or "")
                        .strip()
                        or "learning_path"
                    )
                except Exception:
                    legacy_strategy = "learning_path"

                # Prefer explicit concept_plan, fall back to legacy "concepts"
                raw_concept_plan = legacy_session_plan.get("concept_plan")
                if not isinstance(raw_concept_plan, list):
                    raw_concept_plan = legacy_session_plan.get("concepts") or []

                adapted_session_plan: Dict[str, Any] = {
                    "strategy": legacy_strategy,
                    "concept_plan": raw_concept_plan,
                }
                mdp_state = {
                    "session_plan": adapted_session_plan,
                    "session_plan_index": int(
                        legacy_session_plan.get("current_index") or 0
                    ),
                    "session_strategy": legacy_strategy,
                }

        policy_state = TutorSessionPolicy(
            session_plan=mdp_state.get("session_plan", {}),
            session_plan_index=mdp_state.get("session_plan_index", 0),
            session_strategy=mdp_state.get("session_strategy", "learning_path"),
            concept_episode_id=mdp_state.get("concept_episode_id"),
            concept_episode_mastery_start=mdp_state.get(
                "concept_episode_mastery_start"
            ),
            srl_plan_step_index=mdp_state.get("srl_plan_step_index", 0),
            # NEW: Load concept plan
            concept_plan=mdp_state.get("concept_plan", {}),
            quiz_phase=mdp_state.get("quiz_phase", ""),
            quiz_question_index=mdp_state.get("quiz_question_index", 0),
            quiz_max_questions=mdp_state.get("quiz_max_questions", 0),
            concept_episode_quiz_correct=mdp_state.get(
                "concept_episode_quiz_correct", 0
            ),
            concept_episode_quiz_wrong=mdp_state.get(
                "concept_episode_quiz_wrong", 0
            ),
            concept_episode_step_count=mdp_state.get(
                "concept_episode_step_count", 0
            ),
            concept_episode_last_control_type=mdp_state.get(
                "concept_episode_last_control_type"
            ),
        )

        # 4. Build Context
        context = TutorContext(
            session_id=session_id,
            user_id=user_id,
            policy_state=policy_state,
            mastery_map=mastery_map,
            # We can infer focus concept from policy state or session plan
            # For now, let's leave it to the orchestrator to derive from state
        )

        return context

    def save_context(self, context: TutorContext) -> None:
        """
        Save the TutorContext (specifically the policy state) to the database.
        """
        logger.info(f"Saving context for session {context.session_id}")
        policy_state = context.policy_state
        
        # 1. Serialize Policy State
        mdp_state = {
            "session_plan": policy_state.session_plan,
            "session_plan_index": policy_state.session_plan_index,
            "session_strategy": policy_state.session_strategy,
            "concept_episode_id": policy_state.concept_episode_id,
            "concept_episode_mastery_start": policy_state.concept_episode_mastery_start,
            "srl_plan_step_index": policy_state.srl_plan_step_index,
            # NEW: Save concept plan
            "concept_plan": policy_state.concept_plan,
            
            "quiz_phase": policy_state.quiz_phase,
            "quiz_question_index": policy_state.quiz_question_index,
            "quiz_max_questions": policy_state.quiz_max_questions,
            "concept_episode_quiz_correct": policy_state.concept_episode_quiz_correct,
            "concept_episode_quiz_wrong": policy_state.concept_episode_quiz_wrong,
            "concept_episode_step_count": policy_state.concept_episode_step_count,
            "concept_episode_last_control_type": policy_state.concept_episode_last_control_type,
        }

        # 2. Fetch existing policy to merge (avoid overwriting other fields)
        self.cur.execute(
            """
            SELECT policy
            FROM tutor_session
            WHERE id = %s::uuid
            LIMIT 1
            """,
            (context.session_id,),
        )
        row = self.cur.fetchone()
        current_policy_data = (row[0] or {}) if row else {}
        
        # 3. Update mdp_state
        current_policy_data["mdp_state"] = mdp_state
        
        # 4. Save to DB
        self.cur.execute(
            """
            UPDATE tutor_session
            SET policy = %s,
            updated_at = now()
            WHERE id = %s::uuid
            """,
            (Json(current_policy_data), context.session_id),
        )
        
        # Note: Mastery updates are typically handled by the quiz endpoint or specific
        # mastery update logic. If the orchestrator updates mastery in memory,
        # we should persist it here too.
        # For now, we assume mastery updates happen via `user_concept_mastery` table
        # which might be updated by the orchestrator's tools or separate logic.


def next_turn_index(cur: Any, session_id: str) -> int:
    """
    Get the next turn index for a session.
    
    Args:
        cur: Database cursor
        session_id: Session ID
        
    Returns:
        Next turn index (0-based)
    """
    cur.execute(
        """
        SELECT COUNT(*)
        FROM tutor_turn
        WHERE session_id = %s::uuid
        """,
        (session_id,)
    )
    row = cur.fetchone()
    count = row[0] if row else 0
    return int(count)
