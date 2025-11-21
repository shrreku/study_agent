"""Database utilities: connection factory and schema ensure.

Exposes:
- get_db_conn(): psycopg2 connection (registers pgvector adapter if available)
- ensure_schema(): creates required tables and extensions (idempotent)
"""
from __future__ import annotations
import os
import logging
import psycopg2


def get_db_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        host = os.getenv("POSTGRES_HOST", "postgres")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "app")
        dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    conn = psycopg2.connect(dsn)
    try:
        # Optional: register pgvector adapter if available
        from pgvector.psycopg2 import register_vector  # type: ignore
        register_vector(conn)
    except Exception:
        pass
    return conn


def ensure_schema() -> None:
    ddls = [
        """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
""",
        """
CREATE EXTENSION IF NOT EXISTS vector;
""",
        """
CREATE TABLE IF NOT EXISTS app_user (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email TEXT UNIQUE,
  password_hash TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
""",
        """
CREATE TABLE IF NOT EXISTS llm_model (
  id TEXT PRIMARY KEY,
  provider TEXT,
  model_name TEXT NOT NULL,
  display_name TEXT,
  cost_hint NUMERIC,
  latency_hint NUMERIC,
  enabled BOOLEAN NOT NULL DEFAULT TRUE
);
""",
        """
ALTER TABLE app_user
  ADD COLUMN IF NOT EXISTS display_name TEXT,
  ADD COLUMN IF NOT EXISTS role TEXT,
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();
""",
        """
CREATE TABLE IF NOT EXISTS resource (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES app_user(id) NULL,
  title TEXT,
  filename TEXT,
  content_type TEXT,
  size_bytes BIGINT,
  storage_path TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
""",
        """
ALTER TABLE resource
  ADD COLUMN IF NOT EXISTS status TEXT,
  ADD COLUMN IF NOT EXISTS error_message TEXT,
  ADD COLUMN IF NOT EXISTS metadata JSONB;
""",
        """
CREATE TABLE IF NOT EXISTS extracted_page (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  resource_id UUID REFERENCES resource(id) ON DELETE CASCADE,
  page_number INT,
  raw_text TEXT,
  ocr_confidence NUMERIC NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
""",
        """
CREATE TABLE IF NOT EXISTS job (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NULL,
  resource_id UUID REFERENCES resource(id) NULL,
  type TEXT,
  status TEXT,
  payload JSONB,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);
""",
        """
CREATE TABLE IF NOT EXISTS chunk (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  resource_id UUID REFERENCES resource(id) ON DELETE CASCADE,
  page_number INT,
  source_offset INT,
  chunk_type TEXT,
  concepts TEXT[],
  text_snippet TEXT,
  full_text TEXT,
  math_expressions TEXT[],
  figure_ids UUID[],
  difficulty SMALLINT,
  embedding vector(384),
  embedding_version TEXT,
  search_tsv tsvector,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);
""",
        """
ALTER TABLE chunk
  ADD COLUMN IF NOT EXISTS section_title TEXT,
  ADD COLUMN IF NOT EXISTS section_number TEXT,
  ADD COLUMN IF NOT EXISTS section_path TEXT[],
  ADD COLUMN IF NOT EXISTS section_level INT,
  ADD COLUMN IF NOT EXISTS page_start INT,
  ADD COLUMN IF NOT EXISTS page_end INT,
  ADD COLUMN IF NOT EXISTS token_count INT,
  ADD COLUMN IF NOT EXISTS has_figure BOOLEAN,
  ADD COLUMN IF NOT EXISTS has_equation BOOLEAN,
  ADD COLUMN IF NOT EXISTS figure_labels TEXT[],
  ADD COLUMN IF NOT EXISTS equation_labels TEXT[],
  ADD COLUMN IF NOT EXISTS caption TEXT,
  ADD COLUMN IF NOT EXISTS tags JSONB,
  ADD COLUMN IF NOT EXISTS heading_tsv TSVECTOR,
  ADD COLUMN IF NOT EXISTS body_tsv TSVECTOR,
  ADD COLUMN IF NOT EXISTS tagging_model TEXT,
  ADD COLUMN IF NOT EXISTS tagging_version INT;
""",
        """
CREATE INDEX IF NOT EXISTS idx_chunk_search_tsv ON chunk USING GIN (search_tsv);
""",
        """
CREATE INDEX IF NOT EXISTS idx_chunk_heading_tsv ON chunk USING GIN (heading_tsv);
""",
        """
CREATE INDEX IF NOT EXISTS idx_chunk_tags ON chunk USING GIN (tags);
""",
        """
CREATE TABLE IF NOT EXISTS user_concept_mastery (
  user_id UUID NOT NULL,
  concept TEXT NOT NULL,
  mastery NUMERIC DEFAULT 0.0,
  last_seen TIMESTAMPTZ,
  attempts INT DEFAULT 0,
  correct INT DEFAULT 0,
  PRIMARY KEY (user_id, concept)
);
""",
        """
CREATE TABLE IF NOT EXISTS user_doubt (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL,
  question TEXT NOT NULL,
  asked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  concepts TEXT[] NOT NULL DEFAULT '{}',
  weak_signal BOOLEAN NOT NULL DEFAULT FALSE
);
""",
        """
CREATE TABLE IF NOT EXISTS user_resource_concept_feedback (
  user_id UUID NOT NULL,
  resource_id UUID NOT NULL,
  concept TEXT NOT NULL,
  feedback_type TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, resource_id, concept, feedback_type)
);
""",
        """
CREATE TABLE IF NOT EXISTS tutor_session (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
  resource_id UUID REFERENCES resource(id) ON DELETE SET NULL,
  target_concepts TEXT[] NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'active',
  policy JSONB,
  last_concept TEXT,
  last_action TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
""",
        """
ALTER TABLE tutor_session
  ADD COLUMN IF NOT EXISTS mode TEXT,
  ADD COLUMN IF NOT EXISTS agent_action_mode TEXT,
  ADD COLUMN IF NOT EXISTS resource_ids UUID[],
  ADD COLUMN IF NOT EXISTS config JSONB,
  ADD COLUMN IF NOT EXISTS model_strategy_type TEXT,
  ADD COLUMN IF NOT EXISTS model_ids TEXT[];
""",
        """
CREATE INDEX IF NOT EXISTS idx_tutor_session_user_created ON tutor_session (user_id, created_at DESC);
""",
        """
CREATE TABLE IF NOT EXISTS tutor_turn (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id UUID NOT NULL REFERENCES tutor_session(id) ON DELETE CASCADE,
  turn_index INT NOT NULL,
  user_text TEXT,
  intent TEXT,
  affect TEXT,
  concept TEXT,
  action_type TEXT,
  response_text TEXT,
  source_chunk_ids UUID[] NOT NULL DEFAULT '{}',
  confidence NUMERIC,
  mastery_delta NUMERIC,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
""",
        """
ALTER TABLE tutor_turn
  ADD COLUMN IF NOT EXISTS proposed_action TEXT,
  ADD COLUMN IF NOT EXISTS user_choice TEXT,
  ADD COLUMN IF NOT EXISTS model_id TEXT,
  ADD COLUMN IF NOT EXISTS model_name TEXT,
  ADD COLUMN IF NOT EXISTS tool_calls JSONB,
  ADD COLUMN IF NOT EXISTS retrieval_metadata JSONB,
  ADD COLUMN IF NOT EXISTS policy_trace JSONB,
  ADD COLUMN IF NOT EXISTS candidates JSONB,
  ADD COLUMN IF NOT EXISTS preference_rating JSONB;
""",
        """
CREATE UNIQUE INDEX IF NOT EXISTS idx_tutor_turn_session_turn_index ON tutor_turn (session_id, turn_index);
""",
        """
CREATE INDEX IF NOT EXISTS idx_tutor_turn_created ON tutor_turn (created_at);
""",
        """
CREATE TABLE IF NOT EXISTS tutor_event (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id UUID NOT NULL REFERENCES tutor_session(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,
  payload JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
""",
        """
CREATE INDEX IF NOT EXISTS idx_tutor_event_session_created ON tutor_event (session_id, created_at);
""",
        """
CREATE TABLE IF NOT EXISTS tutor_annotation (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  turn_id UUID NOT NULL REFERENCES tutor_turn(id) ON DELETE CASCADE,
  labeler_id UUID REFERENCES app_user(id) ON DELETE SET NULL,
  correctness SMALLINT,
  helpfulness SMALLINT,
  coverage SMALLINT,
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
""",
    ]
    conn = None
    try:
        conn = get_db_conn()
        test_user_id = os.getenv("TEST_USER_ID")
        test_user_email = os.getenv("TEST_USER_EMAIL", "mock@example.com")
        with conn.cursor() as cur:
            for ddl in ddls:
                cur.execute(ddl)
            if test_user_id:
                try:
                    cur.execute(
                        """
                        INSERT INTO app_user (id, email)
                        VALUES (%s::uuid, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (test_user_id, test_user_email),
                    )
                    cur.execute(
                        """
                        INSERT INTO user_concept_mastery (user_id, concept, mastery, last_seen, attempts, correct)
                        VALUES (%s::uuid, %s, %s, now(), %s, %s)
                        ON CONFLICT (user_id, concept) DO NOTHING
                        """,
                        (test_user_id, "Heat Transfer Fundamentals", 0.45, 4, 1),
                    )
                    cur.execute(
                        """
                        INSERT INTO user_concept_mastery (user_id, concept, mastery, last_seen, attempts, correct)
                        VALUES (%s::uuid, %s, %s, now(), %s, %s)
                        ON CONFLICT (user_id, concept) DO NOTHING
                        """,
                        (test_user_id, "Thermodynamics Basics", 0.72, 6, 5),
                    )
                except Exception:
                    logging.exception("ensure_schema_seed_user_failed")
        conn.commit()
    except Exception as e:
        logging.exception("Failed to ensure DB schema: %s", e)
    finally:
        if conn:
            conn.close()
