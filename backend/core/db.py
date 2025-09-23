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
    ]
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            for ddl in ddls:
                cur.execute(ddl)
        conn.commit()
    except Exception as e:
        logging.exception("Failed to ensure DB schema: %s", e)
    finally:
        if conn:
            conn.close()
