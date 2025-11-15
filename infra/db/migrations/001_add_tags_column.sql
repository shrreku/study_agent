-- Migration: Add tags JSONB column to chunk table for pedagogy role tagging
-- Date: 2025-11-08
-- Ticket: TUTOR-IMPROVE-05

-- Add tags JSONB column to chunk table
ALTER TABLE chunk ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '{}'::jsonb;

-- Create GIN index for efficient JSONB queries
CREATE INDEX IF NOT EXISTS idx_chunk_tags_gin ON chunk USING GIN(tags);

-- Create specialized index for pedagogy_role queries (most common)
CREATE INDEX IF NOT EXISTS idx_chunk_pedagogy_role 
  ON chunk ((tags->>'pedagogy_role')) 
  WHERE tags->>'pedagogy_role' IS NOT NULL;

-- Add comment for documentation
COMMENT ON COLUMN chunk.tags IS 'Educational metadata: pedagogy_role, content_type, difficulty, prerequisites, etc.';

-- Create schema_migrations table if it doesn't exist
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INT PRIMARY KEY,
  name TEXT NOT NULL,
  applied_at TIMESTAMPTZ DEFAULT NOW()
);

-- Migration metadata
INSERT INTO schema_migrations (version, name, applied_at) 
VALUES (1, 'add_chunk_tags_column', NOW())
ON CONFLICT (version) DO NOTHING;
