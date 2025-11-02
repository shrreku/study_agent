-- Sprint 5 — user_doubt table
-- Idempotent creation; use uuid-ossp for UUIDs (consistent with existing schema)

CREATE TABLE IF NOT EXISTS user_doubt (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL,
  question TEXT NOT NULL,
  asked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  concepts TEXT[] NOT NULL DEFAULT '{}',
  weak_signal BOOLEAN NOT NULL DEFAULT FALSE
);

-- Helpful index for analytics by user and time
CREATE INDEX IF NOT EXISTS idx_user_doubt_user_time ON user_doubt(user_id, asked_at DESC);
