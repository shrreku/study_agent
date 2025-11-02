-- tutor session persistence tables

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

CREATE INDEX IF NOT EXISTS idx_tutor_session_user_created ON tutor_session (user_id, created_at DESC);

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

CREATE UNIQUE INDEX IF NOT EXISTS idx_tutor_turn_session_turn_index ON tutor_turn (session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_tutor_turn_created ON tutor_turn (created_at);

CREATE TABLE IF NOT EXISTS tutor_event (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id UUID NOT NULL REFERENCES tutor_session(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,
  payload JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tutor_event_session_created ON tutor_event (session_id, created_at);
