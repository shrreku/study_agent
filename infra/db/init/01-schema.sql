-- schema for StudyAgent minimal tables for Sprint 1

CREATE TABLE IF NOT EXISTS app_user (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email TEXT UNIQUE,
  password_hash TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

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

CREATE TABLE IF NOT EXISTS extracted_page (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  resource_id UUID REFERENCES resource(id) ON DELETE CASCADE,
  page_number INT,
  raw_text TEXT,
  ocr_confidence NUMERIC NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

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

-- simple index for lookup
CREATE INDEX IF NOT EXISTS idx_resource_created_at ON resource(created_at);


