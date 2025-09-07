-- chunks table for structural & semantic chunks
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
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  search_tsv tsvector
);

CREATE INDEX IF NOT EXISTS idx_chunk_resource_id ON chunk(resource_id);


