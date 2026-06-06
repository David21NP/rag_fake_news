-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- -------------------------------------------------------------------------
-- KNOWLEDGE BASE TABLES
-- One table per embedding x chunking combination (ablation OE4)
-- -------------------------------------------------------------------------
-- EMB-A (mxbai-embed-large, 1024d) x CHUNK-A (full article)
CREATE TABLE IF NOT EXISTS documents_emba_chunka (
    id serial PRIMARY KEY,
    source_index INTEGER,
    content text NOT NULL,
    label smallint NOT NULL, -- 0=fake, 1=real
    embedding VECTOR (1024)
);

CREATE INDEX IF NOT EXISTS idx_emba_chunka_embedding ON documents_emba_chunka USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- EMB-A (mxbai-embed-large, 1024d) x CHUNK-B (256 tokens / overlap 64)
CREATE TABLE IF NOT EXISTS documents_emba_chunkb (
    id serial PRIMARY KEY,
    source_index INTEGER,
    content text NOT NULL,
    label smallint NOT NULL,
    embedding VECTOR (1024)
);

CREATE INDEX IF NOT EXISTS idx_emba_chunkb_embedding ON documents_emba_chunkb USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);

-- EMB-B (nomic-embed-text, 768d) x CHUNK-A (full article)
CREATE TABLE IF NOT EXISTS documents_embb_chunka (
    id serial PRIMARY KEY,
    source_index INTEGER,
    content text NOT NULL,
    label smallint NOT NULL,
    embedding VECTOR (768)
);

CREATE INDEX IF NOT EXISTS idx_embb_chunka_embedding ON documents_embb_chunka USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- EMB-B (nomic-embed-text, 768d) x CHUNK-B (256 tokens / overlap 64)
CREATE TABLE IF NOT EXISTS documents_embb_chunkb (
    id serial PRIMARY KEY,
    source_index INTEGER,
    content text NOT NULL,
    label smallint NOT NULL,
    embedding VECTOR (768)
);

CREATE INDEX IF NOT EXISTS idx_embb_chunkb_embedding ON documents_embb_chunkb USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);

--
-------------------------------------------------------------------------
-- VERIFICATIONS TABLE
-- Audit log of all predictions made by the system
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS verifications (
    id serial PRIMARY KEY,
    news_content text NOT NULL,
    retrieved_docs jsonb, -- los k documentos recuperados
    prompt text, -- el prompt exacto enviado al LLM
    llm_reasoning text, -- el campo "thinking" de Qwen3
    label varchar(10), -- 'fake' o 'real'
    confidence float,
    created_at timestamp DEFAULT NOW()
);

