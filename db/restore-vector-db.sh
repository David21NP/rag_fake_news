#!/usr/bin/env bash
set -e

# Exportar desde laptop
# pg_dump -h localhost -U root -d rag > rag_dump.sql


FILE="/rag_dump.sql"

# Check if regular file exists
if [ -f "$FILE" ]; then
    # Importar en servidor
    echo "File exists."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" < "$FILE"
else
    echo "File does not exist."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        -- Enable pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;

        -- -------------------------------------------------------------------------
        -- KNOWLEDGE BASE TABLES
        -- One table per embedding x chunking combination (ablation OE4)
        -- -------------------------------------------------------------------------

        -- EMB-A (mxbai-embed-large, 1024d) x CHUNK-A (full article)
        CREATE TABLE IF NOT EXISTS documents_emba_chunka (
            id      SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            label   SMALLINT NOT NULL,  -- 0=fake, 1=real
            embedding VECTOR(1024)
        );
        CREATE INDEX IF NOT EXISTS idx_emba_chunka_embedding
            ON documents_emba_chunka
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);

        -- EMB-A (mxbai-embed-large, 1024d) x CHUNK-B (256 tokens / overlap 64)
        CREATE TABLE IF NOT EXISTS documents_emba_chunkb (
            id      SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            label   SMALLINT NOT NULL,
            embedding VECTOR(1024)
        );
        CREATE INDEX IF NOT EXISTS idx_emba_chunkb_embedding
            ON documents_emba_chunkb
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 200);

        -- EMB-B (nomic-embed-text, 768d) x CHUNK-A (full article)
        CREATE TABLE IF NOT EXISTS documents_embb_chunka (
            id      SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            label   SMALLINT NOT NULL,
            embedding VECTOR(768)
        );
        CREATE INDEX IF NOT EXISTS idx_embb_chunka_embedding
            ON documents_embb_chunka
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);

        -- EMB-B (nomic-embed-text, 768d) x CHUNK-B (256 tokens / overlap 64)
        CREATE TABLE IF NOT EXISTS documents_embb_chunkb (
            id      SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            label   SMALLINT NOT NULL,
            embedding VECTOR(768)
        );
        CREATE INDEX IF NOT EXISTS idx_embb_chunkb_embedding
            ON documents_embb_chunkb
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 200);

        --
        -------------------------------------------------------------------------
        -- VERIFICATIONS TABLE
        -- Audit log of all predictions made by the system
        -- -------------------------------------------------------------------------
        CREATE TABLE IF NOT EXISTS verifications (
            id             SERIAL PRIMARY KEY,
            news_content   TEXT NOT NULL,
            retrieved_docs JSONB,                   -- los k documentos recuperados
            prompt         TEXT,                    -- el prompt exacto enviado al LLM
            llm_reasoning  TEXT,                    -- el campo "thinking" de Qwen3
            label          VARCHAR(10),             -- 'fake' o 'real'
            confidence     FLOAT,
            created_at     TIMESTAMP DEFAULT NOW()
        );

    EOSQL
fi
