#!/usr/bin/env bash
set -e

# NOTE: Activar la extension de vectores y crear tablas de embeddings
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION vector;

    -- NOTE: Crear tabla principal de embeddings
    CREATE TABLE documents_principal (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding VECTOR($EMBEDDING_SIZE)
    );
    CREATE INDEX ON documents_principal USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);


    -- NOTE: Crear tabla ablation de embeddings
    CREATE TABLE documents_ablation (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding VECTOR($EMBEDDING_SIZE)
    );
    CREATE INDEX ON documents_ablation USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
EOSQL

# NOTE: Crear tabla de logs de tests
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE TABLE verifications (
        id SERIAL PRIMARY KEY,
        news_content TEXT,
        retrieved_docs JSONB,      -- los k documentos recuperados
        prompt TEXT,               -- el prompt exacto enviado al LLM
        llm_reasoning TEXT,        -- el campo "thinking" de Qwen3
        label VARCHAR(10),         -- 'fake' o 'real'
        confidence FLOAT,
        created_at TIMESTAMP DEFAULT NOW()
    );
EOSQL

