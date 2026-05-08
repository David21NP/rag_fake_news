#!/usr/bin/env bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION vector;
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding VECTOR($EMBEDDING_SIZE)
    );
    CREATE INDEX ON documents USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
EOSQL

