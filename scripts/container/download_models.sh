#!/usr/bin/env bash

curl -X POST http://ollama:11434/api/pull -d "{\"name\": \"$OLLAMA_MODEL_EMBEDDING_PRINCIPAL\"}" &&
curl -X POST http://ollama:11434/api/pull -d "{\"name\": \"$OLLAMA_MODEL_EMBEDDING_ABLATION\"}" &&
curl -X POST http://ollama:11434/api/pull -d "{\"name\": \"$OLLAMA_MODEL_LLM_GENERADOR\"}"
