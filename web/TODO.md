# TODO — web/

Pending changes to align the implementation with the designed architecture.

---

## rag/utils.py

- [ ] Replace hardcoded model names (`embeddinggemma`, `qwen3`) with env variables from `models.env` via `config.py` (`OLLAMA_MODEL_EMBEDDING_PRINCIPAL`, `OLLAMA_MODEL_LLM_GENERADOR`)
- [ ] Replace `documents` table references with `documents_principal` and `documents_ablation` (matching `db/init-vector.sh`)
- [ ] Update `make_query` prompt to classify the input news as `fake` or `real` and return a structured response (label + reasoning)
- [ ] Log each classification to the `verifications` table (news content, retrieved docs, prompt, LLM reasoning, label, confidence)
- [ ] Fix `pull_models` to use env variables instead of hardcoded model names

## config.py

- [ ] Add `OLLAMA_MODEL_EMBEDDING_PRINCIPAL`, `OLLAMA_MODEL_EMBEDDING_ABLATION`, and `OLLAMA_MODEL_LLM_GENERADOR` as settings fields (loaded from `models.env`)

## scripts/experiments/run_oe4_ablation.py

- [ ] Implement ablation: 2 embeddings × 2 chunking strategies × 4 top-k values = 16 configurations
- [ ] Run over a stratified subset of the dataset (default 500 examples)
- [ ] Save results to CSV with columns: `config`, `accuracy`, `precision`, `recall`, `f1`, `latency_mean`

## scripts/experiments/run_oe3_baselines.py

- [ ] Implement evaluation of the principal RAG config (mxbai-embed-large / full news / k=3) over 1,000–2,000 examples
- [ ] Compare against the NC-LBFV baseline (`linguistic/pipeline.py`) and Nezafat & Samet (2024)
- [ ] Save results to CSV with same columns as OE4

## tests/rag/test_create_embedding.py

- [ ] Implement tests for embedding generation and pgvector operations

## tests/linguistic/test_pipeline.py

- [ ] Implement tests for the NC-LBFV classification pipeline (`linguistic/pipeline.py`)
