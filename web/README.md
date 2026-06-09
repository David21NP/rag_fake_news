# web/

Aplicación FastAPI que implementa el pipeline RAG completo: indexación de documentos, recuperación semántica sobre pgvector e inferencia con Qwen3 vía Ollama.

---

## Estructura

```
web/
├── main.py                   # Endpoints FastAPI
├── config.py                 # Configuración vía variables de entorno (pydantic-settings)
├── schemas.py                # Tipos compartidos: Metrics, ModelResponse, GeneratedResponse
├── common/
│   └── utils.py              # Carga de datasets, loading_bar, iter_batches
├── rag/
│   └── utils.py              # Núcleo: embed, búsqueda semántica, prompt, inferencia LLM
├── linguistic/
│   ├── __init__.py           # run_NC_LBFV — entrena y evalúa el baseline lingüístico
│   └── utils.py              # Extracción de features: TF-IDF, spaCy, NRCLex, RF
├── scripts/
│   ├── add_source_index.py   # Backfill de source_index en tablas existentes
│   ├── preparations/
│   │   └── create_embeddings.py  # Indexación del corpus en pgvector (con resume y backup)
│   └── experiments/
│       ├── run_oe4_ablation.py   # OE4: ablation 32 configuraciones (think=off)
│       └── run_oe3_baselines.py  # OE3: comparación vs. baselines
└── tests/
    └── linguistic/
        └── test_pipeline.py   # Tests del baseline NC-LBFV
```

---

## API

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Ping |
| `GET` | `/health` | Health check de base de datos |
| `POST` | `/text` | Indexar texto o PDF en la knowledge base |
| `POST` | `/ask` | Clasificar una noticia (fake / real) |

### Ejemplo — indexar texto

```bash
curl -X POST http://localhost:8000/text \
  -F "text=El presidente firmó hoy un nuevo acuerdo..." \
  -F "label=1"
```

### Ejemplo — clasificar noticia

```bash
curl -X POST "http://localhost:8000/ask?text=El+gobierno+anuncia+nueva+ley..."
```

---

## Configuración

Variables de entorno (definidas en `docker-compose` o `models.env`):

| Variable | Descripción |
|---|---|
| `DB_HOST` | Host de PostgreSQL |
| `DB_NAME` | Nombre de la base de datos |
| `DB_USER` | Usuario de PostgreSQL |
| `DB_PASSWORD_FILE` | Ruta al archivo con la contraseña (Docker secrets) |
| `OLLAMA_URL` | URL del servicio Ollama |
| `OLLAMA_MODEL_EMBEDDING_PRINCIPAL` | Modelo de embedding principal (`mxbai-embed-large:335m-v1-fp16`) |
| `OLLAMA_MODEL_EMBEDDING_ABLATION` | Modelo de embedding ablation (`nomic-embed-text:v1.5`) |
| `OLLAMA_MODEL_LLM_GENERADOR` | Modelo LLM (`qwen3:8b-q4_K_M`) |
| `OLLAMA_NUM_CTX` | Tamaño del contexto LLM (8192 — obligatorio para k≥5) |

---

## Indexación del corpus

La indexación se ejecuta **automáticamente** al arrancar el contenedor `web_ai`, a través del lifespan de FastAPI (`main.py`). No requiere intervención manual.

Al completarse, se genera automáticamente un backup de la base de datos en `web/backups/`.

Comportamiento:
- Procesa los 24.353 artículos del split `train` en batches de 128
- Inserta en las 4 tablas en paralelo (ThreadPoolExecutor, 4 workers)
- Soporta **resume automático**: si el contenedor se reinicia, reanuda desde el último artículo consistente entre todas las tablas

### Backfill de source_index (solo si se interrumpió antes de implementar resume)

```bash
PYTHONPATH=/code/app python /code/app/scripts/add_source_index.py
```

---

## Experimentos

Los scripts se ejecutan con Python directamente (fuera del contenedor), apuntando a los servicios levantados via `localhost`.

### OE4 — Ablation de parámetros RAG

32 configuraciones: 2 embeddings × 2 chunking × 4 top-k × 2 temperaturas × thinking off

```bash
cd web/
PYTHONPATH=. python scripts/experiments/run_oe4_ablation.py

# docker exec -it <cont-name> bash -c "PYTHONPATH=/code/app python /code/app/scripts/experiments/run_oe4_ablation.py"

docker exec <cont-name> bash -c "PYTHONPATH=/code/app nohup python /code/app/scripts/experiments/run_oe4_ablation.py > /code/app/results/oe4.log 2>&1 &"
docker exec <cont-name> tail -f /code/app/results/oe4.log
```

Output: `web/results/oe4_ablation.csv`

Columnas: `config_id, embedding, chunking, top_k, temperature, thinking, accuracy, precision, recall, f1, latency_mean_sec`

Soporta **resume**: si se interrumpe, reanuda desde la última configuración completada.

### OE4 — Experimento thinking

2 configuraciones: configuración principal (EMB-A / CHUNK-A / k=3 / T=0.0) con think=off vs think=on

```bash
docker exec <cont-name> bash -c "PYTHONPATH=/code/app nohup python /code/app/scripts/experiments/run_oe4_ablation_think.py > /code/app/results/oe4_think.log 2>&1 &"
docker exec <cont-name> tail -f /code/app/results/oe4_think.log
```

Output: `web/results/oe4_ablation_think.csv`

Soporta **resume**: si se interrumpe, reanuda desde la última configuración completada.

---

### OE3 — Comparación con baselines

1.000 ejemplos estratificados (500 fake + 500 real) del split `test`.

```bash
cd web/
PYTHONPATH=. python scripts/experiments/run_oe3_baselines.py

# docker exec -it <cont-name> bash -c "PYTHONPATH=/code/app python /code/app/scripts/experiments/run_oe3_baselines.py"

docker exec <cont-name> bash -c "PYTHONPATH=/code/app nohup python /code/app/scripts/experiments/run_oe3_baselines.py > /code/app/results/oe3.log 2>&1 &"
docker exec <cont-name> tail -f /code/app/results/oe3.log
```

Output: `web/results/oe3_baselines.csv`

Sistemas comparados:
1. RAG (configuración principal: EMB-A / CHUNK-A / k=3 / T=0.0 / think=off)
2. NC-LBFV (Hashemi et al., 2026) — Random Forest sobre TF-IDF + spaCy + NRCLex
3. Nezafat & Samet (2024) — resultados publicados (dataset ISOT, referencia externa)

---

## Tests

```bash
pytest tests/
```
