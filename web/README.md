# web/

Aplicación FastAPI que implementa el pipeline RAG completo: indexación de documentos, recuperación semántica sobre pgvector e inferencia con Qwen3 vía Ollama.

---

## Estructura

```
web/
├── main.py               # Endpoints FastAPI
├── config.py             # Configuración vía variables de entorno (pydantic-settings)
├── utils.py              # Utilidades generales
├── rag/
│   └── utils.py          # Núcleo del pipeline: embed, search, make_query
├── scripts/
│   └── experiments/
│       ├── run_oe4_ablation.py   # Ablation de parámetros RAG (OE4)
│       └── run_oe3_baselines.py  # Comparación con baselines (OE3)
└── tests/
    ├── rag/
    │   └── test_create_embedding.py
    └── linguistic/
        └── test_pipeline.py
```

---

## API

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Health check db |
| `POST` | `/text` | Indexar texto o PDF en la knowledge base |
| `POST` | `/ask` | Clasificar una noticia (fake / real) |
| `GET` | `/pull_models` | Descargar modelos Ollama |

### Ejemplo — indexar texto

```bash
curl -X POST http://localhost:8000/text \
  -F "text=El presidente firmó hoy un nuevo acuerdo comercial con..."
```

### Ejemplo — clasificar noticia

```bash
curl -X POST "http://localhost:8000/ask?query=El+gobierno+anuncia+nueva+ley..."
```

---

## Configuración

Variables de entorno (definidas en el `docker-compose` o en un `.env` local):

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `DB_HOST` | `localhost` | Host de PostgreSQL |
| `DB_NAME` | `postgres` | Nombre de la base de datos |
| `DB_USER` | `postgres` | Usuario de PostgreSQL |
| `DB_PASSWORD_FILE` | — | Ruta al archivo con la contraseña (Docker secrets) |
| `OLLAMA_URL` | `http://ollama:11434` | URL del servicio Ollama |

---

## Tests

```bash
# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Ejecutar todos los tests
pytest tests/

# Ejecutar solo los tests RAG
pytest tests/rag/

# Con cobertura
pytest tests/ --cov=rag --cov=main
```

---

## Experimentos

Los scripts de experimentos se ejecutan directamente con Python, fuera del contenedor, apuntando a los servicios levantados.

### OE4 — Ablation de parámetros RAG

Evalúa 16 configuraciones (2 embeddings × 2 estrategias de chunking × 4 valores de top-k) sobre un subset estratificado de 500 ejemplos.

```bash
python scripts/experiments/run_oe4_ablation.py \
  --dataset gonzaloa \
  --subset 500 \
  --output results/oe4_ablation.csv
```

### OE3 — Comparación con baselines

Evalúa la configuración principal del sistema (mxbai-embed-large / noticia completa / k=3) sobre 1.000–2.000 ejemplos para comparar con NC-LBFV y Nezafat & Samet (2024).

```bash
python scripts/experiments/run_oe3_baselines.py \
  --dataset gonzaloa \
  --subset 2000 \
  --output results/oe3_baselines.csv
```

Los resultados se guardan en CSV con columnas: `config`, `accuracy`, `precision`, `recall`, `f1`, `latency_mean`.
