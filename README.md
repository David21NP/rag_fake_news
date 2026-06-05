# RAG Fake News Detector

Sistema de detección automática de noticias falsas basado en arquitectura RAG (Retrieval-Augmented Generation). Combina recuperación semántica sobre una base de conocimiento de noticias verificadas con el razonamiento de un LLM local (Qwen3) para clasificar noticias como `fake` o `real`.

Desarrollado como Trabajo de Fin de Máster — Máster Universitario en Ciberseguridad, Universidad de Salamanca.

---

## Arquitectura

```
noticia entrada
      │
      ▼
embedding (mxbai-embed-large)
      │
      ▼
búsqueda semántica en pgvector (top-k documentos)
      │
      ▼
construcción de prompt con contexto recuperado
      │
      ▼
inferencia LLM (Qwen3-8B Q4)
      │
      ▼
etiqueta (fake / real) + razonamiento + log en verifications
```

---

## Entornos

El proyecto tiene dos configuraciones Docker Compose:

| Archivo | Entorno | Uso |
|---|---|---|
| `laptop.yaml` | Local con GPU (GTX 1050) | Desarrollo, indexación del corpus y experimentos |
| `server.yaml` | Servidor sin GPU (Hetzner CX32) | Despliegue y prueba de latencia en producción |

---

## Requisitos previos

- Docker y Docker Compose
- GPU NVIDIA con drivers actualizados (solo para `laptop.yaml`)
- Archivo `secrets/password.txt` con la contraseña de PostgreSQL

---

## Inicio rápido

```bash
# Clonar y entrar al repositorio
git clone <repo-url>
cd <repo>

# Crear el archivo de secretos
echo "yoursecretpassword" > secrets/password.txt

# Construir y levantar (entorno laptop)
./scripts/build_laptop.sh
# Levantar (entorno laptop)
./scripts/run_laptop.sh

# Descargar modelos en Ollama (primera vez)
./scripts/container/download_models.sh
```

La API queda disponible en `http://localhost:8000`. Documentación interactiva en `http://localhost:8000/docs`.

---

## Scripts

| Script | Descripción |
|---|---|
| `scripts/build_laptop.sh` | Build de imágenes Docker para entorno local |
| `scripts/build_server.sh` | Build de imágenes Docker para entorno servidor |
| `scripts/run_laptop.sh` | Levantar stack completo en laptop (con GPU) |
| `scripts/run_server.sh` | Levantar stack completo en servidor (CPU) |
| `scripts/container/download_models.sh` | Descargar modelos Ollama dentro del contenedor |

---

## Módulos

| Módulo | Ruta | Descripción |
|---|---|---|
| Pipeline RAG | `web/rag/utils.py` | Núcleo: embed, búsqueda semántica, construcción de prompt, inferencia LLM |
| Baseline lingüístico | `web/linguistic/pipeline.py` | Clasificador NC-LBFV: TF-IDF + embeddings spaCy + emociones NRCLex + Random Forest |

---

## Experimentos

Los scripts de evaluación están en `web/scripts/experiments/`:

| Script | Objetivo | Descripción |
|---|---|---|
| `run_oe4_ablation.py` | OE4 | Ablation de parámetros RAG: embedding × chunking × top-k (16 configuraciones) |
| `run_oe3_baselines.py` | OE3 | Comparación del sistema vs. baselines sobre el subset de evaluación |

> **Work in progress** — los scripts de experimentos son stubs vacíos pendientes de implementación.

Ver `web/README.md` para instrucciones de ejecución de los experimentos.

---

## Tests

> **Work in progress** — los archivos de test son stubs vacíos pendientes de implementación.

```bash
# Desde el directorio web/
pytest tests/
```

| Módulo | Ruta | Descripción |
|---|---|---|
| RAG | `tests/rag/test_create_embedding.py` | Generación de embeddings y operaciones sobre pgvector |
| Lingüístico | `tests/linguistic/test_pipeline.py` | Pipeline de clasificación lingüística (baseline OE3) |

---

## Base de datos

El esquema se inicializa automáticamente al levantar el contenedor `pgvector` mediante `db/init-vector.sh`. Se crean las siguientes tablas:

| Tabla | Descripción |
|---|---|
| `documents_principal` | Embeddings con `mxbai-embed-large` (configuración principal, 1024 dims) |
| `documents_ablation` | Embeddings con `nomic-embed-text` (configuración ablation) |
| `verifications` | Log de clasificaciones: noticia, docs recuperados, prompt, razonamiento LLM, etiqueta y confianza |


Para exportar la knowledge base indexada y desplegarla en el servidor:

```bash
# Exportar desde laptop
pg_dump -h localhost -U root -d rag > rag_dump.sql
docker exec postgres pg_dump rag > ./db/rag_dump.sql

# Importar en servidor
psql -h <server-host> -U root -d rag < rag_dump.sql
```
