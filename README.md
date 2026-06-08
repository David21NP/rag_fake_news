# RAG Fake News Detector

Sistema de detección automática de noticias falsas basado en arquitectura RAG (Retrieval-Augmented Generation). Combina recuperación semántica sobre una base de conocimiento estática de noticias verificadas con el razonamiento de un LLM local (Qwen3-8B) para clasificar noticias como `fake` o `real`.

Desarrollado como Trabajo de Fin de Máster — Máster Universitario en Ciberseguridad, Universidad de Salamanca.

---

## Arquitectura

```
noticia entrada
      │
      ▼
embedding (mxbai-embed-large o nomic-embed-text)
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
etiqueta (fake / real) + confianza + razonamiento
```

---

## Entornos

| Archivo       | Entorno                         | Uso                                   |
| ------------- | ------------------------------- | ------------------------------------- |
| `laptop.yaml` | Local con GPU (GTX 1050)        | Indexación del corpus                         |
| `server.yaml` | Servidor sin GPU (Hetzner CX43) | Experimentos OE3/OE4 y prueba de latencia OE2 |

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
# o
python -c 'import secrets; print(secrets.token_hex())' > secrets/password.txt


# Construir y levantar (entorno laptop)
./scripts/build_laptop.sh
./scripts/run_laptop.sh
```

La API queda disponible en `http://localhost:8000`. Documentación interactiva en `http://localhost:8000/docs`.

---

## Scripts

| Script                                 | Descripción                                          |
| -------------------------------------- | ---------------------------------------------------- |
| `scripts/build_laptop.sh`              | Build y run de imágenes Docker para entorno local    |
| `scripts/build_server.sh`              | Build y run de imágenes Docker para entorno servidor |
| `scripts/run_laptop.sh`                | Levantar stack completo en laptop (con GPU)          |
| `scripts/run_server.sh`                | Levantar stack completo en servidor (CPU)            |
| `scripts/container/download_models.sh` | Descargar modelos Ollama dentro del contenedor       |

---

## Módulos

| Módulo               | Ruta                  | Descripción                                                                        |
| -------------------- | --------------------- | ---------------------------------------------------------------------------------- |
| Pipeline RAG         | `web/rag/utils.py`    | Núcleo: embed, búsqueda semántica, construcción de prompt, inferencia LLM          |
| Baseline lingüístico | `web/linguistic/`     | Clasificador NC-LBFV: TF-IDF + embeddings spaCy + emociones NRCLex + Random Forest |
| Utilidades comunes   | `web/common/utils.py` | Carga de datasets, barra de progreso, iteración por batches                        |

---

## Experimentos

Los scripts de evaluación están en `web/scripts/experiments/` y se ejecutan directamente con Python apuntando a los servicios levantados.

| Script                 | Objetivo                             | Estado       |
| ---------------------- | ------------------------------------ | ------------ |
| `run_oe4_ablation.py`  | OE4 — ablation de 32 configuraciones | Implementado |
| `run_oe3_baselines.py` | OE3 — comparación vs. baselines      | Implementado |

Ver `web/README.md` para instrucciones de ejecución.

---

## Base de datos

El esquema se inicializa automáticamente al levantar el contenedor `pgvector` mediante `db/init-db.sql`. Se crean las siguientes tablas:

| Tabla                   | Embedding         | Chunking                         | Dims                   |
| ----------------------- | ----------------- | -------------------------------- | ---------------------- |
| `documents_emba_chunka` | mxbai-embed-large | Artículo completo                | 1024                   |
| `documents_emba_chunkb` | mxbai-embed-large | Chunks 256 palabras / overlap 64 | 1024                   |
| `documents_embb_chunka` | nomic-embed-text  | Artículo completo                | 768                    |
| `documents_embb_chunkb` | nomic-embed-text  | Chunks 256 palabras / overlap 64 | 768                    |
| `verifications`         | —                 | —                                | Log de clasificaciones |

Todas las tablas de documentos incluyen la columna `source_index INTEGER` para soporte de resume en la indexación.

### Exportar e importar la knowledge base

```bash
# Exportar desde el contenedor pgvector
docker exec <pgvector-container> pg_dump -U root rag > ./db/rag_dump.sql

# Enviar al servidor vía scp
scp ./db/rag_dump.sql user@<server-ip>:/path/to/repo/db/rag_dump.sql

# Enviar al servidor vía rsync (más eficiente para archivos grandes)
rsync -avz --progress ./db/rag_dump.sql user@<server-ip>:/path/to/repo/db/rag_dump.sql

# El archivo se restaura automáticamente en un contenedor limpio
# via docker-entrypoint-initdb.d/z-restore-db.sql
```

---

## Flujo de indexación

La indexación se lanza **automáticamente** al arrancar el contenedor `web_ai` vía el lifespan de FastAPI. Al completarse, se genera un backup automático en `web/backups/`.

El proceso soporta resume automático: si el contenedor se reinicia, reanuda desde el último artículo procesado en todas las tablas de forma consistente.
