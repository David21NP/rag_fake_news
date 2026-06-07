# import logging
# import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from config import get_settings
from rag.utils import create_generator
from scripts.preparations.create_embeddings import add_embeddings, save_backup
from utils import test_db_connection


@asynccontextmanager
async def lifespan(app: FastAPI):
    add_embeddings()
    save_backup()
    yield


app = FastAPI(lifespan=lifespan)

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# # StreamHandler für die Konsole
# stream_handler = logging.StreamHandler(sys.stdout)
# log_formatter = logging.Formatter(
#     "%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s"
# )
# stream_handler.setFormatter(log_formatter)
# logger.addHandler(stream_handler)
# logger.info("API is starting up")

generate_response = create_generator(
    settings=get_settings(),
    model=get_settings().ollama_model_embedding_selected,
    chunk_type=get_settings().chunk_selected,
    experiment=False,
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def health():
    try:
        test_db_connection()
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=503, detail="db unavailable")


@app.post("/ask")
def ask_if_fake_news(
    text: str,
    title: str | None = None,
    top_k: int | None = None,
):
    return {
        "ok": True,
        "response": generate_response(
            title=title,
            text=text,
            top_k=top_k,
        ),
    }
