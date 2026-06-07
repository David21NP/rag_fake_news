# import logging
# import sys
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from config import get_settings
from rag.utils import (
    create_generator,
    extract_text_from_pdf,
)
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


@app.post("/ask-file")
def ask_if_fake_news_file(
    top_k: Annotated[int | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
):
    if not file:
        raise HTTPException(
            status_code=401,
            detail="Must send 'file'",
        )

    plain_text = ""
    if file.content_type == "plain/text":
        plain_text = file.file.read().decode("utf-8")
    else:
        plain_text = extract_text_from_pdf(file.file.read())

    return {
        "ok": True,
        "response": generate_response(
            text=plain_text,
            top_k=top_k,
        ),
    }


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
