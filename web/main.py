# import logging
# import sys
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from config import get_settings
from rag.utils import (
    create_embedder,
    extract_text_from_pdf,
    create_generator,
)
from utils import test_db_connection

app = FastAPI()

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

create_and_save_embedding = create_embedder(
    settings=get_settings(),
    model=get_settings().ollama_model_embedding_selected,
    chunk_type=get_settings().chunk_selected,
)
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


@app.post("/text")
def add_text(
    text: Annotated[str | None, Form()] = None,
    title: Annotated[str | None, Form()] = None,
    label: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
):
    if not label:
        raise HTTPException(
            status_code=401,
            detail="'label' is required",
        )

    try:
        label_int = int(label)
    except ValueError as value_error:
        raise HTTPException(
            status_code=401,
            detail="'label' must be a valid number",
        ) from value_error

    plain_text = ""
    if not text and not file:
        raise HTTPException(
            status_code=401,
            detail="Must either send 'text' or 'file'",
        )

    if file:
        if file.content_type == "plain/text":
            plain_text = file.file.read().decode("utf-8")
        else:
            plain_text = extract_text_from_pdf(file.file.read())

    if text:
        plain_text = text

    create_and_save_embedding(
        title=title,
        text=plain_text,
        label=label_int,
    )

    return {"ok": True, "msg": "Text added to vector db"}


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
