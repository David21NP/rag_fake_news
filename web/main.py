import logging
import sys
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from config import Settings, get_settings
from rag.utils import extract_text_from_pdf, make_query, pull_models, save_text

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


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/text")
def add_text(
    settings: Annotated[Settings, Depends(get_settings)],
    text: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
):
    plain_text = ""
    if not text and not file:
        raise HTTPException(
            status_code=401, detail="Must either send text or file"
        )

    if file:
        if file.content_type == "plain/text":
            plain_text = file.file.read().decode("utf-8")
        else:
            plain_text = extract_text_from_pdf(file.file.read())

    if text:
        plain_text = text

    # logger.info("====================================")
    # logger.info(settings.db_user)
    # logger.info(settings.db_password)
    # logger.info(settings.db_name)
    # logger.info(settings.db_host)
    # logger.info("====================================")

    save_text(plain_text, settings)

    return {"ok": True, "msg": "Text added to vector db"}


@app.post("/ask")
def ask(
    settings: Annotated[Settings, Depends(get_settings)],
    query: str,
):
    return {"ok": True, "response": make_query(query, settings)}


@app.get("/pull_models")
def pull_models_ollama(settings: Annotated[Settings, Depends(get_settings)]):
    pull_models(settings)
    return {"ok": True}
