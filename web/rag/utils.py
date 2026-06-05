import io
import os
from itertools import repeat
from typing import Callable, Literal

import ollama
import psycopg2
import psycopg2.extras
import psycopg2.sql
import PyPDF2
from fastapi import HTTPException

from config import Settings
from schemas import GeneratedResponse, ModelResponse

PROMPT_PATH = os.path.join(os.path.dirname(__file__))


def extract_text_from_pdf(pdf_content: bytes):
    """Extract text from a PDF file content."""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() if page.extract_text() else ""
    return pdf_text.replace("\x00", "")


def build_chunk_full(
    title: str | None,
    text: str,
) -> str:
    if title is None:
        return text
    return f"{title}\n\n{text}"


def build_chunks_sliding(
    title: str | None,
    text: str,
    chunk_size: int = 256,
    overlap: int = 64,
) -> list[str]:
    ## If the text has already the title line in a pdf
    if title is None:
        full_text = text
    else:
        full_text = f"{title}\n\n{text}"

    words = full_text.split()
    chunks: list[str] = []
    step = chunk_size - overlap  # 192 palabras de avance por chunk

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks


def select_by_model_chunk(
    settings: Settings,
    model: str,
    chunk_type: Literal["full", "sliding"],
) -> tuple[str, Callable[[str | None, str], str | list[str]]]:
    if model == settings.ollama_model_embedding_principal:
        if chunk_type == "full":
            return "documents_emba_chunka", build_chunk_full
        else:
            return "documents_emba_chunkb", build_chunks_sliding
    else:
        if chunk_type == "full":
            return "documents_embb_chunka", build_chunk_full
        else:
            return "documents_embb_chunkb", build_chunks_sliding


def create_embedder(
    model: str,
    settings: Settings,
    chunk_type: Literal["full", "sliding"],
):
    table, chunk_fn = select_by_model_chunk(settings, model, chunk_type)

    def create_and_save_embedding(
        *,
        text: str,
        label: int,
        title: str | None = None,
    ):
        chunks_or_text = chunk_fn(title, text)

        with psycopg2.connect(
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
        ) as conn:
            query = psycopg2.sql.SQL(
                (
                    "INSERT INTO {table} (content, label, embedding)"
                    " VALUES (%s, %s, %s)"
                )
            ).format(table=psycopg2.sql.Identifier(table))
            with conn.cursor() as cur:
                client_ollama = ollama.Client(host=str(settings.ollama_url))
                embedding: ollama.EmbedResponse = client_ollama.embed(
                    model=model,
                    input=chunks_or_text,
                )
                if isinstance(chunks_or_text, list):
                    psycopg2.extras.execute_batch(
                        cur,
                        query,
                        zip(
                            chunks_or_text, repeat(label), embedding.embeddings
                        ),
                    )
                else:
                    cur.execute(
                        query,
                        (chunks_or_text, label, embedding.embeddings[0]),
                    )

    return create_and_save_embedding


def search_similar_embedding(
    *,
    text: str,
    settings: Settings,
    model: str,
    top_k: int | None = None,
    title: str | None = None,
    chunk_type: Literal["full"] | Literal["sliding"],
):
    if top_k is None:
        top_k = settings.top_k_selected

    table, _ = select_by_model_chunk(settings, model, chunk_type)
    text_to_test = build_chunk_full(title, text)

    with psycopg2.connect(
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
    ) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            client_ollama = ollama.Client(host=str(settings.ollama_url))
            embedding: ollama.EmbedResponse = client_ollama.embed(
                model=model,
                input=text_to_test,
            )
            cur.execute(
                psycopg2.sql.SQL(
                    (
                        "SELECT content, label, "
                        "embedding <=> %s::vector AS distance "
                        "FROM {table} "
                        "ORDER BY distance ASC "
                        "LIMIT %s;"
                    )
                ).format(table=psycopg2.sql.Identifier(table)),
                (embedding.embeddings[0], top_k),
            )
            return cur.fetchall(), text_to_test


def create_generator(
    model: str,
    settings: Settings,
    chunk_type: Literal["full", "sliding"],
):
    def generate_response(
        *,
        text: str,
        title: str | None = None,
        top_k: int | None = None,
        think: bool = True,
        temperature: float | None = None,
    ):
        if temperature is None:
            temperature = settings.temperature_selected

        results, text_to_test = search_similar_embedding(
            text=text,
            settings=settings,
            model=model,
            top_k=top_k,
            title=title,
            chunk_type=chunk_type,
        )
        context = "\n\n".join(
            [f"[{i + 1}] {r[0]}" for i, r in enumerate(results)]
        )

        system_prompt: str = ""
        with open(
            os.path.join(PROMPT_PATH, "system_prompt.txt"),
            "r",
            encoding="utf-8",
        ) as user_file:
            system_prompt = user_file.read()

        user_prompt: str = ""
        with open(
            os.path.join(PROMPT_PATH, "user_prompt.txt"),
            "r",
            encoding="utf-8",
        ) as user_file:
            user_prompt = user_file.read().format(
                context=context,
                query=text_to_test,
            )

        client_ollama = ollama.Client(host=str(settings.ollama_url))
        response: ollama.ChatResponse = (
            client_ollama.chat(  # pyright: ignore[reportUnknownMemberType]
                model=settings.ollama_model_llm_generador,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                think=think,
                stream=False,
                format=ModelResponse.model_json_schema(),
                options={
                    "num_ctx": settings.ollama_num_ctx,
                    "temperature": temperature,
                },
            )
        )

        if response.message.content is None:
            raise HTTPException(500, "Content from llm empty")

        answer = ModelResponse.model_validate_json(response.message.content)

        ## Add verification
        with psycopg2.connect(
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                        INSERT INTO verifications (
                            news_content,
                            retrieved_docs,
                            prompt,
                            llm_reasoning,
                            label,
                            confidence
                        ) VAlUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        text_to_test,
                        psycopg2.extras.Json(results),
                        (
                            f"SYSTEM PROMPT:\n{system_prompt}"
                            f"\n\nUSER_PROMPT:\n{user_prompt}"
                        ),
                        response.message.thinking,
                        answer.label,
                        answer.confidence,
                    ),
                )

        return GeneratedResponse(
            thinking=response.message.thinking,
            answer=answer,
        )

    return generate_response
