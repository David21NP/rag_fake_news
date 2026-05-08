import io

import ollama
import psycopg2
import PyPDF2

from config import Settings


def extract_text_from_pdf(pdf_content: bytes):
    """Extract text from a PDF file content."""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() if page.extract_text() else ""
    return pdf_text


def save_text(plain_text: str, settings: Settings):
    with psycopg2.connect(
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
    ) as conn:
        with conn.cursor() as cur:
            client_ollama = ollama.Client(host=str(settings.ollama_url))
            embedding: ollama.EmbedResponse = client_ollama.embed(
                model="embeddinggemma",
                input=plain_text,
            )
            cur.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                (plain_text, embedding.embeddings[0]),
            )


def search_text(search_query: str, settings: Settings):
    with psycopg2.connect(
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
    ) as conn:
        with conn.cursor() as cur:
            client_ollama = ollama.Client(host=str(settings.ollama_url))
            embedding: ollama.EmbedResponse = client_ollama.embed(
                model="embeddinggemma",
                input=search_query,
            )
            cur.execute(
                (
                    "SELECT content, embedding <=> %s::vector AS distance "
                    "FROM documents "
                    "ORDER BY distance ASC "
                    "LIMIT 3;"
                ),
                (embedding.embeddings[0],),
            )
            return cur.fetchall()


def make_query(search_query: str, settings: Settings):
    results = search_text(search_query, settings)

    context = "\n".join([r[0] for r in results])

    prompt = f"""
    You are an assistant that answers questions using the context below.

    Context:
    {context}

    Question: {search_query}
    """

    client_ollama = ollama.Client(host=str(settings.ollama_url))
    response: ollama.ChatResponse = client_ollama.chat(
        # model="nomic-embed-text",
        model="qwen3",
        messages=[{"role": "user", "content": prompt}],
        think=True,
        stream=False,
    )

    return {
        "Thinking": response.message.thinking,
        "answer": response.message.content,
    }

def pull_models(settings: Settings):
    client_ollama = ollama.Client(host=str(settings.ollama_url))
    client_ollama.pull(model="embeddinggemma")
    client_ollama.pull(model="qwen3")
