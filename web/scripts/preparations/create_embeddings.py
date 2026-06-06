import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import psycopg2
import psycopg2.sql

import common.utils
from config import get_settings
from rag.utils import create_embedder

EXPERIMENT_CHUNK_TYPES: list[Literal["full", "sliding"]] = ["full", "sliding"]


def get_current_embedding_index():
    settings = get_settings()
    tables = [
        "documents_emba_chunka",
        "documents_emba_chunkb",
        "documents_embb_chunka",
        "documents_embb_chunkb",
    ]
    with psycopg2.connect(
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
    ) as conn:
        with conn.cursor() as cur:
            min_si = float("inf")
            for t in tables:
                cur.execute(
                    psycopg2.sql.SQL(
                        "SELECT MAX(source_index) FROM {table}"
                    ).format(table=psycopg2.sql.Identifier(t))
                )
                curr_si = cur.fetchone()
                if curr_si is None:
                    raise ValueError("No data when looking source indexes")
                min_si = min(
                    min_si, int(curr_si[0] if curr_si[0] is not None else -1)
                )

            for t in tables:
                cur.execute(
                    psycopg2.sql.SQL(
                        "DELETE FROM {table} WHERE source_index > %s"
                    ).format(table=psycopg2.sql.Identifier(t)),
                    (min_si,),
                )

            return int(min_si)


def add_embeddings():
    batch_size: int = 128
    settings = get_settings()
    print("Loading train dataset ...")
    df_train = common.utils.get_df_train()
    total_data = len(df_train)
    print(f"Loaded: {total_data} samples")
    models = [
        settings.ollama_model_embedding_principal,
        settings.ollama_model_embedding_ablation,
    ]
    embedders = [
        create_embedder(
            settings=settings,
            model=model,
            chunk_type=chunk_type,
        )
        for model, chunk_type in itertools.product(
            models, EXPERIMENT_CHUNK_TYPES
        )
    ]
    print("Creating embeddings ...")
    start_time = time.perf_counter()
    curr_si = get_current_embedding_index()
    common.utils.loading_bar(
        ((curr_si + 1) / total_data) * 100.0,
        f"ETA ?m?s | Loading: ",
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, df_batch in common.utils.iter_batches(
            df_train,
            batch_size=batch_size,
            start=curr_si + 1,
        ):
            texts = (
                df_batch["title"].fillna("") + "\n\n" + df_batch["text"]
            ).tolist()
            labels = df_batch["label"].tolist()
            futures = [
                executor.submit(
                    embedder,
                    texts=texts,
                    labels=labels,
                    start_index=index,
                )
                for embedder in embedders
            ]
            for future in futures:
                future.result()
            if index > 0:
                elapsed = time.perf_counter() - start_time
                rate = (index + batch_size - (curr_si + 1)) / elapsed
                remaining_sec = (total_data - index) / rate
                eta_min = int(remaining_sec // 60)
                eta_sec = int(remaining_sec % 60)
                common.utils.loading_bar(
                    (index / total_data) * 100.0,
                    f"ETA {eta_min}m{eta_sec}s | Loading: ",
                )

    common.utils.loading_bar(100.0, "Done: ")
    print()
    print("Finish creating embeddings!")
