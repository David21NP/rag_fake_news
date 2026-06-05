import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import common.utils
from config import get_settings
from rag.utils import create_embedder

EXPERIMENT_CHUNK_TYPES: list[Literal["full", "sliding"]] = ["full", "sliding"]


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
    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, df_batch in common.utils.iter_batches(
            df_train,
            batch_size=batch_size,
        ):
            if index == 0:
                common.utils.loading_bar(
                    (index / total_data) * 100.0,
                    f"ETA ?m?s | Loading: ",
                )
            texts = (
                df_batch["title"].fillna("") + "\n\n" + df_batch["text"]
            ).tolist()
            labels = df_batch["label"].tolist()
            futures = [
                executor.submit(embedder, texts=texts, labels=labels)
                for embedder in embedders
            ]
            for future in futures:
                future.result()
            if (index // batch_size) % 10 == 0 and index > 0:
                elapsed = time.perf_counter() - start_time
                rate = index / elapsed
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
