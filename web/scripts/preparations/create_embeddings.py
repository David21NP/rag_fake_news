import itertools
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
    for index, df_batch in common.utils.iter_batches(
        df_train,
        batch_size=batch_size,
    ):
        texts = (
            df_batch["title"].fillna("") + "\n\n" + df_batch["text"]
        ).tolist()
        labels = df_batch["label"].tolist()
        for embedder in embedders:
            embedder(texts=texts, labels=labels)
        if (index // batch_size) % 10 == 0:
            common.utils.loading_bar(
                ((index + 1) / total_data) * 100.0,
                "Loading: ",
            )

    common.utils.loading_bar(100.0, "Done: ")
    print()
    print("Finish creating embeddings!")
