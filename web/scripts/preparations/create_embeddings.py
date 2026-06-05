import itertools
from typing import Literal

import common.utils
from config import get_settings
from rag.utils import create_embedder

EXPERIMENT_CHUNK_TYPES: list[Literal["full", "sliding"]] = ["full", "sliding"]


def add_embeddings():
    settings = get_settings()
    print("Loading train dataset ...")
    df_train = common.utils.get_df_train()
    data = df_train.to_dict("records")
    print(f"Loaded: {len(data)} samples")
    models = [
        settings.ollama_model_embedding_principal,
        settings.ollama_model_embedding_ablation,
    ]
    embedders = [
        create_embedder(settings=settings, model=model, chunk_type=chunk_type)
        for model, chunk_type in itertools.product(
            models, EXPERIMENT_CHUNK_TYPES
        )
    ]
    print("Creating embeddings ...")
    for row in data:
        for embedder in embedders:
            embedder(
                title=row["title"] or None,
                text=row["text"],
                label=row["label"],
            )
    print("Finish creating embeddings!")
