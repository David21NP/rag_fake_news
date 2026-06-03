from typing import Literal

from config import get_settings
from rag.utils import create_embedder, create_generator

from ...types import Generator

EXPERIMENT_COMBINATIONS_EMBEDDINGS: tuple[
    tuple[str, Literal["full"] | Literal["sliding"]], ...
] = (
    ("mxbai-embed-large", "full"),
    ("mxbai-embed-large", "sliding"),
    ("nomic-embed-text", "full"),
    ("nomic-embed-text", "sliding"),
)

EXPERIMENT_TOP_Ks = [1, 3, 5, 10]

EXPERIMENT_COMBINATIONS = (
    ("mxbai-embed-large", "full", 1),
    ("mxbai-embed-large", "full", 3),
    ("mxbai-embed-large", "full", 5),
    ("mxbai-embed-large", "full", 10),
    ("mxbai-embed-large", "sliding", 1),
    ("mxbai-embed-large", "sliding", 3),
    ("mxbai-embed-large", "sliding", 5),
    ("mxbai-embed-large", "sliding", 10),
    ("nomic-embed-text", "full", 1),
    ("nomic-embed-text", "full", 3),
    ("nomic-embed-text", "full", 5),
    ("nomic-embed-text", "full", 10),
    ("nomic-embed-text", "sliding", 1),
    ("nomic-embed-text", "sliding", 3),
    ("nomic-embed-text", "sliding", 5),
    ("nomic-embed-text", "sliding", 10),
)


def create_embeddings_for_combinations():
    experiments: list[tuple[Generator, int]] = []
    for model, chunk_type in EXPERIMENT_COMBINATIONS_EMBEDDINGS:
        embedder = create_embedder(
            settings=get_settings(),
            model=model,
            chunk_type=chunk_type,
        )
        generator = create_generator(
            settings=get_settings(),
            model=model,
            chunk_type=chunk_type,
        )
        for top_k in EXPERIMENT_TOP_Ks:
            experiments.append((generator, top_k))
