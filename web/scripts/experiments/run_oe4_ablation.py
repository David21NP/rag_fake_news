import csv
import itertools
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import common.utils
from config import get_settings
from rag.utils import create_generator

RESULTS_PATH = Path(__file__).parent.parent.parent / "results"
RESULTS_FILE_PATH = RESULTS_PATH / "oe4_ablation.csv"

EXPERIMENT_CHUNK_TYPES: list[Literal["full", "sliding"]] = ["full", "sliding"]
EXPERIMENT_TOP_Ks = [1, 3, 5, 10]
EXPERIMENT_TEMPERATURES = [0.0, 0.3]
EXPERIMENT_THINK_OPTIONS = [False, True]
TEST_SUBSET_N_PER_CLASS = 250


class ExperimentConfigs(BaseModel):
    config_id: int
    model: str
    embedding: str
    chunk_type: Literal["full", "sliding"]
    chunking: Literal["CHUNK-A", "CHUNK-B"]
    top_k: int
    temperature: float
    thinking: bool


CSV_COLUMNS = [
    "config_id",
    "embedding",
    "chunking",
    "top_k",
    "temperature",
    "thinking",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "latency_mean_sec",
]


def get_test_subset() -> pd.DataFrame:
    df_test = common.utils.get_df_test()
    fake = df_test[df_test["label"] == 0].sample(
        n=TEST_SUBSET_N_PER_CLASS,
        random_state=42,
    )
    real = df_test[df_test["label"] == 1].sample(
        n=TEST_SUBSET_N_PER_CLASS,
        random_state=42,
    )
    return (
        pd.concat([fake, real])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


def run_oe4():
    settings = get_settings()
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    print("Loading test subset (500 samples: 250 fake + 250 real)...")
    df_subset = get_test_subset()
    data = df_subset.to_dict("records")
    print(f"Loaded: {len(data)} samples")

    models = [
        (settings.ollama_model_embedding_principal, "EMB-A"),
        (settings.ollama_model_embedding_ablation, "EMB-B"),
    ]

    # Build the full list of 64 configs up front
    configs = [
        ExperimentConfigs(
            config_id=i + 1,
            model=model,
            embedding=emb_label,
            chunk_type=chunk_type,
            chunking="CHUNK-A" if chunk_type == "full" else "CHUNK-B",
            top_k=top_k,
            temperature=temperature,
            thinking=think,
        )
        for i, (
            (model, emb_label),
            chunk_type,
            temperature,
            think,
            top_k,
        ) in enumerate(
            itertools.product(
                models,
                EXPERIMENT_CHUNK_TYPES,
                EXPERIMENT_TEMPERATURES,
                EXPERIMENT_THINK_OPTIONS,
                EXPERIMENT_TOP_Ks,
            )
        )
    ]

    # Resume support: skip already-completed configs
    done_ids: set[int] = set()
    if RESULTS_FILE_PATH.exists():
        with open(RESULTS_FILE_PATH, newline="") as f:
            for row in csv.DictReader(f):
                done_ids.add(int(row["config_id"]))
        print(f"Resuming: {len(done_ids)}/{len(configs)} configs already done")

    done_cfgs = 0
    first = True

    start_time = time.perf_counter()
    with open(RESULTS_FILE_PATH, "a" if done_ids else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not done_ids:
            writer.writeheader()

        for cfg in configs:
            if cfg.config_id in done_ids:
                done_cfgs += 1
                continue

            if first:
                eta_str = f"ETA ?m?s"
                first = False
            else:
                elapsed = time.perf_counter() - start_time
                rate = (cfg.config_id - done_cfgs) / elapsed
                remaining_sec = (len(configs) - cfg.config_id) / rate
                eta_min = int(remaining_sec // 60)
                eta_sec = int(remaining_sec % 60)
                eta_str = f"ETA {eta_min}m{eta_sec}s"

            common.utils.loading_bar(
                ((cfg.config_id) / len(configs)) * 100.0,
                (
                    f"[{cfg.config_id}/{len(configs)}] "
                    f"{cfg.embedding} {cfg.chunking} "
                    f"k={cfg.top_k} T={cfg.temperature} think={cfg.thinking}"
                    f" | {eta_str} | Loading: "
                ),
            )

            generator = create_generator(
                settings=settings,
                model=cfg.model,
                chunk_type=cfg.chunk_type,
            )

            y_true: list[int] = []
            y_pred: list[int] = []
            latencies: list[float] = []

            for i, row in enumerate(data):
                try:
                    t0 = time.perf_counter()
                    response = generator(
                        title=(
                            row["title"]
                            if isinstance(row["title"], str)
                            else None
                        ),
                        text=row["text"],
                        top_k=cfg.top_k,
                        think=cfg.thinking,
                        temperature=cfg.temperature,
                    )
                    latencies.append(time.perf_counter() - t0)
                    y_true.append(int(row["label"]))
                    # Dataset: label=0 → fake, label=1 → real
                    y_pred.append(0 if response.answer.label == "fake" else 1)
                except Exception as e:
                    print(f"  Row {i} error: {e}")

                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(data)}")

            if not y_true:
                print("  All rows failed, skipping config")
                continue

            writer.writerow(
                {
                    "config_id": cfg.config_id,
                    "embedding": cfg.embedding,
                    "chunking": cfg.chunking,
                    "top_k": cfg.top_k,
                    "temperature": cfg.temperature,
                    "thinking": cfg.thinking,
                    "accuracy": round(accuracy_score(y_true, y_pred), 4),
                    # pos_label=0 (fake) per project convention
                    "precision": round(
                        float(
                            precision_score(
                                y_true,
                                y_pred,
                                pos_label=0,
                                zero_division=0,
                            )
                        ),
                        4,
                    ),
                    "recall": round(
                        float(
                            recall_score(
                                y_true,
                                y_pred,
                                pos_label=0,
                                zero_division=0,
                            )
                        ),
                        4,
                    ),
                    "f1": round(
                        float(
                            f1_score(
                                y_true,
                                y_pred,
                                pos_label=0,
                                zero_division=0,
                            )
                        ),
                        4,
                    ),
                    "latency_mean_sec": round(
                        sum(latencies) / len(latencies),
                        4,
                    ),
                }
            )
            f.flush()

    print(f"\nDone. Results saved to {RESULTS_FILE_PATH}")


if __name__ == "__main__":
    run_oe4()
