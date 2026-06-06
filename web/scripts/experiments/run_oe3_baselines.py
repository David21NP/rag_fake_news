import csv
import time
from pathlib import Path
from typing import TypedDict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import common.utils
from config import get_settings
from linguistic import run_NC_LBFV
from rag.utils import create_generator

RESULTS_PATH = Path(__file__).parent.parent.parent / "results"
RESULTS_FILE_PATH = RESULTS_PATH / "oe3_baselines.csv"

TEST_SUBSET_N_PER_CLASS = 500

CSV_COLUMNS = [
    "system",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "latency_mean_sec",
]


class ResultDict(TypedDict):
    system: str
    accuracy: float
    precision: float
    recall: float | str
    f1: float
    latency_mean_sec: float | str


def get_test_subset() -> pd.DataFrame:
    df_test = common.utils.get_df_test()
    fake = df_test[df_test["label"] == 0].sample(
        n=TEST_SUBSET_N_PER_CLASS, random_state=42
    )
    real = df_test[df_test["label"] == 1].sample(
        n=TEST_SUBSET_N_PER_CLASS, random_state=42
    )
    return (
        pd.concat([fake, real])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


def run_oe3():
    settings = get_settings()
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    print("Loading test subset (1000 samples: 500 fake + 500 real)...")
    df_subset = get_test_subset()
    print(f"Loaded: {len(df_subset)} samples")

    rows: list[ResultDict] = []

    # --- 1. NC-LBFV ---
    print("\nRunning NC-LBFV (Hashemi et al.)...")
    nc_metrics, nc_latency = run_NC_LBFV(df_subset)
    rows.append(
        {
            "system": "NC-LBFV",
            "accuracy": round(nc_metrics["accuracy"], 4),
            "precision": round(nc_metrics["precision"], 4),
            "recall": round(nc_metrics["recall"], 4),
            "f1": round(nc_metrics["f1"], 4),
            "latency_mean_sec": round(nc_latency, 6),
        }
    )
    print(
        f"NC-LBFV done. F1={nc_metrics['f1']:.4f}, latency={nc_latency:.6f}s/sample"
    )

    # --- 2. RAG — configuración principal ---
    # EMB-A (mxbai-embed-large) / CHUNK-A (full) / k=3 / temperature=0.0 / thinking=off
    print("\nRunning RAG (principal config)...")
    generator = create_generator(
        settings=settings,
        model=settings.ollama_model_embedding_principal,
        chunk_type="full",
    )
    data = df_subset.to_dict("records")
    y_true: list[int] = []
    y_pred: list[int] = []
    latencies: list[float] = []

    for i, row in enumerate(data):
        try:
            t0 = time.perf_counter()
            response = generator(
                title=row["title"] or None,
                text=row["text"],
                top_k=3,
                think=False,
                temperature=0.0,
            )
            latencies.append(time.perf_counter() - t0)
            y_true.append(int(row["label"]))
            # Dataset: label=0 → fake, label=1 → real
            y_pred.append(0 if response.answer.label == "fake" else 1)
        except Exception as e:
            print(f"  Row {i} error: {e}")

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(data)}")

    rag_latency = sum(latencies) / len(latencies) if latencies else 0.0
    rows.append(
        {
            "system": "RAG",
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            # pos_label=0 (fake) per project convention
            "precision": round(
                float(
                    precision_score(
                        y_true, y_pred, pos_label=0, zero_division=0
                    )
                ),
                4,
            ),
            "recall": round(
                float(
                    recall_score(y_true, y_pred, pos_label=0, zero_division=0)
                ),
                4,
            ),
            "f1": round(
                float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
                4,
            ),
            "latency_mean_sec": round(rag_latency, 4),
        }
    )
    print(
        f"RAG done. F1={rows[-1]['f1']:.4f}, latency={rag_latency:.4f}s/sample"
    )

    # --- 3. Nezafat & Samet (2024) — resultados publicados, dataset diferente (ISOT) ---
    # Accuracy 88%, Precision 94%, F1 87%. Recall y latencia no reportados.
    rows.append(
        {
            "system": "Nezafat & Samet (2024)*",
            "accuracy": 0.88,
            "precision": 0.94,
            "recall": "N/A",
            "f1": 0.87,
            "latency_mean_sec": "N/A",
        }
    )

    with open(RESULTS_FILE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results saved to {RESULTS_FILE_PATH}")
    print(
        "* Nezafat & Samet evaluated on ISOT dataset, not GonzaloA/fake_news"
    )


if __name__ == "__main__":
    run_oe3()
