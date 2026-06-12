import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_PATH = Path(__file__).parent.parent.parent / "results"
RESULTS_FILE_PATH = RESULTS_PATH / "oe4_ablation.csv"

# # ── Datos completos OE4 ───────────────────────────────────────────────────────
# # Puedes sustituir esto por: df = pd.read_csv("oe4_ablation.csv")
# data = [
#     (1,  "EMB-A", "mxbai-embed-large", "Completa",   1,  0.0, 0.640, 0.640, 0.640, 0.640, 49.3),
#     (2,  "EMB-A", "mxbai-embed-large", "Completa",   3,  0.0, 0.698, 0.705, 0.660, 0.681, 94.9),
#     (3,  "EMB-A", "mxbai-embed-large", "Completa",   5,  0.0, 0.660, 0.674, 0.620, 0.646, 143.4),
#     (4,  "EMB-A", "mxbai-embed-large", "Completa",   10, 0.0, 0.644, 0.700, 0.553, 0.618, 226.7),
#     (5,  "EMB-A", "mxbai-embed-large", "Completa",   1,  0.3, 0.640, 0.640, 0.640, 0.640, 48.7),
#     (6,  "EMB-A", "mxbai-embed-large", "Completa",   3,  0.3, 0.680, 0.696, 0.640, 0.667, 94.6),
#     (7,  "EMB-A", "mxbai-embed-large", "Completa",   5,  0.3, 0.650, 0.660, 0.620, 0.639, 144.8),
#     (8,  "EMB-A", "mxbai-embed-large", "Completa",   10, 0.3, 0.658, 0.710, 0.579, 0.638, 232.4),
#     (9,  "EMB-A", "mxbai-embed-large", "Sub-chunks", 1,  0.0, 0.700, 0.727, 0.640, 0.681, 37.3),
#     (10, "EMB-A", "mxbai-embed-large", "Sub-chunks", 3,  0.0, 0.680, 0.714, 0.600, 0.652, 55.2),
#     (11, "EMB-A", "mxbai-embed-large", "Sub-chunks", 5,  0.0, 0.670, 0.707, 0.580, 0.637, 73.6),
#     (12, "EMB-A", "mxbai-embed-large", "Sub-chunks", 10, 0.0, 0.660, 0.667, 0.640, 0.653, 123.5),
#     (13, "EMB-A", "mxbai-embed-large", "Sub-chunks", 1,  0.3, 0.690, 0.711, 0.640, 0.674, 36.4),
#     (14, "EMB-A", "mxbai-embed-large", "Sub-chunks", 3,  0.3, 0.690, 0.732, 0.600, 0.659, 55.3),
#     (15, "EMB-A", "mxbai-embed-large", "Sub-chunks", 5,  0.3, 0.660, 0.691, 0.580, 0.630, 75.5),
#     (16, "EMB-A", "mxbai-embed-large", "Sub-chunks", 10, 0.3, 0.670, 0.681, 0.640, 0.660, 129.0),
#     (17, "EMB-B", "nomic-embed-text",  "Completa",   1,  0.0, 0.700, 0.672, 0.780, 0.722, 51.2),
#     (18, "EMB-B", "nomic-embed-text",  "Completa",   3,  0.0, 0.660, 0.654, 0.680, 0.667, 103.8),
#     (19, "EMB-B", "nomic-embed-text",  "Completa",   5,  0.0, 0.660, 0.654, 0.680, 0.667, 157.7),
#     (20, "EMB-B", "nomic-embed-text",  "Completa",   10, 0.0, 0.714, 0.720, 0.621, 0.667, 243.6),
#     (21, "EMB-B", "nomic-embed-text",  "Completa",   1,  0.3, 0.700, 0.679, 0.760, 0.717, 51.4),
#     (22, "EMB-B", "nomic-embed-text",  "Completa",   3,  0.3, 0.660, 0.654, 0.680, 0.667, 104.6),
#     (23, "EMB-B", "nomic-embed-text",  "Completa",   5,  0.3, 0.660, 0.654, 0.680, 0.667, 155.7),
#     (24, "EMB-B", "nomic-embed-text",  "Completa",   10, 0.3, 0.730, 0.750, 0.621, 0.679, 243.1),
#     (25, "EMB-B", "nomic-embed-text",  "Sub-chunks", 1,  0.0, None,  None,  None,  None,  None),
#     (26, "EMB-B", "nomic-embed-text",  "Sub-chunks", 3,  0.0, 0.640, 0.652, 0.600, 0.625, 58.1),
#     (27, "EMB-B", "nomic-embed-text",  "Sub-chunks", 5,  0.0, 0.650, 0.674, 0.580, 0.624, 83.2),
#     (28, "EMB-B", "nomic-embed-text",  "Sub-chunks", 10, 0.0, 0.630, 0.651, 0.560, 0.602, 128.8),
#     (29, "EMB-B", "nomic-embed-text",  "Sub-chunks", 1,  0.3, 0.610, 0.590, 0.720, 0.649, 33.1),
#     (30, "EMB-B", "nomic-embed-text",  "Sub-chunks", 3,  0.3, 0.640, 0.652, 0.600, 0.625, 52.1),
#     (31, "EMB-B", "nomic-embed-text",  "Sub-chunks", 5,  0.3, 0.650, 0.674, 0.580, 0.624, 70.3),
#     (32, "EMB-B", "nomic-embed-text",  "Sub-chunks", 10, 0.3, 0.640, 0.659, 0.580, 0.617, 130.1),
# ]

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


cols = [
    "C",
    "emb_id",
    "embedding",
    "chunking",
    "k",
    "T",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "latencia",
]

cols_select = [
    "config_id",
    "emb_id",
    "embedding",
    "chunking",
    "k",
    "T",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "latency_mean_sec",
]


def get_embedding_name(row):
    if row["emb_id"] == "EMB-A":
        return "mxbai-embed-large"
    else:
        return "nomic-embed-text"


def main():
    df = pd.read_csv(RESULTS_FILE_PATH)

    df.rename(
        columns={
            "embedding": "emb_id",
            "top_k": "k",
            "temperature": "T",
        },
        inplace=True,
    )
    df["embedding"] = df.apply(get_embedding_name, axis=1)
    # Guardar CSV completo
    # df[cols_select].to_csv(
    #     RESULTS_PATH / "anexo_c_tablas_oe4.csv",
    #     # columns=cols,
    #     index=False,
    #     float_format="%.4f",
    # )
    # print(f"Guardado: {RESULTS_PATH / "anexo_c_tablas_oe4.csv"}")

    table_selections = [
        "emb_id",
        "chunking",
        "k",
        "T",
    ]
    for ts in table_selections:
        summary = (
            df.dropna(subset=["f1"])
            .groupby(ts)[
                ["accuracy", "precision", "recall", "f1", "latency_mean_sec"]
            ]
            .mean()
            .round(4)
        )
        if ts == "emb_id":
            summary.index = [
                "EMB-A (mxbai-embed-large)",
                "EMB-B (nomic-embed-text)",
            ]
        elif ts == "T":
            summary.index = ["T=0.0", "T=0.3"]

        name = f"anexo_c_tabla_{ts}_oe4.csv"
        filepath = RESULTS_PATH / name
        summary.to_csv(filepath)
        print(f"Guardada: {filepath}")


if __name__ == "__main__":
    main()
