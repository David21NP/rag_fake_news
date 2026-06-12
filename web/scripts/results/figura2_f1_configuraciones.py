import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_PATH = Path(__file__).parent.parent.parent / "results"
RESULTS_FILE_PATH = RESULTS_PATH / "oe4_ablation.csv"

# ── Datos ────────────────────────────────────────────────────────────────────
# Sustituir con la ruta real al CSV si se prefiere cargar desde archivo
# df = pd.read_csv("oe4_ablation.csv")

# data = [
#     (1, "EMB-A", "Completa", 1, 0.0, 0.640),
#     (2, "EMB-A", "Completa", 3, 0.0, 0.681),  # config principal
#     (3, "EMB-A", "Completa", 5, 0.0, 0.646),
#     (4, "EMB-A", "Completa", 10, 0.0, 0.618),  # errores contexto
#     (5, "EMB-A", "Completa", 1, 0.3, 0.640),
#     (6, "EMB-A", "Completa", 3, 0.3, 0.667),
#     (7, "EMB-A", "Completa", 5, 0.3, 0.639),
#     (8, "EMB-A", "Completa", 10, 0.3, 0.638),  # errores contexto
#     (9, "EMB-A", "Sub-chunks", 1, 0.0, 0.681),
#     (10, "EMB-A", "Sub-chunks", 3, 0.0, 0.652),
#     (11, "EMB-A", "Sub-chunks", 5, 0.0, 0.637),
#     (12, "EMB-A", "Sub-chunks", 10, 0.0, 0.653),
#     (13, "EMB-A", "Sub-chunks", 1, 0.3, 0.674),
#     (14, "EMB-A", "Sub-chunks", 3, 0.3, 0.659),
#     (15, "EMB-A", "Sub-chunks", 5, 0.3, 0.630),
#     (16, "EMB-A", "Sub-chunks", 10, 0.3, 0.660),
#     (17, "EMB-B", "Completa", 1, 0.0, 0.722),  # óptima
#     (18, "EMB-B", "Completa", 3, 0.0, 0.667),
#     (19, "EMB-B", "Completa", 5, 0.0, 0.667),
#     (20, "EMB-B", "Completa", 10, 0.0, 0.667),  # errores contexto
#     (21, "EMB-B", "Completa", 1, 0.3, 0.717),
#     (22, "EMB-B", "Completa", 3, 0.3, 0.667),
#     (23, "EMB-B", "Completa", 5, 0.3, 0.667),
#     (24, "EMB-B", "Completa", 10, 0.3, 0.679),  # errores contexto
#     # C25 omitida (no completada)
#     (26, "EMB-B", "Sub-chunks", 3, 0.0, 0.625),
#     (27, "EMB-B", "Sub-chunks", 5, 0.0, 0.624),
#     (28, "EMB-B", "Sub-chunks", 10, 0.0, 0.602),
#     (29, "EMB-B", "Sub-chunks", 1, 0.3, 0.649),
#     (30, "EMB-B", "Sub-chunks", 3, 0.3, 0.625),
#     (31, "EMB-B", "Sub-chunks", 5, 0.3, 0.624),
#     (32, "EMB-B", "Sub-chunks", 10, 0.3, 0.617),
# ]

# CSV_COLUMNS = [
#     "config_id",
#     "embedding",
#     "chunking",
#     "top_k",
#     "temperature",
#     "f1",
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


# ── Colores por grupo ─────────────────────────────────────────────────────────
def get_color(row):
    if row["config_id"] == 17:
        return "#1a7abf"  # óptima — azul oscuro
    elif row["config_id"] == 2:
        return "#f0a500"  # config principal — naranja
    elif row["config_id"] in [4, 8, 20, 24]:
        return "#cccccc"  # errores contexto — gris
    elif row["embedding"] == "EMB-A" and row["chunking"] == "CHUNK-A":
        return "#5aade0"
    elif row["embedding"] == "EMB-A" and row["chunking"] == "CHUNK-B":
        return "#8ecae6"
    elif row["embedding"] == "EMB-B" and row["chunking"] == "CHUNK-A":
        return "#e76f51"
    else:
        return "#f4a261"


def main():
    df = pd.read_csv(RESULTS_FILE_PATH)

    df["color"] = df.apply(get_color, axis=1)
    df = df.sort_values("config_id").reset_index(drop=True)

    # ── Figura ────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))

    x = np.arange(len(df))
    bars = ax.bar(
        x,
        df["f1"],
        color=df["color"],
        width=0.7,
        edgecolor="white",
        linewidth=0.5,
    )

    # Línea de referencia — F1 NC-LBFV y Nezafat
    ax.axhline(
        y=0.9242,
        color="#2d6a4f",
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label="NC-LBFV (F1=0,924)",
    )
    ax.axhline(
        y=0.8700,
        color="#52b788",
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label="Nezafat & Samet 2024 (F1=0,870)",
    )

    # Etiquetas valor encima de barras destacadas
    for i, row in df.iterrows():
        if row["config_id"] in [17, 2, 28]:
            ax.text(
                x[i],
                row["f1"] + 0.004,
                f"{row['f1']:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#222222",
            )

    # Etiquetas eje X
    labels = [f"C{int(r.config_id)}" for _, r in df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")

    # Ejes y título
    ax.set_ylim(0.55, 0.97)
    ax.set_ylabel("F1-score", fontsize=11)
    ax.set_xlabel("Configuración", fontsize=11)
    # ax.set_title(
    #     "Figura 2. F1-score por configuración — análisis de sensibilidad paramétrica (OE4)\n"
    #     "N=100 ejemplos por configuración.",
    #     fontsize=11,
    #     pad=12,
    # )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)

    # Leyenda actual
    handles, _labels = ax.get_legend_handles_labels()
    # Leyenda
    legend_patches = [
        mpatches.Patch(
            color="#1a7abf",
            label="C17 — Configuración óptima (EMB-B/Completa/k=1)",
        ),
        mpatches.Patch(
            color="#f0a500",
            label="C2 — Configuración principal del sistema (OE3)",
        ),
        mpatches.Patch(color="#5aade0", label="EMB-A / Noticia completa"),
        mpatches.Patch(color="#8ecae6", label="EMB-A / Sub-chunks"),
        mpatches.Patch(color="#e76f51", label="EMB-B / Noticia completa"),
        mpatches.Patch(color="#f4a261", label="EMB-B / Sub-chunks"),
        mpatches.Patch(color="#cccccc", label="Errores de contexto (k=10)"),
        *handles,
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        ncol=2,
        borderpad=0.8,
    )

    plt.tight_layout()
    plt.savefig(
        RESULTS_PATH / "figura2_f1_configuraciones.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Guardada: {RESULTS_PATH / "figura2_f1_configuraciones.png"}")


if __name__ == "__main__":
    main()
