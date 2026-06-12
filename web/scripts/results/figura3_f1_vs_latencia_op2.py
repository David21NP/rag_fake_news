import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = Path(__file__).parent.parent.parent / "results"
RESULTS_FILE_PATH = RESULTS_PATH / "oe4_ablation.csv"

# # ── Datos ─────────────────────────────────────────────────────────────────────
# data = [
#     (1,  "EMB-A", "Completa",   1,  0.0, 0.640, 49.3),
#     (2,  "EMB-A", "Completa",   3,  0.0, 0.681, 94.9),
#     (3,  "EMB-A", "Completa",   5,  0.0, 0.646, 143.4),
#     (4,  "EMB-A", "Completa",   10, 0.0, 0.618, 226.7),
#     (5,  "EMB-A", "Completa",   1,  0.3, 0.640, 48.7),
#     (6,  "EMB-A", "Completa",   3,  0.3, 0.667, 94.6),
#     (7,  "EMB-A", "Completa",   5,  0.3, 0.639, 144.8),
#     (8,  "EMB-A", "Completa",   10, 0.3, 0.638, 232.4),
#     (9,  "EMB-A", "Sub-chunks", 1,  0.0, 0.681, 37.3),
#     (10, "EMB-A", "Sub-chunks", 3,  0.0, 0.652, 55.2),
#     (11, "EMB-A", "Sub-chunks", 5,  0.0, 0.637, 73.6),
#     (12, "EMB-A", "Sub-chunks", 10, 0.0, 0.653, 123.5),
#     (13, "EMB-A", "Sub-chunks", 1,  0.3, 0.674, 36.4),
#     (14, "EMB-A", "Sub-chunks", 3,  0.3, 0.659, 55.3),
#     (15, "EMB-A", "Sub-chunks", 5,  0.3, 0.630, 75.5),
#     (16, "EMB-A", "Sub-chunks", 10, 0.3, 0.660, 129.0),
#     (17, "EMB-B", "Completa",   1,  0.0, 0.722, 51.2),
#     (18, "EMB-B", "Completa",   3,  0.0, 0.667, 103.8),
#     (19, "EMB-B", "Completa",   5,  0.0, 0.667, 157.7),
#     (20, "EMB-B", "Completa",   10, 0.0, 0.667, 243.6),
#     (21, "EMB-B", "Completa",   1,  0.3, 0.717, 51.4),
#     (22, "EMB-B", "Completa",   3,  0.3, 0.667, 104.6),
#     (23, "EMB-B", "Completa",   5,  0.3, 0.667, 155.7),
#     (24, "EMB-B", "Completa",   10, 0.3, 0.679, 243.1),
#     (26, "EMB-B", "Sub-chunks", 3,  0.0, 0.625, 58.1),
#     (27, "EMB-B", "Sub-chunks", 5,  0.0, 0.624, 83.2),
#     (28, "EMB-B", "Sub-chunks", 10, 0.0, 0.602, 128.8),
#     (29, "EMB-B", "Sub-chunks", 1,  0.3, 0.649, 33.1),
#     (30, "EMB-B", "Sub-chunks", 3,  0.3, 0.625, 52.1),
#     (31, "EMB-B", "Sub-chunks", 5,  0.3, 0.624, 70.3),
#     (32, "EMB-B", "Sub-chunks", 10, 0.3, 0.617, 130.1),
# ]

# PANDAS_COLUMNS = [
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


def main():
    df = pd.read_csv(RESULTS_FILE_PATH)

    df.rename(columns={'latency_mean_sec': 'latency'}, inplace=True)

    # ── Mapeo visual ──────────────────────────────────────────────────────────────
    # Forma del marcador: círculo = Completa, cuadrado = Sub-chunks
    # Color: azul = EMB-A, naranja = EMB-B
    # Tamaño: proporcional al top-k

    def marker_style(row):
        return "o" if row["chunking"] == "CHUNK-A" else "s"

    color_map = {"EMB-A": "#5aade0", "EMB-B": "#e76f51"}
    size_map = {1: 60, 3: 100, 5: 140, 10: 200}

    fig, ax = plt.subplots(figsize=(11, 7))

    # Dibujar puntos normales
    for _, row in df.iterrows():
        if row["config_id"] in [17, 2]:
            continue  # los destacados se dibujan al final
        ax.scatter(
            row["latency"],
            row["f1"],
            color=color_map[row["embedding"]],
            marker=marker_style(row),
            s=size_map[row["top_k"]],
            alpha=0.65,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )

    # Dibujar C2 (config principal) destacada
    c2 = df[df["config_id"] == 2].iloc[0]
    ax.scatter(
        c2["latency"],
        c2["f1"],
        color="#f0a500",
        marker="o",
        s=180,
        edgecolors="#333333",
        linewidths=1.5,
        zorder=4,
    )
    ax.annotate(
        "C2\n(principal)",
        xy=(c2["latency"], c2["f1"]),
        xytext=(c2["latency"] + 8, c2["f1"] - 0.012),
        fontsize=8.5,
        fontweight="bold",
        color="#b37400",
        arrowprops=dict(arrowstyle="-", color="#b37400", lw=1),
    )

    # Dibujar C17 (óptima) destacada
    c17 = df[df["config_id"] == 17].iloc[0]
    ax.scatter(
        c17["latency"],
        c17["f1"],
        color="#1a7abf",
        marker="*",
        s=350,
        edgecolors="#333333",
        linewidths=1.0,
        zorder=4,
    )
    ax.annotate(
        "C17 ★\n(óptima)",
        xy=(c17["latency"], c17["f1"]),
        xytext=(c17["latency"] + 8, c17["f1"] + 0.006),
        fontsize=8.5,
        fontweight="bold",
        color="#1a7abf",
        arrowprops=dict(arrowstyle="-", color="#1a7abf", lw=1),
    )

    # Zona de interés (F1 alto, latencia baja) — cuadrante ideal
    ax.axvspan(
        0,
        110,
        ymin=(0.68 - 0.58) / (0.75 - 0.58),
        # ymin=0.32,
        alpha=0.06,
        color="#2d6a4f",
        label="_nolegend_",
    )
    ax.text(
        22,
        0.736,
        "Zona preferida\n(F1 alto, latencia baja)",
        fontsize=7.5,
        color="#2d6a4f",
        alpha=0.7,
        style="italic",
    )

    # Ejes
    ax.set_xlabel(
        "Latencia media de inferencia (s/ejemplo) — Hetzner CX43", fontsize=11
    )
    ax.set_ylabel("F1-score", fontsize=11)
    # ax.set_title(
    #     "Figura 3. Trade-off F1 vs. latencia de inferencia por configuración (OE4)\n"
    #     "N=100 ejemplos por configuración. Infra: Hetzner CX43 (CPU pura, sin GPU).",
    #     fontsize=11,
    #     pad=12,
    # )
    ax.set_xlim(20, 275)
    ax.set_ylim(0.58, 0.78)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.set_axisbelow(True)

    # Leyenda
    legend_elements = [
        mpatches.Patch(color="#5aade0", label="EMB-A (mxbai-embed-large)"),
        mpatches.Patch(color="#e76f51", label="EMB-B (nomic-embed-text)"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#888888",
            markersize=8,
            label="Chunking: Noticia completa",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#888888",
            markersize=8,
            label="Chunking: Sub-chunks",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#888888",
            markersize=5,
            label="k=1",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#888888",
            markersize=7,
            label="k=3",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#888888",
            markersize=9,
            label="k=5",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#888888",
            markersize=11,
            label="k=10",
        ),
        mpatches.Patch(color="#1a7abf", label="C17 ★ — Configuración óptima"),
        mpatches.Patch(
            color="#f0a500", label="C2 — Configuración principal (OE3)"
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=7.5,
        framealpha=0.92,
        ncol=2,
        borderpad=0.8,
    )

    plt.tight_layout()
    plt.savefig(
        RESULTS_PATH / "figura3_f1_vs_latencia_opt2.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Guardada: {RESULTS_PATH / "figura3_f1_vs_latencia_opt2.png"}")


if __name__ == "__main__":
    main()
