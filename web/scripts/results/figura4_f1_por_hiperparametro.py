from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_PATH = Path(__file__).parent.parent.parent / "results"
RESULTS_FILE_PATH = RESULTS_PATH / "oe4_ablation.csv"

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

BAR_COLOR = "#5aade0"
HIGHLIGHT_COLOR = "#1a7abf"
EDGE_COLOR = "white"


def add_value_labels(ax, x, values, fmt="{:.3f}"):
    for xi, v in zip(x, values):
        ax.text(
            xi,
            v + 0.006,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#222222",
        )


def plot_axis(ax, labels, means, title, xlabel, highlight_idx=None):
    x = np.arange(len(labels))
    colors = [BAR_COLOR] * len(labels)
    if highlight_idx is not None:
        colors[highlight_idx] = HIGHLIGHT_COLOR

    ax.bar(x, means, color=colors, width=0.55, edgecolor=EDGE_COLOR, linewidth=0.5)
    add_value_labels(ax, x, means)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("F1-score medio", fontsize=9)
    ax.set_ylim(0.58, 0.74)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)


def main():
    df = pd.read_csv(RESULTS_FILE_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # ── Panel A: por embedding ──────────────────────────────────────────────
    emb_means = df.groupby("embedding")["f1"].mean()
    emb_labels = ["EMB-A\n(mxbai-embed-large)", "EMB-B\n(nomic-embed-text)"]
    plot_axis(
        axes[0, 0],
        emb_labels,
        [emb_means["EMB-A"], emb_means["EMB-B"]],
        "A. Efecto del modelo de embedding (§5.2.1)",
        "Modelo de embedding",
    )

    # ── Panel B: por chunking ────────────────────────────────────────────────
    chunk_means = df.groupby("chunking")["f1"].mean()
    chunk_labels = ["CHUNK-A\n(noticia completa)", "CHUNK-B\n(sub-chunks)"]
    plot_axis(
        axes[0, 1],
        chunk_labels,
        [chunk_means["CHUNK-A"], chunk_means["CHUNK-B"]],
        "B. Efecto de la estrategia de chunking (§5.2.2)",
        "Estrategia de chunking",
        highlight_idx=0,
    )

    # ── Panel C: por top-k ───────────────────────────────────────────────────
    k_means = df.groupby("top_k")["f1"].mean().sort_index()
    k_labels = [f"k={k}" for k in k_means.index]
    best_k_idx = int(np.argmax(k_means.values))
    plot_axis(
        axes[1, 0],
        k_labels,
        k_means.values,
        "C. Efecto del parámetro top-k (§5.2.3)",
        "Documentos recuperados (k)",
        highlight_idx=best_k_idx,
    )

    # ── Panel D: por temperatura ─────────────────────────────────────────────
    t_means = df.groupby("temperature")["f1"].mean().sort_index()
    t_labels = [f"T={t}" for t in t_means.index]
    plot_axis(
        axes[1, 1],
        t_labels,
        t_means.values,
        "D. Efecto de la temperatura (§5.2.4)",
        "Temperatura del LLM",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(
        RESULTS_PATH / "figura4_f1_por_hiperparametro.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Guardada: {RESULTS_PATH / 'figura4_f1_por_hiperparametro.png'}")


if __name__ == "__main__":
    main()
