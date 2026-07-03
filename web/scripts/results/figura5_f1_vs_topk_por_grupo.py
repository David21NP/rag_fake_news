from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# ── Estilo por combinación embedding/chunking ────────────────────────────────
LINE_STYLES = {
    ("EMB-A", "CHUNK-A"): {"color": "#5aade0", "marker": "o", "label": "EMB-A / Noticia completa"},
    ("EMB-A", "CHUNK-B"): {"color": "#8ecae6", "marker": "s", "label": "EMB-A / Sub-chunks"},
    ("EMB-B", "CHUNK-A"): {"color": "#e76f51", "marker": "o", "label": "EMB-B / Noticia completa"},
    ("EMB-B", "CHUNK-B"): {"color": "#f4a261", "marker": "s", "label": "EMB-B / Sub-chunks"},
}

ERROR_CONFIGS = {4, 8, 20, 24}  # k=10 con desbordamiento de contexto


def main():
    df = pd.read_csv(RESULTS_FILE_PATH)

    # Media de F1 sobre las dos temperaturas, agrupado por embedding/chunking/k
    grouped = (
        df.groupby(["embedding", "chunking", "top_k"])
        .agg(f1=("f1", "mean"), config_ids=("config_id", list))
        .reset_index()
        .sort_values("top_k")
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))
    # fig, ax = plt.subplots(figsize=(10, 9))

    for (emb, chunk), style in LINE_STYLES.items():
        subset = grouped[(grouped["embedding"] == emb) & (grouped["chunking"] == chunk)]
        subset = subset.sort_values("top_k")

        ax.plot(
            subset["top_k"],
            subset["f1"],
            color=style["color"],
            marker=style["marker"],
            markersize=8,
            linewidth=2,
            label=style["label"],
            zorder=3,
        )

        # Marcar puntos afectados por errores de contexto (k=10)
        for _, row in subset.iterrows():
            has_error = any(cid in ERROR_CONFIGS for cid in row["config_ids"])
            if has_error:
                ax.scatter(
                    row["top_k"],
                    row["f1"],
                    facecolors="none",
                    edgecolors="#cc0000",
                    s=180,
                    linewidths=1.8,
                    zorder=4,
                )

    # # Líneas de referencia baselines
    # ax.axhline(y=0.9242, color="#2d6a4f", linewidth=1.3, linestyle="--", alpha=0.6)
    # ax.text(9.7, 0.918, "NC-LBFV (F1=0,924)", fontsize=8, color="#2d6a4f", ha="right")
    # ax.axhline(y=0.8700, color="#52b788", linewidth=1.3, linestyle="--", alpha=0.6)
    # ax.text(9.7, 0.864, "Nezafat & Samet 2024 (F1=0,870)", fontsize=8, color="#52b788", ha="right")

    # Marcador de configuración óptima C17 (k=1, EMB-B/CHUNK-A)
    ax.annotate(
        "C17 ★ (óptima)\nF1=0,722",
        xy=(1, 0.722),
        xytext=(1.6, 0.74),
        fontsize=8.5,
        fontweight="bold",
        color="#1a7abf",
        arrowprops=dict(arrowstyle="-", color="#1a7abf", lw=1),
    )

    ax.set_xlabel("Documentos recuperados (top-k)", fontsize=11)
    ax.set_ylabel("F1-score (media sobre T=0,0 y T=0,3)", fontsize=11)
    # ax.set_title(
    #     "Figura 5. Evolución del F1-score en función de top-k por combinación embedding/chunking\n"
    #     "Círculos rojos: configuraciones con errores de desbordamiento de contexto (k=10). N=100 ejemplos por configuración.",
    #     fontsize=10.5,
    #     pad=12,
    # )
    ax.set_xticks([1, 3, 5, 10])
    ax.set_ylim(0.58, 0.78)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(
        RESULTS_PATH / "figura5_f1_vs_topk_por_grupo.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Guardada: {RESULTS_PATH / 'figura5_f1_vs_topk_por_grupo.png'}")


if __name__ == "__main__":
    main()
