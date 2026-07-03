import csv
from pathlib import Path

from pydantic import BaseModel

RES_PATH = Path(__file__).parent.parent.parent / "results" / "oe4_ablation.csv"
RES_ADDED_PATH = (
    Path(__file__).parent.parent.parent
    / "results"
    / "oe4_ablation_with_specificity.csv"
)


class OE4Row(BaseModel):
    config_id: int
    embedding: str
    chunking: str
    top_k: int
    temperature: float
    thinking: bool
    accuracy: float
    precision: float
    specificity: float | None = None
    recall: float
    f1: float
    latency_mean_sec: float


TOTAL_MUESTRAS = 100
POSITIVOS_REALES = 50  # Cantidad de "real"
NEGATIVOS_REALES = 50  # Cantidad de "fake"


NEW_HEADER = [
    "config_id",
    "embedding",
    "chunking",
    "top_k",
    "temperature",
    "thinking",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "latency_mean_sec",
]


def main():
    data: list[OE4Row] = []
    with open(RES_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(OE4Row.model_validate(row))

    with open(RES_ADDED_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NEW_HEADER)
        writer.writeheader()
        for row in data:
            TP = int(row.recall * POSITIVOS_REALES)
            TN = int((row.accuracy * TOTAL_MUESTRAS) - TP)
            FP = NEGATIVOS_REALES - TN
            FN = POSITIVOS_REALES - TP

            row.specificity = TN / NEGATIVOS_REALES
            # print(f"[ {TP}, {FP} ]")
            # print(f"[ {FN}, {TN} ]")
            writer.writerow(row.model_dump())

if __name__ == "__main__":
    main()
