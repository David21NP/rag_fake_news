import os
from collections.abc import Generator
from pathlib import Path

import pandas as pd

from datasets import DatasetDict, load_dataset

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets")
DATASET_TRAIN_PATH = os.path.join(DATASET_PATH, "news_dataset_train.df")
DATASET_TEST_PATH = os.path.join(DATASET_PATH, "news_dataset_test.df")
DATASET_VALIDATE_PATH = os.path.join(DATASET_PATH, "news_dataset_validate.df")


def save_splits(dataset: DatasetDict):
    df_train = pd.DataFrame(dataset["train"])
    df_test = pd.DataFrame(dataset["test"])
    df_validate = pd.DataFrame(dataset["validation"])
    Path(DATASET_TRAIN_PATH).parent.mkdir(parents=True, exist_ok=True)
    df_train.to_feather(DATASET_TRAIN_PATH)
    df_test.to_feather(DATASET_TEST_PATH)
    df_validate.to_feather(DATASET_VALIDATE_PATH)


def get_df_train():
    if os.path.exists(DATASET_TRAIN_PATH):
        df_news = pd.read_feather(DATASET_TRAIN_PATH)
    else:
        dataset = load_dataset("GonzaloA/fake_news")
        df_news = pd.DataFrame(dataset["train"])
        save_splits(dataset)
    df_news["content"] = (
        df_news["title"].fillna("") + " " + df_news["text"].fillna("")
    )
    return df_news


def get_df_test():
    if os.path.exists(DATASET_TEST_PATH):
        df_news = pd.read_feather(DATASET_TEST_PATH)
    else:
        dataset = load_dataset("GonzaloA/fake_news")
        df_news = pd.DataFrame(dataset["test"])
        save_splits(dataset)
    df_news["content"] = (
        df_news["title"].fillna("") + " " + df_news["text"].fillna("")
    )
    return df_news


def get_df_validate():
    if os.path.exists(DATASET_VALIDATE_PATH):
        df_news = pd.read_feather(DATASET_VALIDATE_PATH)
    else:
        dataset = load_dataset("GonzaloA/fake_news")
        df_news = pd.DataFrame(dataset["validation"])
        save_splits(dataset)
    df_news["content"] = (
        df_news["title"].fillna("") + " " + df_news["text"].fillna("")
    )
    return df_news


def loading_bar(progress: float, prefix: str = "", size: int = 50):
    percent = progress / 100.0
    completed_length = int(size * percent)
    bar = "#" * completed_length + "-" * (size - completed_length)
    # if sys.stdout.isatty():
    #     sys.stdout.write(f"\r{prefix} |{bar}| {progress:.1f}%")
    #     sys.stdout.flush()
    # else:
    print(f"{prefix} |{bar}| {progress:.1f}%", flush=True)


def iter_batches(
    df: pd.DataFrame,
    batch_size: int,
) -> Generator[tuple[int, pd.DataFrame], None, None]:
    for start in range(0, len(df), batch_size):
        yield start, df.iloc[start : start + batch_size]
