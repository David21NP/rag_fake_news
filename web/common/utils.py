import os
from pathlib import Path

import pandas as pd

from datasets import load_dataset

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets")
DATASET_TRAIN_PATH = os.path.join(DATASET_PATH, "news_dataset_train.df")
DATASET_TEST_PATH = os.path.join(DATASET_PATH, "news_dataset_test.df")
DATASET_VALIDATE_PATH = os.path.join(DATASET_PATH, "news_dataset_validate.df")


def get_df_train():
    if os.path.exists(DATASET_TRAIN_PATH):
        df_news = pd.read_feather(DATASET_TRAIN_PATH)
    else:
        dataset = load_dataset("GonzaloA/fake_news")
        df_news = pd.DataFrame(dataset["train"])
        Path(DATASET_TRAIN_PATH).parent.mkdir(parents=True, exist_ok=True)
        df_news.to_feather(DATASET_TRAIN_PATH)
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
        Path(DATASET_TEST_PATH).parent.mkdir(parents=True, exist_ok=True)
        df_news.to_feather(DATASET_TEST_PATH)
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
        Path(DATASET_VALIDATE_PATH).parent.mkdir(parents=True, exist_ok=True)
        df_news.to_feather(DATASET_VALIDATE_PATH)
    df_news["content"] = (
        df_news["title"].fillna("") + " " + df_news["text"].fillna("")
    )
    return df_news
