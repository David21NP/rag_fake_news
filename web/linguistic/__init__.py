import time

from pandas import DataFrame

import common.utils
import linguistic.utils
from schemas import Metrics


def run_NC_LBFV(df_test_subset: DataFrame) -> tuple[Metrics, float]:
    df_train = common.utils.get_df_train()
    df_test = df_test_subset.copy()
    df_test["content"] = (
        df_test["title"].fillna("") + " " + df_test["text"].fillna("")
    )

    x_train = df_train["content"].apply(linguistic.utils.clean_text)
    x_test = df_test["content"].apply(linguistic.utils.clean_text)

    # Fit transformers on full train set
    tfidf_train = linguistic.utils.extract_tf_idf_fit_transform(x_train)
    emb_train = linguistic.utils.extract_embeddings_fit_transform(x_train)
    emo_train = linguistic.utils.extract_emotions(x_train)
    X_train = linguistic.utils.build_lbfv(tfidf_train, emb_train, emo_train)
    y_train = df_train["label"].values

    rf = linguistic.utils.train_classifier(X_train, y_train)

    # Time test inference only (feature extraction + predict, not training)
    t0 = time.perf_counter()
    tfidf_test = linguistic.utils.extract_tf_idf_transform(x_test)
    emb_test = linguistic.utils.extract_embeddings_transform(x_test)
    emo_test = linguistic.utils.extract_emotions(x_test)
    X_test = linguistic.utils.build_lbfv(tfidf_test, emb_test, emo_test)
    y_pred = rf.predict(X_test)
    latency_mean_sec = (time.perf_counter() - t0) / len(x_test)

    metrics = linguistic.utils.compute_metrics(df_test["label"].values, y_pred)
    return metrics, latency_mean_sec
