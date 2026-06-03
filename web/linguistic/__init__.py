from pandas import DataFrame

import common.utils
import linguistic.utils


def run_NC_LBFV(df_test_subset: DataFrame):
    df_train = common.utils.get_df_train()
    df_test_subset["content"] = (
        df_test_subset["title"].fillna("")
        + " "
        + df_test_subset["text"].fillna("")
    )

    x_train = df_train["content"].apply(linguistic.utils.clean_text)
    x_test = df_test_subset["content"].apply(linguistic.utils.clean_text)

    tfidf_train = linguistic.utils.extract_tf_idf_fit_transform(x_train)
    tfidf_test = linguistic.utils.extract_tf_idf_transform(x_test)
    emb_train = linguistic.utils.extract_embeddings_fit_transform(x_train)
    emb_test = linguistic.utils.extract_embeddings_transform(x_test)
    emo_train = linguistic.utils.extract_emotions(x_train)
    emo_test = linguistic.utils.extract_emotions(x_test)

    X_train = linguistic.utils.build_lbfv(tfidf_train, emb_train, emo_train)
    X_test = linguistic.utils.build_lbfv(tfidf_test, emb_test, emo_test)
    y_train = df_train["label"].values
    y_test = df_test_subset["label"].values

    rf = linguistic.utils.train_classifier(X_train, y_train)
    metrics = linguistic.utils.evaluate_classifier(rf, X_test, y_test)

    return df_train.shape, df_test_subset.shape, metrics
