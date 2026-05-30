import numpy as np

import common.utils
import linguistic.utils

SAMPLE = 500


def test_dataframe_loading():
    df_train = common.utils.get_df_train()
    df_test = common.utils.get_df_test()
    df_validate = common.utils.get_df_validate()

    for df, name in [(df_train, "train"), (df_test, "test"), (df_validate, "validate")]:
        assert df is not None, f"{name} dataframe is None"
        assert len(df) > 0, f"{name} dataframe is empty"
        assert "title" in df.columns, f"{name} missing 'title' column"
        assert "text" in df.columns, f"{name} missing 'text' column"
        assert "label" in df.columns, f"{name} missing 'label' column"
        assert "content" in df.columns, f"{name} missing 'content' column"
        assert df["label"].isin([0, 1]).all(), f"{name} labels must be 0 or 1"


def test_data_cleaning():
    df_train = common.utils.get_df_train()
    df_test = common.utils.get_df_test()
    df_validate = common.utils.get_df_validate()

    for df, name in [(df_train, "train"), (df_test, "test"), (df_validate, "validate")]:
        cleaned = df["content"].apply(linguistic.utils.clean_text)

        assert cleaned.notna().all(), f"{name} has NaN after cleaning"
        assert (cleaned.str.len() > 0).any(), f"{name} all rows empty after cleaning"
        assert not cleaned.str.contains(r"https?://\S+", regex=True).any(), f"{name} still contains URLs"
        assert not cleaned.str.contains(r"[^a-z\s]", regex=True).any(), f"{name} still contains non-alpha chars"
        assert not cleaned.str.startswith(" ").any(), f"{name} has leading spaces"
        assert not cleaned.str.endswith(" ").any(), f"{name} has trailing spaces"


def test_tfidf():
    df_train = common.utils.get_df_train().sample(SAMPLE, random_state=42)
    df_test = common.utils.get_df_test().sample(SAMPLE // 2, random_state=42)

    x_train = df_train["content"].apply(linguistic.utils.clean_text)
    x_test = df_test["content"].apply(linguistic.utils.clean_text)

    tfidf_train = linguistic.utils.extract_tf_idf_fit_transform(x_train)
    tfidf_test = linguistic.utils.extract_tf_idf_transform(x_test)

    assert tfidf_train.shape == (SAMPLE, 20), f"TF-IDF train shape mismatch: {tfidf_train.shape}"
    assert tfidf_test.shape == (SAMPLE // 2, 20), f"TF-IDF test shape mismatch: {tfidf_test.shape}"
    assert not np.isnan(tfidf_train).any(), "TF-IDF train contains NaN"
    assert not np.isnan(tfidf_test).any(), "TF-IDF test contains NaN"


def test_embeddings():
    df_train = common.utils.get_df_train().sample(SAMPLE, random_state=42)
    df_test = common.utils.get_df_test().sample(SAMPLE // 2, random_state=42)

    x_train = df_train["content"].apply(linguistic.utils.clean_text)
    x_test = df_test["content"].apply(linguistic.utils.clean_text)

    emb_train = linguistic.utils.extract_embeddings_fit_transform(x_train)
    emb_test = linguistic.utils.extract_embeddings_transform(x_test)

    assert emb_train.shape == (SAMPLE, 20), f"Embeddings train shape mismatch: {emb_train.shape}"
    assert emb_test.shape == (SAMPLE // 2, 20), f"Embeddings test shape mismatch: {emb_test.shape}"
    assert not np.isnan(emb_train).any(), "Embeddings train contains NaN"
    assert not np.isnan(emb_test).any(), "Embeddings test contains NaN"


def test_emotions():
    df_train = common.utils.get_df_train().sample(SAMPLE, random_state=42)
    df_test = common.utils.get_df_test().sample(SAMPLE // 2, random_state=42)

    x_train = df_train["content"].apply(linguistic.utils.clean_text)
    x_test = df_test["content"].apply(linguistic.utils.clean_text)

    emo_train = linguistic.utils.extract_emotions(x_train)
    emo_test = linguistic.utils.extract_emotions(x_test)

    assert emo_train.shape == (SAMPLE, 8), f"Emotions train shape mismatch: {emo_train.shape}"
    assert emo_test.shape == (SAMPLE // 2, 8), f"Emotions test shape mismatch: {emo_test.shape}"
    assert not np.isnan(emo_train).any(), "Emotions train contains NaN"
    assert ((emo_train >= 0) & (emo_train <= 1)).all(), "Emotion scores must be in [0, 1]"


def test_lbfv():
    df_train = common.utils.get_df_train().sample(SAMPLE, random_state=42)
    df_test = common.utils.get_df_test().sample(SAMPLE // 2, random_state=42)

    x_train = df_train["content"].apply(linguistic.utils.clean_text)
    x_test = df_test["content"].apply(linguistic.utils.clean_text)

    tfidf_train = linguistic.utils.extract_tf_idf_fit_transform(x_train)
    tfidf_test = linguistic.utils.extract_tf_idf_transform(x_test)
    emb_train = linguistic.utils.extract_embeddings_fit_transform(x_train)
    emb_test = linguistic.utils.extract_embeddings_transform(x_test)
    emo_train = linguistic.utils.extract_emotions(x_train)
    emo_test = linguistic.utils.extract_emotions(x_test)

    X_train = linguistic.utils.build_lbfv(tfidf_train, emb_train, emo_train)
    X_test = linguistic.utils.build_lbfv(tfidf_test, emb_test, emo_test)

    assert X_train.shape == (SAMPLE, 48), f"LBFV train shape mismatch: {X_train.shape}"
    assert X_test.shape == (SAMPLE // 2, 48), f"LBFV test shape mismatch: {X_test.shape}"


def test_classifier():
    df_train = common.utils.get_df_train().sample(SAMPLE, random_state=42)
    df_test = common.utils.get_df_test().sample(SAMPLE // 2, random_state=42)

    x_train = df_train["content"].apply(linguistic.utils.clean_text)
    x_test = df_test["content"].apply(linguistic.utils.clean_text)

    tfidf_train = linguistic.utils.extract_tf_idf_fit_transform(x_train)
    tfidf_test = linguistic.utils.extract_tf_idf_transform(x_test)
    emb_train = linguistic.utils.extract_embeddings_fit_transform(x_train)
    emb_test = linguistic.utils.extract_embeddings_transform(x_test)
    emo_train = linguistic.utils.extract_emotions(x_train)
    emo_test = linguistic.utils.extract_emotions(x_test)

    X_train = linguistic.utils.build_lbfv(tfidf_train, emb_train, emo_train)
    X_test = linguistic.utils.build_lbfv(tfidf_test, emb_test, emo_test)
    y_train = df_train["label"].values
    y_test = df_test["label"].values

    rf = linguistic.utils.train_classifier(X_train, y_train)
    metrics = linguistic.utils.evaluate_classifier(rf, X_test, y_test)

    assert metrics["accuracy"] > 0.75, f"Accuracy too low: {metrics['accuracy']:.4f}"
    assert metrics["precision"] > 0.75, f"Precision too low: {metrics['precision']:.4f}"
    assert metrics["recall"] > 0.75, f"Recall too low: {metrics['recall']:.4f}"
    assert metrics["f1"] > 0.75, f"F1 too low: {metrics['f1']:.4f}"

    importances = linguistic.utils.get_feature_importances(rf)
    assert len(importances) == 48
    assert abs(importances.sum() - 1.0) < 1e-6
