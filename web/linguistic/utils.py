import re
from typing import Any

import nltk
import numpy as np
from pandas import Series
import spacy
from nrclex import NRCLex
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from schemas import Metrics


# TF-IDF: máximo 10000 términos, ignorar palabras muy frecuentes y muy raras
tfidf = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95)

# Reducir a 20 dimensiones con PCA (igual que Hashemi)
pca_tfidf = PCA(n_components=20)

# Cargar modelo de spacy para ingles
nlp = spacy.load("en_core_web_md")

# Reducir a 20 dimensiones con PCA (igual que Hashemi)
pca_emb = PCA(n_components=20)

# Descargar pacquetes de tokens
nltk.download("punkt_tab")
nltk.download("wordnet")

# Las 8 emociones del lexicón NRC (igual que Hashemi)
EMOTIONS = [
    "fear",
    "anger",
    "anticipation",
    "trust",
    "surprise",
    "sadness",
    "disgust",
    "joy",
]


# Limitar a 100 palabras por texto para acelerar (igual que Hashemi con 100 tweets)
def get_embedding(text: str):
    doc = nlp(text[:1000])  # limitar caracteres para velocidad
    return doc.vector  # vector de 300 dimensiones


def get_emotions(text: str):
    emotion = NRCLex()
    emotion.load_raw_text(text[:1000])
    scores = emotion.affect_frequencies
    # Devolver las 8 emociones en orden fijo, 0 si no aparece
    return [scores.get(e, 0.0) for e in EMOTIONS]


# Función de limpieza básica
def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # quitar URLs
    text = re.sub(r"[^a-z\s]", "", text)  # quitar caracteres especiales
    text = re.sub(r"\s+", " ", text).strip()  # quitar espacios extra
    return text


def extract_tf_idf_fit_transform(clean_content: Series[str]):
    """Extract term frequency - inverse document frecuency"""
    # print("Extrayendo features TF-IDF...")

    # TF-IDF: máximo 10000 términos, ignorar palabras muy frecuentes y muy raras
    tfidf_terms = tfidf.fit_transform(clean_content)

    # Reducir a 20 dimensiones con PCA (igual que Hashemi)
    # print("Reduciendo dimensionalidad TF-IDF con PCA...")
    tfidf_pca = pca_tfidf.fit_transform(tfidf_terms.toarray())

    # print(f"TF-IDF train shape: {tfidf_pca.shape}")
    # print("TF-IDF OK")
    return tfidf_pca


def extract_tf_idf_transform(clean_content: Series[str]):
    """Extract term frequency - inverse document frecuency"""
    # print("Extrayendo features TF-IDF...")

    # TF-IDF: máximo 10000 términos, ignorar palabras muy frecuentes y muy raras
    tfidf_terms = tfidf.transform(clean_content)

    # Reducir a 20 dimensiones con PCA (igual que Hashemi)
    # print("Reduciendo dimensionalidad TF-IDF con PCA...")
    tfidf_pca = pca_tfidf.transform(tfidf_terms.toarray())

    # print(f"TF-IDF test shape:  {tfidf_pca.shape}")
    # print("TF-IDF OK")
    return tfidf_pca


def extract_embeddings_fit_transform(clean_content: Series[str]):
    """Extract spaCy sentence embeddings and fit PCA, returning 20-dim vectors."""
    embeddings = np.array([get_embedding(t) for t in clean_content])
    embeddings_pca = pca_emb.fit_transform(embeddings)
    return embeddings_pca


def extract_embeddings_transform(clean_content: Series[str]):
    """Project spaCy embeddings with the already-fitted PCA, returning 20-dim vectors."""
    embeddings = np.array([get_embedding(t) for t in clean_content])
    embeddings_pca = pca_emb.transform(embeddings)
    return embeddings_pca


def extract_emotions(clean_content: Series[str]):
    """Extract 8 NRCLex emotion scores for each text, returning shape (n, 8)."""
    return np.array([get_emotions(t) for t in clean_content])


def build_lbfv(
    tfidf_features: np.ndarray[tuple[int, int], Any],
    embedding_features: np.ndarray[tuple[int, int], Any],
    emotion_features: np.ndarray[tuple[int, int], Any],
):
    """Concatenate TF-IDF (20) + embeddings (20) + emotions (8) into a 48-dim LBFV matrix."""
    return np.concatenate(
        [tfidf_features, embedding_features, emotion_features],
        axis=1,
    )


def train_classifier(
    x_train: np.ndarray[tuple[int, int], Any],
    y_train: np.ndarray[Any],
):
    """Fit and return a Random Forest classifier on the LBFV features."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(x_train, y_train)
    return rf


def evaluate_classifier(
    clf: RandomForestClassifier,
    x_test: np.ndarray[tuple[int, int], Any],
    y_test: np.ndarray[Any],
) -> Metrics:
    """Return dict with accuracy, precision, recall, f1 for the given classifier."""
    y_pred = clf.predict(x_test)

    metrics = Metrics(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
    )

    return metrics


def get_feature_importances(clf: RandomForestClassifier):
    """Return feature importances array from a fitted classifier."""
    # Nombres de las features
    feature_names = (
        [f"tfidf_{i}" for i in range(20)] +
        [f"emb_{i}" for i in range(20)] +
        EMOTIONS
    )

    # Importancia por grupo
    importances = clf.feature_importances_

    return importances
