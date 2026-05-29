#!/usr/bin/env python
# coding: utf-8

# # Prueba de analisis con noticia de dataset
# 
# ## Paso 1 Cargar el dataset y limpiar el texto

# In[1]:


import re

import pandas as pd
from datasets import load_dataset

# Cargar dataset
dataset = load_dataset("GonzaloA/fake_news")
df_train = pd.DataFrame(dataset["train"])
df_test = pd.DataFrame(dataset["test"])

# Combinar título y texto en un solo campo
df_train["content"] = (
    df_train["title"].fillna("") + " " + df_train["text"].fillna("")
)
df_test["content"] = (
    df_test["title"].fillna("") + " " + df_test["text"].fillna("")
)


# Función de limpieza básica
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # quitar URLs
    text = re.sub(r"[^a-z\s]", "", text)  # quitar caracteres especiales
    text = re.sub(r"\s+", " ", text).strip()  # quitar espacios extra
    return text


df_train["content_clean"] = df_train["content"].apply(clean_text)
df_test["content_clean"] = df_test["content"].apply(clean_text)

print(f"Train: {len(df_train)} muestras")
print(f"Test: {len(df_test)} muestras")
print("---")
print("Ejemplo texto original:")
print(df_train["content"].iloc[0][:200])
print("---")
print("Ejemplo texto limpio:")
print(df_train["content_clean"].iloc[0][:200])


# ## Paso 2 Extraer features TF-IDF

# In[2]:


import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

print("Extrayendo features TF-IDF...")

# TF-IDF: máximo 10000 términos, ignorar palabras muy frecuentes y muy raras
tfidf = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95)
tfidf_train = tfidf.fit_transform(df_train["content_clean"])
tfidf_test = tfidf.transform(df_test["content_clean"])

# Reducir a 20 dimensiones con PCA (igual que Hashemi)
print("Reduciendo dimensionalidad TF-IDF con PCA...")
pca_tfidf = PCA(n_components=20)
tfidf_train_pca = pca_tfidf.fit_transform(tfidf_train.toarray())
tfidf_test_pca = pca_tfidf.transform(tfidf_test.toarray())

print(f"TF-IDF train shape: {tfidf_train_pca.shape}")
print(f"TF-IDF test shape:  {tfidf_test_pca.shape}")
print("TF-IDF OK")


# ## Paso 3 Extraer embeddings semánticos con spaCy

# In[3]:


import spacy

print("Cargando modelo spaCy...")
nlp = spacy.load("en_core_web_md")

# Limitar a 100 palabras por texto para acelerar (igual que Hashemi con 100 tweets)
def get_embedding(text):
    doc = nlp(text[:1000])  # limitar caracteres para velocidad
    return doc.vector        # vector de 300 dimensiones

print("Extrayendo embeddings train (puede tardar unos minutos)...")
embeddings_train = np.array([get_embedding(t) for t in df_train['content_clean']])

print("Extrayendo embeddings test...")
embeddings_test = np.array([get_embedding(t) for t in df_test['content_clean']])

# Reducir a 20 dimensiones con PCA (igual que Hashemi)
print("Reduciendo dimensionalidad embeddings con PCA...")
pca_emb = PCA(n_components=20)
embeddings_train_pca = pca_emb.fit_transform(embeddings_train)
embeddings_test_pca = pca_emb.transform(embeddings_test)

print(f"Embeddings train shape: {embeddings_train_pca.shape}")
print(f"Embeddings test shape:  {embeddings_test_pca.shape}")
print("Embeddings OK")


# ## Paso intermedio traer paquetes de tokens

# In[5]:


import nltk

nltk.download('punkt_tab')
nltk.download('wordnet')


# ## Paso 4 Extraer señales emocionales con NRCLex

# In[6]:


from nrclex import NRCLex

print("Extrayendo señales emocionales...")

# Las 8 emociones del lexicón NRC (igual que Hashemi)
EMOTIONS = ['fear', 'anger', 'anticipation', 'trust', 
            'surprise', 'sadness', 'disgust', 'joy']

def get_emotions(text):
    emotion = NRCLex()
    emotion.load_raw_text(text[:1000])
    scores = emotion.affect_frequencies
    # Devolver las 8 emociones en orden fijo, 0 si no aparece
    return [scores.get(e, 0.0) for e in EMOTIONS]

emotions_train = np.array([get_emotions(t) for t in df_train['content_clean']])
emotions_test = np.array([get_emotions(t) for t in df_test['content_clean']])

print(f"Emotions train shape: {emotions_train.shape}")
print(f"Emotions test shape:  {emotions_test.shape}")

# Ejemplo: ver las emociones de la primera noticia
print("---")
print(f"Label primera noticia: {'FAKE' if df_train['label'].iloc[0] == 1 else 'REAL'}")
for e, v in zip(EMOTIONS, emotions_train[0]):
    print(f"  {e}: {v:.4f}")
print("Emociones OK")


# ## Paso 5 Combinar los tres bloques y entrenar el clasificador

# In[7]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

print("Combinando features en LBFV...")

# Concatenar los tres bloques: TF-IDF (20) + Embeddings (20) + Emociones (8) = 48 features
X_train = np.concatenate([tfidf_train_pca, embeddings_train_pca, emotions_train], axis=1)
X_test = np.concatenate([tfidf_test_pca, embeddings_test_pca, emotions_test], axis=1)
y_train = df_train['label'].values
y_test = df_test['label'].values

print(f"Shape LBFV train: {X_train.shape}")
print(f"Shape LBFV test:  {X_test.shape}")
print("---")

print("Entrenando Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("Evaluando...")
y_pred = rf.predict(X_test)

print("---")
print("RESULTADOS")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print("---")
print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))


# ## Paso 6 Análisis de importancia de features

# In[8]:


print("Analizando importancia de features...")

# Nombres de las features
feature_names = (
    [f"tfidf_{i}" for i in range(20)] +
    [f"emb_{i}" for i in range(20)] +
    EMOTIONS
)

# Importancia por grupo
importances = rf.feature_importances_
tfidf_imp = importances[:20].mean()
emb_imp = importances[20:40].mean()
emo_imp = importances[40:].mean()

print("---")
print("IMPORTANCIA POR GRUPO DE FEATURES")
print(f"TF-IDF:     {tfidf_imp:.4f}")
print(f"Embeddings: {emb_imp:.4f}")
print(f"Emociones:  {emo_imp:.4f}")
print("---")
print("IMPORTANCIA POR EMOCION")
for name, imp in zip(EMOTIONS, importances[40:]):
    print(f"  {name}: {imp:.4f}")


# ## Paso 8 Guardar el modelo y los resultados

# In[12]:


import joblib
import json
import os

print("Guardando modelo y resultados...")

os.makedirs("modelo", exist_ok=True)   

# Guardar modelo y transformadores
joblib.dump(rf, 'modelo/modelo_rf.pkl')
joblib.dump(tfidf, 'modelo/tfidf_vectorizer.pkl')
joblib.dump(pca_tfidf, 'modelo/pca_tfidf.pkl')
joblib.dump(pca_emb, 'modelo/pca_emb.pkl')

# Guardar resultados en JSON para el TFM
resultados = {
    "dataset": "GonzaloA/fake_news",
    "train_samples": len(df_train),
    "test_samples": len(df_test),
    "features": {
        "tfidf_dims": 20,
        "embedding_dims": 20,
        "emotion_dims": 8,
        "total_dims": 48
    },
    "metricas": {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4)
    },
    "importancia_grupos": {
        "tfidf": round(float(tfidf_imp), 4),
        "embeddings": round(float(emb_imp), 4),
        "emociones": round(float(emo_imp), 4)
    },
    "importancia_emociones": {
        e: round(float(importances[40+i]), 4)
        for i, e in enumerate(EMOTIONS)
    }
}

with open('resultados.json', 'w') as f:
    json.dump(resultados, f, indent=2)

print("Guardado: modelo_rf.pkl, tfidf_vectorizer.pkl, pca_tfidf.pkl, pca_emb.pkl")
print("Guardado: resultados.json")
print("---")
print("PIPELINE COMPLETO")


# In[ ]:




