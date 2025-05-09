import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

feedback_file = "feedback.csv"
stop_words = set(stopwords.words('spanish'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def entrenar_modelo_es_desde_feedback():
    if not os.path.exists(feedback_file):
        st.warning("No hay datos de retroalimentación aún.")
        return

    df_fb = pd.read_csv(feedback_file)

    # Solo mensajes en español
    df_fb = df_fb[df_fb['etiqueta_real'].isin(["✅ Seguro", "🚨 FRAUDE/ESTAFA"])]

    if df_fb.empty:
        st.warning("No hay suficientes ejemplos en español.")
        return

    # Procesamiento y etiquetas
    df_fb['processed'] = df_fb['mensaje'].apply(preprocess)
    df_fb['label'] = df_fb['etiqueta_real'].map({
        "✅ Seguro": 0,
        "🚨 FRAUDE/ESTAFA": 1
    })

    # Vectorización y entrenamiento
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_fb['processed'])
    y = df_fb['label']

    modelo = MultinomialNB()
    modelo.fit(X, y)

    # Guardar modelos actualizados
    joblib.dump(modelo, "modelo_es.pkl")
    joblib.dump(vectorizer, "vectorizer_es.pkl")

    st.success("✅ Modelo en español reentrenado con retroalimentación.")
