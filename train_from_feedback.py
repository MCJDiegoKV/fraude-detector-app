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

stop_words = set(stopwords.words('spanish'))
stemmer = PorterStemmer()

feedback_file = "feedback.csv"
datos_originales_file = "datos_originales_es.csv"

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def entrenar_modelo_es_desde_feedback():
    # Verifica existencia de ambos archivos
    if not os.path.exists(feedback_file) and not os.path.exists(datos_originales_file):
        st.warning("No hay datos disponibles para entrenar.")
        return

    dataframes = []

    # Cargar feedback si existe
    if os.path.exists(feedback_file):
        df_fb = pd.read_csv(feedback_file)
        df_fb = df_fb[df_fb['etiqueta_real'].isin(["✅ Seguro", "🚨 FRAUDE/ESTAFA"])]
        dataframes.append(df_fb)

    # Cargar originales si existe
    if os.path.exists(datos_originales_file):
        df_orig = pd.read_csv(datos_originales_file)
        df_orig = df_orig[df_orig['etiqueta_real'].isin(["✅ Seguro", "🚨 FRAUDE/ESTAFA"])]
        dataframes.append(df_orig)

    # Combinar
    df_total = pd.concat(dataframes, ignore_index=True)

    if df_total.empty:
        st.warning("No hay suficientes datos válidos para entrenar.")
        return

    # Preprocesar
    df_total['processed'] = df_total['mensaje'].apply(preprocess)
    df_total['label'] = df_total['etiqueta_real'].map({
        "✅ Seguro": 0,
        "🚨 FRAUDE/ESTAFA": 1
    })

    # Vectorizar y entrenar
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_total['processed'])
    y = df_total['label']

    modelo = MultinomialNB()
    modelo.fit(X, y)

    # Guardar
    joblib.dump(modelo, "modelo_es.pkl")
    joblib.dump(vectorizer, "vectorizer_es.pkl")

    st.success("✅ Modelo reentrenado combinando datos originales y feedback.")
