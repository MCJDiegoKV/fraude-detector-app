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
stop_words_en = set(stopwords.words('english'))
stemmer_en = PorterStemmer()

#Preprocesamiento
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def preprocess_en(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer_en.stem(word) for word in words if word not in stop_words_en]
    return " ".join(words)

#Reentrenamiento Español
def entrenar_modelo_es_desde_feedback():
    feedback_file = "feedback_es.csv"
    datos_originales_file = "datos_originales_es.csv"

    dataframes = []

    if os.path.exists(feedback_file):
        df_fb = pd.read_csv(feedback_file)
        df_fb = df_fb[df_fb['etiqueta_real'].isin(["✅ Seguro", "🚨 FRAUDE/ESTAFA"])]
        dataframes.append(df_fb)

    if os.path.exists(datos_originales_file):
        df_orig = pd.read_csv(datos_originales_file)
        df_orig = df_orig[df_orig['etiqueta_real'].isin(["✅ Seguro", "🚨 FRAUDE/ESTAFA"])]
        dataframes.append(df_orig)

    if not dataframes:
        st.warning("No hay datos disponibles para reentrenar el modelo en español.")
        return

    df_total = pd.concat(dataframes, ignore_index=True)

    df_total['processed'] = df_total['mensaje'].apply(preprocess)
    df_total['label'] = df_total['etiqueta_real'].map({
        "✅ Seguro": 0,
        "🚨 FRAUDE/ESTAFA": 1
    })

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_total['processed'])
    y = df_total['label']

    modelo = MultinomialNB()
    modelo.fit(X, y)

    joblib.dump(modelo, "modelo_es.pkl")
    joblib.dump(vectorizer, "vectorizer_es.pkl")

    st.success("✅ Modelo en español reentrenado con feedback y datos originales.")

#Reentrenamiento Inglés
def entrenar_modelo_en_desde_feedback():
    feedback_file = "feedback_en.csv"
    datos_originales_file = "datos_originales_en.csv"

    dataframes = []

    if os.path.exists(feedback_file):
        df_fb = pd.read_csv(feedback_file)
        df_fb = df_fb[df_fb['etiqueta_real'].isin(["✅ Safe", "🚨 FRAUD/SPAM"])]
        dataframes.append(df_fb)

    if os.path.exists(datos_originales_file):
        df_orig = pd.read_csv(datos_originales_file)
        df_orig = df_orig[df_orig['etiqueta_real'].isin(["✅ Safe", "🚨 FRAUD/SPAM"])]
        dataframes.append(df_orig)

    if not dataframes:
        st.warning("No hay datos disponibles para reentrenar el modelo en inglés.")
        return

    df_total = pd.concat(dataframes, ignore_index=True)

    df_total['processed'] = df_total['mensaje'].apply(preprocess_en)
    df_total['label'] = df_total['etiqueta_real'].map({
        "✅ Safe": 0,
        "🚨 FRAUD/SPAM": 1
    })

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_total['processed'])
    y = df_total['label']

    modelo = MultinomialNB()
    modelo.fit(X, y)

    joblib.dump(modelo, "modelo_en.pkl")
    joblib.dump(vectorizer, "vectorizer_en.pkl")

    st.success("✅ Modelo en inglés reentrenado con feedback y datos originales.")

