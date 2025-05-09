import streamlit as st
import joblib
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
import nltk
import pandas as pd
import re
import zipfile
import io
import csv
import os
from langdetect import detect

nltk.download('stopwords')
feedback_file = "feedback.csv"

def clasificar_mensaje_multilenguaje(texto):
    try:
        idioma = detect(texto)
    except:
        return "❌ No se pudo detectar idioma"

    if idioma == 'es':
        return detectar_fraude_es(texto)
    elif idioma == 'en':
        return detectar_fraude_en(texto)
    else:
        return "🌐 Idioma no soportado"

if not os.path.exists(feedback_file):
    with open(feedback_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["mensaje", "prediccion", "etiqueta_real"])

modelo_en = joblib.load('modelo_en.pkl')
vectorizer_en = joblib.load('vectorizer_en.pkl')

stop_words_en = set(stopwords.words('english'))
stemmer_en = PorterStemmer()

def preprocess_en(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer_en.stem(word) for word in words if word not in stop_words_en]
    return " ".join(words)

def detectar_fraude_en(texto):
    texto_proc = preprocess_en(texto)
    texto_vect = vectorizer_en.transform([texto_proc])
    pred = modelo_en.predict(texto_vect)[0]
    return "🚨 FRAUD/SPAM" if pred == 1 else "✅ Safe"

modelo_es = joblib.load('modelo_es.pkl')
vectorizer_es = joblib.load('vectorizer_es.pkl')

stop_words = set(stopwords.words('spanish'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def detectar_fraude_es(texto):
    texto_proc = preprocess(texto)
    texto_vect = vectorizer_es.transform([texto_proc])
    pred = modelo_es.predict(texto_vect)[0]
    return "🚨 FRAUDE/ESTAFA" if pred == 1 else "✅ Seguro"

def extraer_mensaje(linea):
    match = re.search(r"\] (.*?): (.*)", linea)
    if match:
        return match.group(2).strip()
    return None

# UI app
st.title("🛡️ Detector de Fraude en Mensajes")
st.subheader("🔍 Analiza un mensaje individual")
mensaje_usuario = st.text_area("Escribe el mensaje que deseas analizar")

if st.button("Analizar mensaje"):
    resultado = clasificar_mensaje_multilenguaje(mensaje_usuario)
    st.markdown(f"**Resultado:** {resultado}")

    st.markdown("¿Fue esta clasificación correcta?")
    col1, col2 = st.columns(2)
    if col1.button("✅ Sí, fue correcta"):
        st.success("Gracias por confirmar.")
        with open(feedback_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([mensaje_usuario, resultado, resultado])
    if col2.button("❌ No, fue incorrecta"):
        st.warning("Gracias. ¿Cuál es la clasificación correcta?")
        opcion = st.radio("Selecciona la clase correcta", ["✅ Seguro", "🚨 FRAUDE/ESTAFA", "🚨 FRAUD/SPAM"])
        if st.button("Guardar corrección"):
            with open(feedback_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([mensaje_usuario, resultado, opcion])
            st.success("Se ha guardado la corrección.")

# --- Análisis por archivo ---
st.subheader("📂 Analiza un archivo de WhatsApp (.txt)")

# Uploader acepta .txt y .zip
archivo = st.file_uploader("Sube el archivo .txt o .zip exportado de WhatsApp", type=["txt", "zip"])

st.subheader("🔄 Reentrenar modelo en español con retroalimentación")
if st.button("🧠 Reentrenar ahora"):
    entrenar_modelo_es_desde_feedback()

if archivo is not None:
    mensajes = []

    if archivo.name.endswith(".zip"):
        with zipfile.ZipFile(archivo) as zip_ref:
            nombre_txt = [f for f in zip_ref.namelist() if f.endswith(".txt")][0]
            with zip_ref.open(nombre_txt) as f:
                lineas = f.read().decode("utf-8").splitlines()
    else:
        lineas = archivo.read().decode("utf-8").splitlines()

    for linea in lineas:
        mensaje = extraer_mensaje(linea)
        if mensaje:
            mensajes.append({
                "Mensaje": mensaje,
                "Clasificación": clasificar_mensaje_multilenguaje(mensaje)
            })

    df_resultados = pd.DataFrame(mensajes)
    st.success(f"Se analizaron {len(df_resultados)} mensajes.")
    st.dataframe(df_resultados)

    csv = df_resultados.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar resultados como CSV", data=csv, file_name="resultados_fraude.csv", mime="text/csv")

