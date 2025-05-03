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
from langdetect import detect

nltk.download('stopwords')

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


# Cargar modelo y vectorizador en inglés
modelo_en = joblib.load('modelo_en.pkl')
vectorizer_en = joblib.load('vectorizer_en.pkl')

# Preprocesamiento en inglés
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

# Cargar modelo y vectorizador
modelo_es = joblib.load('modelo_es.pkl')
vectorizer_es = joblib.load('vectorizer_es.pkl')

# Preprocesamiento
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

# UI de la app
st.title("🛡️ Detector de Fraude en Mensajes")

st.subheader("🔍 Analiza un mensaje individual")
mensaje_usuario = st.text_area("Escribe el mensaje que deseas analizar")

if st.button("Analizar mensaje"):
    resultado = clasificar_mensaje_multilenguaje(mensaje_usuario)
    st.markdown(f"**Resultado:** {resultado}")

# --- Análisis por archivo ---
st.subheader("📂 Analiza un archivo de WhatsApp (.txt)")

# Uploader acepta .txt y .zip
archivo = st.file_uploader("Sube el archivo .txt o .zip exportado de WhatsApp", type=["txt", "zip"])

if archivo is not None:
    mensajes = []

    if archivo.name.endswith(".zip"):
        # Leer ZIP y buscar el archivo _chat.txt
        with zipfile.ZipFile(archivo) as zip_ref:
            nombre_txt = [f for f in zip_ref.namelist() if f.endswith(".txt")][0]
            with zip_ref.open(nombre_txt) as f:
                lineas = f.read().decode("utf-8").splitlines()
    else:
        # Leer archivo de texto directo
        lineas = archivo.read().decode("utf-8").splitlines()

    # Procesar y clasificar
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

    # Botón para descargar
    csv = df_resultados.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar resultados como CSV", data=csv, file_name="resultados_fraude.csv", mime="text/csv")

