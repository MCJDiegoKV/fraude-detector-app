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
from train_from_feedback import entrenar_modelo_es_desde_feedback
from train_from_feedback import entrenar_modelo_en_desde_feedback

nltk.download('stopwords')
feedback_file = "feedback.csv"

def clasificar_mensaje_multilenguaje(texto):
    texto = texto.strip()
    if len(texto) < 6:
        return "⚠️ Mensaje muy corto para analizar"
    try:
        idioma = detect(texto)
    except:
        idioma = "es" 

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

def explicar_clasificacion(texto, modelo, vectorizer, preprocess_func, clase_objetivo):
    try:
        texto_proc = preprocess_func(texto)
        texto_vect = vectorizer.transform([texto_proc])

        palabras = vectorizer.get_feature_names_out()
        log_probs = modelo.feature_log_prob_[clase_objetivo]

        import numpy as np
        pesos = texto_vect.toarray()[0] * log_probs
        importantes = [(palabras[i], pesos[i]) for i in range(len(palabras)) if pesos[i] > 0]
        importantes.sort(key=lambda x: x[1], reverse=True)
        return importantes[:5]
    except:
        return []

# UI app
st.title("🛡️ Detector de Fraude en Mensajes")
st.subheader("🔍 Analiza un mensaje individual")
mensaje_usuario = st.text_area("Escribe el mensaje que deseas analizar")

def guardar_feedback(mensaje, prediccion, etiqueta_real):
    try:
        idioma = detect(mensaje)
    except:
        idioma = "es"
    archivo = None
    if idioma == "es":
        archivo = "feedback_es.csv"
    elif idioma == "en":
        archivo = "feedback_en.csv"
    if archivo:
        if not os.path.exists(archivo):
            with open(archivo, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["mensaje", "prediccion", "etiqueta_real"])
        with open(archivo, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([mensaje, prediccion, etiqueta_real])

if st.button("Analizar mensaje"):
    resultado = clasificar_mensaje_multilenguaje(mensaje_usuario)
    st.markdown(f"**Resultado:** {resultado}")
    try:
        idioma = detect(mensaje_usuario)
    except:
        idioma = "es"
    
    if idioma == "es":
        clase = 1 if resultado == "🚨 FRAUDE/ESTAFA" else 0
        explicacion = explicar_clasificacion(
            mensaje_usuario, modelo_es, vectorizer_es, preprocess, clase_objetivo=clase
        )
    elif idioma == "en":
        clase = 1 if resultado == "🚨 FRAUD/SPAM" else 0
        explicacion = explicar_clasificacion(
            mensaje_usuario, modelo_en, vectorizer_en, preprocess_en, clase_objetivo=clase
        )
    else:
        explicacion = []

    
    if explicacion:
        palabras_clave = [f"• {palabra} (peso: {round(peso, 2)})" for palabra, peso in explicacion]
        st.markdown("**🔎 Palabras que influyeron en la clasificación:**")
        st.markdown("\n".join(palabras_clave))

    st.markdown("¿Fue esta clasificación correcta?")
    col1, col2 = st.columns(2)
    if col1.button("✅ Sí, fue correcta"):
        st.success("Gracias por confirmar.")
        guardar_feedback(mensaje_usuario, resultado, resultado)
    if "mostrar_correccion" not in st.session_state:
        st.session_state.mostrar_correccion = False
    if col2.button("❌ No, fue incorrecta"):
        st.session_state.mostrar_correccion = True
    if st.session_state.mostrar_correccion:
        st.warning("Gracias. ¿Cuál es la clasificación correcta?")
        opcion = st.radio("Selecciona la clase correcta", ["✅ Seguro", "🚨 FRAUDE/ESTAFA", "✅ Safe", "🚨 FRAUD/SPAM"])
        if st.button("Guardar corrección"):
            guardar_feedback(mensaje_usuario, resultado, opcion)
            st.success("Se ha guardado la corrección.")
            st.session_state.mostrar_correccion = False


# Análisis por archivo
st.subheader("📂 Analiza un archivo de WhatsApp (.txt)")

# Uploader acepta .txt y .zip
archivo = st.file_uploader("Sube el archivo .txt o .zip exportado de WhatsApp", type=["txt", "zip"])

st.subheader("🔄 Reentrenar modelos con retroalimentación")
if st.button("🧠 Reentrenar ambos modelos"):
    entrenar_modelo_es_desde_feedback()
    entrenar_modelo_en_desde_feedback()


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
        resultado = clasificar_mensaje_multilenguaje(mensaje)

        try:
            idioma = detect(mensaje)
        except:
            idioma = "es"

        if idioma == "es":
            clase = 1 if resultado == "🚨 FRAUDE/ESTAFA" else 0
            explicacion = explicar_clasificacion(
                mensaje, modelo_es, vectorizer_es, preprocess, clase_objetivo=clase
            )
        elif idioma == "en":
            clase = 1 if resultado == "🚨 FRAUD/SPAM" else 0
            explicacion = explicar_clasificacion(
                mensaje, modelo_en, vectorizer_en, preprocess_en, clase_objetivo=clase
            )
        else:
            explicacion = []

        top_palabras = ", ".join([f"{palabra}" for palabra, _ in explicacion]) if explicacion else "No disponible"

        mensajes.append({
            "Mensaje": mensaje,
            "Clasificación": resultado,
            "Palabras Clave": top_palabras
        })


    df_resultados = pd.DataFrame(mensajes)
    st.success(f"Se analizaron {len(df_resultados)} mensajes.")
    st.dataframe(df_resultados)

    csv = df_resultados.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar resultados como CSV", data=csv, file_name="resultados_fraude.csv", mime="text/csv")

