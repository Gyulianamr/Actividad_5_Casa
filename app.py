import streamlit as st
import numpy as np
import joblib
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MNIST Predictor", layout="centered")

st.title("🧠 Clasificador de Dígitos MNIST")
st.write("Sube una imagen de un número y el modelo lo predecirá")

# =========================
# CARGAR MODELOS
# =========================
svm = joblib.load("models/svm.pkl")
pca = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")

# =========================
# SUBIR IMAGEN
# =========================
uploaded_file = st.file_uploader("📤 Sube una imagen (png/jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # abrir imagen
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img)

    # 🔥 corregir estilo MNIST
    img_array = 255 - img_array

    st.image(img_array, caption="Imagen procesada (28x28)", width=150)

    # =========================
    # PREPROCESAMIENTO
    # =========================
    img_array = img_array.reshape(1, -1) / 255.0
    img_scaled = scaler.transform(img_array)
    img_pca = pca.transform(img_scaled)

    # =========================
    # PREDICCIÓN
    # =========================
    pred = svm.predict(img_pca)

    st.subheader(f"🔮 Predicción: {pred[0]}")