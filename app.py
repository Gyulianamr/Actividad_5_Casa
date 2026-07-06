import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MNIST IA", layout="centered")

st.title("🧠 MNIST Predictor (PCA + SVM + KMeans)")

# =========================
# CARGAR MODELOS
# =========================
svm = joblib.load("models/svm.pkl")
pca = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Configuración PCA")
n_components = st.sidebar.slider("Componentes PCA (visual)", 2, 50, 2)

# =========================
# PREDICCIÓN DE IMAGEN
# =========================
st.subheader("🔮 Predicción de dígito")

uploaded = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded:

    img = Image.open(uploaded).convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img)

    # 🔥 IMPORTANTE PARA MNIST
    img_array = 255 - img_array

    st.image(img_array, caption="Imagen procesada", width=150)

    img_array = img_array.reshape(1, -1) / 255.0
    img_scaled = scaler.transform(img_array)
    img_pca = pca.transform(img_scaled)

    pred = svm.predict(img_pca)

    st.success(f"Predicción: {pred[0]}")

# =========================
# VISUAL SIMPLE PCA (OPCIONAL DEMO)
# =========================
st.subheader("📉 Demo PCA 2D")

st.info("Esta visualización es ilustrativa (dataset reducido en app no incluido)")

X_demo = np.random.rand(500, 784)
X_demo = scaler.transform(X_demo)

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_demo)

fig, ax = plt.subplots()
ax.scatter(X_2d[:,0], X_2d[:,1], s=5)
st.pyplot(fig)
