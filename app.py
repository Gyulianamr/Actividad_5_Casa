import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MNIST IA", layout="wide")

st.title("🧠 MNIST - PCA + K-Means + SVM")
st.subheader("Genesis Yuliana Medina - 20231900117")

st.write("Sistema de clasificación de dígitos manuscritos con IA")

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
n_components = st.sidebar.slider("Componentes PCA (visualización)", 2, 50, 2)
# =========================
# PREDICCIÓN DE IMAGEN
# =========================
st.subheader("🔮 Predicción de dígito")

uploaded = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded:

    img = Image.open(uploaded).convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img)

    # 🔥 corregir estilo MNIST
    img_array = 255 - img_array

    st.image(img_array, caption="Imagen procesada", width=150)

    img_array = img_array.reshape(1, -1) / 255.0
    img_scaled = scaler.transform(img_array)
    img_pca = pca.transform(img_scaled)

    pred = svm.predict(img_pca)

    st.success(f"🔮 Predicción: {pred[0]}")

# =========================
# VISUALIZACIÓN PCA 2D
# =========================
st.subheader("📉 Visualización PCA en 2D")

# datos simulados SOLO para visualización (evita train.csv)
X_demo = np.random.rand(1000, 784)
X_demo = scaler.transform(X_demo)

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_demo)

fig, ax = plt.subplots()
ax.scatter(X_2d[:, 0], X_2d[:, 1], s=5)
ax.set_title("Proyección PCA 2D")
st.pyplot(fig)

# =========================
# K-MEANS
# =========================
st.subheader("🔵 Clusters con K-Means")

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_demo)

fig2, ax2 = plt.subplots()
ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=5)
ax2.set_title("K-Means Clustering")
st.pyplot(fig2)

# =========================
# MÉTRICAS (desde CSV opcional)
# =========================
st.subheader("📊 Métricas del modelo")

try:
    df_metrics = pd.read_csv("metricas.csv")
    st.dataframe(df_metrics)
except:
    st.warning("No se encontró metricas.csv. Ejecuta entrenamiento en Colab.")



# =========================
# FOOTER
# =========================
st.markdown("---")
st.write("📌 Elaborado por: Genesis Yuliana Medina - 20231900117")
