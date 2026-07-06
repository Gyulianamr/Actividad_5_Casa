import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
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

st.write("Aplicación de Inteligencia Artificial con reducción de dimensionalidad, clustering y clasificación")

# =========================
# CARGAR MODELOS
# =========================
svm = joblib.load("models/svm.pkl")
pca = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")

# =========================
# CARGAR DATASET
# =========================
data = pd.read_csv("train.csv")

X = data.drop("label", axis=1)
y = data["label"]

X = X / 255.0

# =========================
# ESCALAR
# =========================
X_scaled = scaler.transform(X)

# =========================
# SIDEBAR PCA
# =========================
st.sidebar.header("⚙️ Configuración")

n_components = st.sidebar.slider("Componentes PCA", 2, 100, 50)

# =========================
# PCA 2D (VISUALIZACIÓN REAL)
# =========================
st.subheader("📉 Visualización PCA en 2D")

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

fig, ax = plt.subplots()
ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=1)
ax.set_title("Proyección PCA 2D")
st.pyplot(fig)

# =========================
# K-MEANS
# =========================
st.subheader("🔵 Clusters con K-Means")

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

fig2, ax2 = plt.subplots()
ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=1)
ax2.set_title("K-Means Clustering")
st.pyplot(fig2)

# =========================
# MÉTRICAS SVM
# =========================
st.subheader("📊 Métricas del modelo SVM")

X_train = X_scaled[:60000]
y_train = y[:60000]

X_test = X_scaled[60000:]
y_test = y[60000:]

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

y_pred = svm.predict(X_test_pca)

acc = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{acc:.4f}")
st.text(classification_report(y_test, y_pred))

# =========================
# PREDICCIÓN CON IMAGEN
# =========================
st.subheader("🔮 Predicción de dígito")

uploaded = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded:

    img = Image.open(uploaded).convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img)

    # corregir estilo MNIST
    img_array = 255 - img_array

    st.image(img_array, caption="Imagen procesada", width=150)

    img_array = img_array.reshape(1, -1) / 255.0
    img_scaled = scaler.transform(img_array)
    img_pca = pca.transform(img_scaled)

    pred = svm.predict(img_pca)

    st.success(f"🔮 Predicción: {pred[0]}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.write("📌 Elaborado por: Genesis Yuliana Medina - 20231900117")
