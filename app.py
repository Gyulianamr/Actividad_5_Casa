import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MNIST IA", layout="wide")

st.title("🧠 MNIST - PCA + KMeans + SVM")
st.write("Aplicación de reducción de dimensionalidad, clustering y clasificación")

# =========================
# CARGAR MODELOS
# =========================
svm = joblib.load("models/svm.pkl")
pca = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")

# =========================
# CARGAR DATASET (para métricas y visualización)
# =========================
data = pd.read_csv("train.csv")

X = data.drop("label", axis=1)
y = data["label"]

X = X / 255.0

# =========================
# SIDEBAR - PCA
# =========================
st.sidebar.header("⚙️ Configuración")

n_components = st.sidebar.slider("Componentes PCA", 2, 100, 50)

# =========================
# ESCALAR Y PCA
# =========================
X_scaled = scaler.transform(X)
pca_model = PCA(n_components=n_components)
X_pca = pca_model.fit_transform(X_scaled)

# =========================
# 1. VISUALIZACIÓN PCA 2D
# =========================
st.subheader("📉 Visualización PCA (2D)")

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

fig1, ax1 = plt.subplots()
scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=1)
ax1.set_title("PCA 2D (MNIST)")
st.pyplot(fig1)

# =========================
# 2. K-MEANS
# =========================
st.subheader("🔵 Clusters con K-Means")

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_pca)

fig2, ax2 = plt.subplots()
ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=1)
ax2.set_title("K-Means Clustering")
st.pyplot(fig2)

# =========================
# 3. MÉTRICAS SVM
# =========================
st.subheader("📊 Métricas del modelo SVM")

X_train = X_pca[:60000]
y_train = y[:60000]

X_test = X_pca[60000:]
y_test = y[60000:]

y_pred = svm.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{acc:.4f}")
st.text(classification_report(y_test, y_pred))

# =========================
# 4. PREDICCIÓN CON IMAGEN
# =========================
st.subheader("🔮 Predicción de imagen")

uploaded = st.file_uploader("Sube una imagen de un dígito", type=["png", "jpg", "jpeg"])

if uploaded is not None:

    img = Image.open(uploaded).convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img)

    # 🔥 corregir estilo MNIST
    img_array = 255 - img_array

    st.image(img_array, width=150)

    img_array = img_array.reshape(1, -1) / 255.0
    img_scaled = scaler.transform(img_array)
    img_pca = pca.transform(img_scaled)

    pred = svm.predict(img_pca)

    st.success(f"Predicción del modelo: {pred[0]}")
