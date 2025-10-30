import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- Configuración ---
st.title("📊 Clustering con K-Means (Actividad 2.4)")
st.write("Selecciona los parámetros para personalizar el algoritmo K-Means.")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube un archivo CSV (ejemplo: analisis.csv)", type=["csv"])

if uploaded_file is not None:
    # Leer CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # Columnas numéricas
    columnas_num = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    st.sidebar.subheader("⚙️ Configuración de parámetros")

    # Selección de variables
    variables = st.sidebar.multiselect("Selecciona variables para clustering:", columnas_num, default=columnas_num)

    if len(variables) > 0:
        X = df[variables]

        # Escalado de datos
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Parámetros de KMeans
        k = st.sidebar.slider("Número de clusters (k)", 2, 10, 3)
        init = st.sidebar.selectbox("Método de inicialización (init)", ["k-means++", "random"])
        max_iter = st.sidebar.slider("Máx. iteraciones (max_iter)", 100, 1000, 300, step=50)
        n_init = st.sidebar.slider("Número de reinicios (n_init)", 1, 20, 10)
        random_state = st.sidebar.number_input("Random State", min_value=0, value=42)

        # Modelo
        kmeans = KMeans(
            n_clusters=k,
            init=init,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state
        )
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        # Mostrar resultados
        st.subheader("📌 Resultados del Clustering")
        st.dataframe(df)

        # Gráfico
        st.subheader("📈 Visualización de Clusters")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df["Cluster"], cmap="tab10")
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1] if len(variables) > 1 else variables[0])
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)

else:
    st.info("📂 Carga un archivo CSV para comenzar. Ejemplo: analisis.csv")
