import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Clustering", layout="wide")

@st.cache_resource
def load_artifact():
    artifact = joblib.load("credit_clustering_models_simple.pkl")
    return artifact

@st.cache_data
def load_data():
    # If you're storing the CSV in the repo
    df = pd.read_csv("Credit Card Customer Data.csv")
    return df
    
artifact = load_artifact()
kmeans = artifact["kmeans"]
dbscan = artifact["dbscan"]
feature_columns = artifact["feature_columns"]
kmeans_labels = artifact["kmeans_labels"]
dbscan_labels = artifact["dbscan_labels"]

st.title("Credit Card Customers – K‑Means vs DBSCAN")

df = load_data()
X = df[feature_columns].copy()
X_proc = preprocessor.transform(X)

algo = st.sidebar.selectbox("Choose clustering algorithm", ["KMeans", "DBSCAN"])

if algo == "KMeans":
    labels = kmeans.predict(X_proc)
    st.subheader(f"K‑Means (k = {kmeans.n_clusters})")
else:
    # DBSCAN has no predict; we cluster the full data again
    labels = dbscan.fit_predict(X_proc)
    st.subheader(
        f"DBSCAN (eps={dbscan.eps}, min_samples={dbscan.min_samples})"
    )

df_clusters = df.copy()
df_clusters[f"{algo}_cluster"] = labels

st.write("Cluster counts (including noise = -1 for DBSCAN):")
st.write(df_clusters[f"{algo}_cluster"].value_counts().sort_index())

st.write("Sample of clustered data:")
st.dataframe(df_clusters.head(20))

# Simple 2D projection via PCA, for visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_proc)
coords_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
coords_df["cluster"] = labels

st.write("2D PCA scatter by cluster:")

st.scatter_chart(coords_df, x="PC1", y="PC2", color="cluster")
