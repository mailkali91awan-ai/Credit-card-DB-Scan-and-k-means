import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

st.set_page_config(page_title="Credit Card Clustering", layout="wide")


# ---------- LOAD PICKLED MODELS ----------
@st.cache_resource
def load_artifact():
    # credit_clustering_models_simple.pkl must contain:
    #   "kmeans", "dbscan", "feature_columns", "kmeans_labels", "dbscan_labels"
    return joblib.load("credit_clustering_models_simple.pkl")


# ---------- LOAD RAW DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv("Credit Card Customer Data.csv")


artifact = load_artifact()
kmeans = artifact["kmeans"]
dbscan = artifact["dbscan"]
feature_columns = artifact["feature_columns"]
kmeans_labels_trained = artifact["kmeans_labels"]
dbscan_labels_trained = artifact["dbscan_labels"]

df = load_data()
X = df[feature_columns].copy()


# ---------- BUILD PREPROCESSOR (same logic as in Kaggle) ----------
@st.cache_resource
def build_preprocessor(df_subset: pd.DataFrame):
    X_sub = df_subset.copy()

    numeric_features = X_sub.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_sub.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_proc_sub = preprocessor.fit_transform(X_sub)
    return preprocessor, X_proc_sub


preprocessor, X_proc = build_preprocessor(X)


# ---------- SIDEBAR: CHOOSE ALGORITHM ----------
st.title("Credit Card Customers – K‑Means vs DBSCAN")

algo = st.sidebar.selectbox("Choose clustering algorithm", ["KMeans", "DBSCAN"])

if algo == "KMeans":
    labels = kmeans.predict(X_proc)
    st.subheader(f"K‑Means (k = {kmeans.n_clusters})")
else:
    # DBSCAN has no standard predict; we cluster the full data again
    labels = dbscan.fit_predict(X_proc)
    st.subheader(f"DBSCAN (eps={dbscan.eps}, min_samples={dbscan.min_samples})")


# ---------- ATTACH LABELS AND SHOW COUNTS ----------
df_clusters = df.copy()
df_clusters[f"{algo}_cluster"] = labels

st.write("Cluster counts (DBSCAN: -1 = noise):")
st.write(df_clusters[f"{algo}_cluster"].value_counts().sort_index())

st.write("Sample of clustered data:")
st.dataframe(df_clusters.head(20))


# ---------- PCA 2D VISUAL ----------
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_proc)
coords_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
coords_df["cluster"] = labels

st.write("2D PCA scatter by cluster:")
st.scatter_chart(coords_df, x="PC1", y="PC2", color="cluster")
