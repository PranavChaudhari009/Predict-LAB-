import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Fraud Clustering", layout="wide")

st.title("💳 Credit Card Fraud Detection (Clustering)")
st.write("Detect suspicious transaction patterns using KMeans and DBSCAN")

# =========================
# Cached Data Loading + Preprocessing
# =========================
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("data/credit_card_fraud_dataset.csv").copy()

    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["Hour"] = df["TransactionDate"].dt.hour
    df["Day"] = df["TransactionDate"].dt.day
    df["Month"] = df["TransactionDate"].dt.month
    df["DayOfWeek"] = df["TransactionDate"].dt.dayofweek

    df.drop(columns=["TransactionDate", "TransactionID"], inplace=True, errors="ignore")

    df = pd.get_dummies(df, columns=["TransactionType", "Location"], drop_first=True)

    if "MerchantID" in df.columns:
        df.drop(columns=["MerchantID"], inplace=True)

    y = df["IsFraud"]
    X = df.drop(columns=["IsFraud"])
    feature_columns = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X, y, X_scaled, scaler, feature_columns


@st.cache_resource
def train_kmeans(X_scaled, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)
    return model, clusters


@st.cache_resource
def train_dbscan(X_scaled, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(X_scaled)
    return model, clusters


@st.cache_data
def compute_pca(X_scaled):
    pca = PCA(n_components=2)
    return pca.fit_transform(X_scaled)


# =========================
# Load Data
# =========================
df, X, y, X_scaled, scaler, feature_columns = load_and_preprocess_data()

# =========================
# Model Selection
# =========================
model_choice = st.selectbox("Select Model", ["KMeans", "DBSCAN"])

if model_choice == "KMeans":
    n_clusters = st.slider("Number of Clusters", 2, 6, 3)
    model, clusters = train_kmeans(X_scaled, n_clusters)
else:
    eps = st.slider("EPS", 0.1, 2.0, 0.5)
    min_samples = st.slider("Min Samples", 2, 10, 5)
    model, clusters = train_dbscan(X_scaled, eps, min_samples)

df = df.copy()
df["Cluster"] = clusters

# =========================
# User Input
# =========================
st.subheader("🔍 Check New Transaction")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Amount", value=1000.0, min_value=0.0)
    transaction_type = st.selectbox("Transaction Type", ["Purchase", "Refund"])

with col2:
    location = st.selectbox("Location", ["Dallas", "Houston", "New York", "Los Angeles"])
    date = st.date_input("Transaction Date")

input_df = pd.DataFrame({
    "Amount": [amount],
    "TransactionDate": [date]
})

input_df["TransactionDate"] = pd.to_datetime(input_df["TransactionDate"])
input_df["Hour"] = input_df["TransactionDate"].dt.hour
input_df["Day"] = input_df["TransactionDate"].dt.day
input_df["Month"] = input_df["TransactionDate"].dt.month
input_df["DayOfWeek"] = input_df["TransactionDate"].dt.dayofweek
input_df.drop(columns=["TransactionDate"], inplace=True)

if "TransactionType_Refund" in feature_columns:
    input_df["TransactionType_Refund"] = 1 if transaction_type == "Refund" else 0

for loc in ["Houston", "Los Angeles", "New York"]:
    col_name = f"Location_{loc}"
    if col_name in feature_columns:
        input_df[col_name] = 1 if location == loc else 0

input_df = input_df.reindex(columns=feature_columns, fill_value=0)
input_scaled = scaler.transform(input_df)

# =========================
# Prediction
# =========================
if st.button("Check Transaction Pattern"):
    if model_choice == "KMeans":
        cluster = model.predict(input_scaled)[0]
        fraud_rate = df.groupby("Cluster")["IsFraud"].mean()
        risk = float(fraud_rate.get(cluster, 0.0))

        if risk > 0.5:
            st.error("🚨 High Risk Transaction")
        elif risk > 0.2:
            st.warning("⚠️ Medium Risk Transaction")
        else:
            st.success("✅ Low Risk Transaction")

        st.write(f"Cluster: {cluster}")
        st.write(f"Fraud Probability in Cluster: {risk:.2f}")

    else:
        distances = np.linalg.norm(X_scaled - input_scaled, axis=1)
        neighbor_count = int(np.sum(distances <= eps))
        nearest_distance = float(np.min(distances))

        if neighbor_count < min_samples:
            st.error("🚨 Potential Fraud (Outlier Detected)")
        else:
            st.success("✅ Normal Transaction")

        st.write(f"Neighbors within EPS: {neighbor_count}")
        st.write(f"Nearest Distance: {nearest_distance:.3f}")

# =========================
# Visualization
# =========================
st.subheader("📊 Cluster Visualization")

X_pca = compute_pca(X_scaled)

fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster"], cmap="viridis", s=20)
ax.set_title("Clusters (PCA)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
st.pyplot(fig)

# =========================
# Cluster Summary
# =========================
st.subheader("📋 Cluster Summary")
summary = df.groupby("Cluster")["IsFraud"].mean().reset_index(name="FraudRate")
st.dataframe(summary)

st.subheader("Dataset Preview")
st.dataframe(df.head())
