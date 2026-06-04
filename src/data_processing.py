import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================
# Load Data
# ==========================

df = pd.read_csv("data/raw/data.csv")

# Convert date column
df["TransactionStartTime"] = pd.to_datetime(
    df["TransactionStartTime"]
)

# ==========================
# Feature Extraction
# ==========================

df["Transaction_Hour"] = df["TransactionStartTime"].dt.hour
df["Transaction_Day"] = df["TransactionStartTime"].dt.day
df["Transaction_Month"] = df["TransactionStartTime"].dt.month
df["Transaction_Year"] = df["TransactionStartTime"].dt.year

# ==========================
# Aggregate Features
# ==========================

customer_features = df.groupby("CustomerId").agg(
    Total_Transaction_Amount=("Amount", "sum"),
    Average_Transaction_Amount=("Amount", "mean"),
    Transaction_Count=("Amount", "count"),
    Std_Transaction_Amount=("Amount", "std"),
    Avg_Transaction_Hour=("Transaction_Hour", "mean"),
    Avg_Transaction_Day=("Transaction_Day", "mean"),
    Avg_Transaction_Month=("Transaction_Month", "mean")
).reset_index()

# ==========================
# Handle Missing Values
# ==========================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_cols = customer_features.select_dtypes(
    include=["number"]
).columns

customer_features[numeric_cols] = numeric_pipeline.fit_transform(
    customer_features[numeric_cols]
)

# ==========================
# RFM Calculation
# ==========================

snapshot_date = (
    df["TransactionStartTime"].max()
    + pd.Timedelta(days=1)
)

rfm = df.groupby("CustomerId").agg({
    "TransactionStartTime":
        lambda x: (snapshot_date - x.max()).days,
    "TransactionId": "count",
    "Amount": "sum"
})

rfm.columns = [
    "Recency",
    "Frequency",
    "Monetary"
]

# ==========================
# KMeans Clustering
# ==========================

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# View cluster characteristics
cluster_summary = rfm.groupby("Cluster").mean()

print("\nCluster Summary:")
print(cluster_summary)

# ==========================
# Define High Risk Cluster
# ==========================

high_risk_cluster = cluster_summary["Frequency"].idxmin()

rfm["is_high_risk"] = (
    rfm["Cluster"] == high_risk_cluster
).astype(int)

# ==========================
# Merge Target Variable
# ==========================

rfm = rfm.reset_index()

processed = customer_features.merge(
    rfm[["CustomerId", "is_high_risk"]],
    on="CustomerId",
    how="left"
)

# ==========================
# Save Processed Dataset
# ==========================

processed.to_csv(
    "data/processed/processed_data.csv",
    index=False
)

print("\nProcessed dataset saved successfully!")
print(processed.head())
