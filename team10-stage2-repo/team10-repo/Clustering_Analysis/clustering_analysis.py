"""
Clustering Analysis - Stage 2 Deliverable
ACS WILDA - Customer Churn Analysis for Telecommunications Company (Team 10)

Produces:
- cluster_selection.png (elbow method + silhouette score to find optimal k)
- kmeans_model.joblib (trained K-Means model)
- segments_scatter.png, churn_rate_by_segment.png (visualised, labelled clusters)
- segment_profiles.csv (interpretation table)
- clustered_dataset.csv (dataset + Segment feature, for the Predictive Modelling stage)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
MAX_K = 10

# ============================================================
# 1. LOAD PREPROCESSED DATA (from Data Preparation stage)
# ============================================================
df = pd.read_csv("../Data_Preparation/preprocessed_dataset.csv")
print("Shape:", df.shape)

# Cluster on scaled behavioural features - tenure and monthly charges are
# the strongest, most interpretable signal for segmenting customers here.
cluster_cols = ["tenure_scaled", "MonthlyCharges_scaled"]
X_scaled = df[cluster_cols].values

# ============================================================
# 2. IDENTIFY OPTIMAL NUMBER OF CLUSTERS (Elbow Method + Silhouette)
# ============================================================
inertias, silhouette_scores = [], []
k_range = range(2, MAX_K + 1)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster SSE)")
axes[0].set_title("Elbow Method")
axes[1].plot(list(k_range), silhouette_scores, marker="o", color="orange")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Analysis")
plt.tight_layout()
plt.savefig("cluster_selection.png", dpi=120)
print("Saved cluster_selection.png")

print("\nOptimal cluster selection results:")
for k, inertia, sil in zip(k_range, inertias, silhouette_scores):
    print(f"  k={k}: inertia={inertia:.1f}, silhouette={sil:.4f}")

best_k = list(k_range)[int(np.argmax(silhouette_scores))]
print(f"\nBest k by silhouette score: {best_k}")

# ============================================================
# 3. TRAIN FINAL K-MEANS MODEL
# ============================================================
OPTIMAL_K = best_k  # chosen via elbow method / highest silhouette score above

kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=RANDOM_STATE, n_init=10)
df["Segment"] = kmeans.fit_predict(X_scaled)
final_silhouette = silhouette_score(X_scaled, df["Segment"])
print(f"\nFinal model: k={OPTIMAL_K}, silhouette score={final_silhouette:.4f}")

# Save the trained K-Means model
joblib.dump(kmeans, "kmeans_model.joblib")
print("Saved kmeans_model.joblib")

# ============================================================
# 4. PROFILE & LABEL CLUSTERS FOR INTERPRETATION
# ============================================================
profile = df.groupby("Segment")[["tenure", "MonthlyCharges"]].mean().round(1)
profile["count"] = df.groupby("Segment").size()
profile["pct_of_total"] = (profile["count"] / len(df) * 100).round(1)
profile["churn_rate_%"] = (df.groupby("Segment")["Churn_enc"].mean() * 100).round(1)

# Assign interpretable labels based on tenure/spend/churn combination
def label_segment(row):
    tenure_high = row["tenure"] >= df["tenure"].median()
    charge_high = row["MonthlyCharges"] >= df["MonthlyCharges"].median()
    if not tenure_high and charge_high:
        return "New, High-Spend (At-Risk)"
    elif not tenure_high and not charge_high:
        return "New, Low-Spend"
    elif tenure_high and charge_high:
        return "Loyal, High-Spend"
    else:
        return "Loyal, Low-Spend (Stable)"

profile["Label"] = profile.apply(label_segment, axis=1)
print("\n=== Segment Profiles (labelled) ===")
print(profile)
profile.to_csv("segment_profiles.csv")
print("Saved segment_profiles.csv")

# ============================================================
# 5. VISUALISE CLUSTERS
# ============================================================
label_map = profile["Label"].to_dict()
df["Segment_Label"] = df["Segment"].map(label_map)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="tenure", y="MonthlyCharges", hue="Segment_Label",
                 palette="tab10", alpha=0.7)
plt.title(f"Customer Segments (k={OPTIMAL_K}) - K-Means Clustering")
plt.xlabel("Tenure (months)")
plt.ylabel("Monthly Charges ($)")
plt.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("segments_scatter.png", dpi=120)
print("Saved segments_scatter.png")

plt.figure(figsize=(7, 4))
sns.barplot(data=profile.reset_index(), x="Label", y="churn_rate_%")
plt.title("Churn Rate (%) by Customer Segment")
plt.xlabel("")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("churn_rate_by_segment.png", dpi=120)
print("Saved churn_rate_by_segment.png")

# ============================================================
# 6. SAVE FINAL CLUSTERED DATASET (for Predictive Modelling stage)
# ============================================================
df_out = pd.get_dummies(df, columns=["Segment"], prefix="Segment")
df_out.to_csv("clustered_dataset.csv", index=False)
print("\nSaved clustered_dataset.csv")

print("\n=== Key Insight ===")
highest = profile["churn_rate_%"].idxmax()
lowest = profile["churn_rate_%"].idxmin()
print(f"Segment {highest} ('{profile.loc[highest,'Label']}') has the highest churn rate: {profile.loc[highest,'churn_rate_%']}%")
print(f"Segment {lowest} ('{profile.loc[lowest,'Label']}') has the lowest churn rate: {profile.loc[lowest,'churn_rate_%']}%")
