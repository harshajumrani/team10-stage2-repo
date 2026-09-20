# Clustering Analysis Deliverables

## Files in this folder

| File | Description |
|---|---|
| `clustering_analysis.py` | Script that produces all deliverables below from the preprocessed dataset |
| `cluster_selection.png` | Elbow Method + Silhouette Score analysis used to determine the optimal number of clusters |
| `kmeans_model.joblib` | Trained K-Means clustering model (load with `joblib.load("kmeans_model.joblib")`) |
| `segments_scatter.png` | Visualisation of the resulting customer segments (tenure vs. monthly charges) |
| `churn_rate_by_segment.png` | Churn rate comparison across the identified segments |
| `segment_profiles.csv` | Segment profiles with interpretive labels and churn rates |
| `clustered_dataset.csv` | Full dataset with the Segment feature added, for use in Predictive Modelling |

## Optimal Number of Clusters

The Elbow Method and Silhouette Score were both used to evaluate cluster counts from k=2 to k=10:

| k | Silhouette Score |
|---|---|
| 2 | 0.4109 |
| 3 | 0.4464 |
| **4** | **0.4761 (best)** |
| 5 | 0.4297 |
| 6 | 0.4155 |
| 7 | 0.4340 |
| 8 | 0.4341 |
| 9 | 0.4402 |
| 10 | 0.4306 |

**k = 4** was selected as the optimal number of clusters, giving the highest silhouette score.

## Segment Profiles & Labels

| Segment | Label | Avg. Tenure | Avg. Monthly Charges | Churn Rate |
|---|---|---|---|---|
| 0 | New, High-Spend (At-Risk) | 14.8 months | $81.20 | **49.2%** |
| 1 | New, Low-Spend | 10.5 months | $32.50 | 24.5% |
| 2 | Loyal, High-Spend | 58.8 months | $93.10 | 15.7% |
| 3 | Loyal, Low-Spend (Stable) | 54.1 months | $34.20 | 4.8% |

## Key Insight

Customers in the **"New, High-Spend (At-Risk)"** segment (newer customers on higher monthly plans) churn at nearly 10x the rate of the **"Loyal, Low-Spend (Stable)"** segment (49.2% vs. 4.8%). This suggests tenure and monthly charge level are likely to be strong predictors in the upcoming Predictive Modelling stage, and this segment should be prioritised in retention recommendations.

Note: the Churn variable itself was **not** used to form the clusters — clustering was based only on tenure and monthly charges. Churn rate was calculated per segment afterward, purely for interpretation.

## How to Run

```bash
cd Clustering_Analysis
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python clustering_analysis.py
```

Requires `../Data_Preparation/preprocessed_dataset.csv` to exist first (run Data Preparation stage before this).
