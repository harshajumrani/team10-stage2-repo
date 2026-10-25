"""
Stage 3 - Customer segments x ANN churn predictions
Owner: Mitul (Cluster Analyst)
ACS WILDA - Customer Churn Analysis for Telecommunications Company (Team 10)

Links the Stage 2 K-Means segments to the Stage 3 ANN predictions on the
held-out test set (1,409 customers the model never saw during training):

  - actual vs predicted churn rate per segment
  - how well the ANN catches churners inside each segment (recall / precision)
  - risk tiers (Low / Medium / High churn probability) per segment

Step 3 of 4 (run train_ann.py first)

Inputs : test_predictions.csv, evaluation_metrics.json
Outputs: segment_prediction_summary.csv, risk_tier_by_segment.csv,
         segment_actual_vs_predicted.png, risk_tier_by_segment.png

Run:
    cd Predictive_Modeling
    python segment_churn_analysis.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ANN = HERE    # outputs of train_ann.py live in this folder

pred = pd.read_csv(ANN / "test_predictions.csv")
THRESHOLD = json.loads((ANN / "evaluation_metrics.json").read_text())["ann_test"]["threshold"]
ORDER = ["New, High-Spend (At-Risk)", "New, Low-Spend",
         "Loyal, High-Spend", "Loyal, Low-Spend (Stable)"]

# ------------------------------------------------------------------
# 1. Per-segment summary
# ------------------------------------------------------------------
rows = []
for seg in ORDER:
    s = pred[pred["Segment_Label"] == seg]
    y, p, yhat = s["Churn_enc"], s["churn_probability"], s["predicted_churn"]
    rows.append({
        "segment": seg,
        "customers": len(s),
        "share_of_test_%": round(len(s) / len(pred) * 100, 1),
        "actual_churn_rate_%": round(y.mean() * 100, 1),
        "mean_predicted_probability_%": round(p.mean() * 100, 1),
        "flagged_as_churn": int(yhat.sum()),
        "flagged_%": round(yhat.mean() * 100, 1),
        "actual_churners": int(y.sum()),
        "churners_caught": int(((yhat == 1) & (y == 1)).sum()),
        "recall_%": round(recall_score(y, yhat, zero_division=0) * 100, 1),
        "precision_%": round(precision_score(y, yhat, zero_division=0) * 100, 1),
        "within_segment_auc": round(roc_auc_score(y, p), 3) if y.nunique() == 2 else None,
        "share_of_all_churners_%": round(y.sum() / pred["Churn_enc"].sum() * 100, 1),
    })
summary = pd.DataFrame(rows)
summary.to_csv(HERE / "segment_prediction_summary.csv", index=False)
print(summary.to_string(index=False))

# ------------------------------------------------------------------
# 2. Risk tiers
# ------------------------------------------------------------------
bins = [0, 0.30, THRESHOLD, 1.0001]
labels = ["Low (<30%)", f"Medium (30-{int(THRESHOLD*100)}%)", f"High (>={int(THRESHOLD*100)}%)"]
pred["risk_tier"] = pd.cut(pred["churn_probability"], bins=bins, labels=labels, right=False)
tiers = (pred.groupby(["Segment_Label", "risk_tier"], observed=False).size()
         .unstack(fill_value=0).reindex(ORDER))
tier_churn = (pred.groupby("risk_tier", observed=False)["Churn_enc"]
              .agg(["size", "mean"]).rename(columns={"size": "customers", "mean": "actual_churn_rate"}))
tier_churn["actual_churn_rate"] = (tier_churn["actual_churn_rate"] * 100).round(1)
tiers.to_csv(HERE / "risk_tier_by_segment.csv")
tier_churn.to_csv(HERE / "risk_tier_actual_churn.csv")
print("\n", tiers, "\n\n", tier_churn)

# ------------------------------------------------------------------
# 3. Charts
# ------------------------------------------------------------------
short = ["New\nHigh-Spend", "New\nLow-Spend", "Loyal\nHigh-Spend", "Loyal\nLow-Spend"]
x = np.arange(len(ORDER)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w/2, summary["actual_churn_rate_%"], w, label="Actual churn rate", color="#3b6ea5")
b2 = ax.bar(x + w/2, summary["mean_predicted_probability_%"], w,
            label="ANN mean predicted probability", color="#e8a33d")
ax.bar_label(b1, fmt="%.1f%%", fontsize=9); ax.bar_label(b2, fmt="%.1f%%", fontsize=9)
ax.set_xticks(x, short); ax.set_ylabel("%")
ax.set_title("Actual vs ANN-predicted churn by customer segment (test set)")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(HERE / "segment_actual_vs_predicted.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#9bc59d", "#f2c14e", "#d1495b"]
left = np.zeros(len(ORDER))
pct = tiers.div(tiers.sum(axis=1), axis=0) * 100
for col, c in zip(pct.columns, colors):
    ax.barh(short, pct[col].values, left=left, color=c, label=col)
    for i, v in enumerate(pct[col].values):
        if v > 6:
            ax.text(left[i] + v/2, i, f"{v:.0f}%", ha="center", va="center", fontsize=9)
    left += pct[col].values
ax.invert_yaxis(); ax.set_xlabel("% of customers in segment")
ax.set_title("ANN churn-risk tiers within each segment (test set)")
ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(HERE / "risk_tier_by_segment.png", dpi=130); plt.close()
print("Saved charts")
