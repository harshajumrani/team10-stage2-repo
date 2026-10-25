"""
Stage 3 - Factors contributing to churn and retention (based on the ANN)
Owner: Ashrafun (Business Analyst)
ACS WILDA - Customer Churn Analysis for Telecommunications Company (Team 10)

Three complementary views:
  1. Permutation importance - how much the ANN's test ROC AUC drops when a
     customer attribute is shuffled (bigger drop = the model relies on it more).
  2. Effect profiles - the ANN's average predicted churn probability as tenure
     and monthly charges are varied, holding every other attribute as it is
     (partial dependence).
  3. Observed churn rates by attribute in the full dataset, to check the
     model's story against the raw data.

Step 4 of 4 (run train_ann.py first)

Inputs : churn_ann_model.keras, model_test.csv, scaler.joblib, feature_columns.json
         ../Clustering_Analysis/kmeans_model.joblib, ../Dataset_ATS_v2.csv
Outputs: feature_importance.csv / .png, effect_tenure_charges.png,
         attribute_effects.csv, churn_rate_by_attribute.csv / .png

Run:
    cd Predictive_Modeling
    python churn_factors_analysis.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
DATA = HERE
ANN = HERE
REPO = HERE.parent
SEED = 42
rng = np.random.default_rng(SEED)

model = keras.models.load_model(ANN / "churn_ann_model.keras")
FEATURES = json.loads((DATA / "feature_columns.json").read_text())
scaler = joblib.load(DATA / "scaler.joblib")
test = pd.read_csv(DATA / "model_test.csv")
X = test[FEATURES].values.astype("float32"); y = test["Churn_enc"].values


def predict(arr):
    return model.predict(arr, verbose=0, batch_size=2048).ravel()


base_auc = roc_auc_score(y, predict(X))
print("Baseline test AUC:", round(base_auc, 4))

# ------------------------------------------------------------------
# 1. Permutation importance (grouped by business attribute)
# ------------------------------------------------------------------
GROUPS = {
    "Tenure (months with company)": ["tenure_scaled"],
    "Monthly charges": ["MonthlyCharges_scaled"],
    "Customer segment (Stage 2)": ["Segment_0", "Segment_1", "Segment_2", "Segment_3"],
    "Senior citizen": ["SeniorCitizen"],
    "Has dependents": ["Dependents_enc"],
    "Contract type": ["Contract_One year", "Contract_Two year"],
    "Internet service type": ["InternetService_enc"],
    "Phone service": ["PhoneService_enc"],
    "Multiple lines": ["MultipleLines_enc"],
    "Gender": ["gender_enc"],
}
REPEATS = 20
rows = []
for name, cols in GROUPS.items():
    idx = [FEATURES.index(c) for c in cols]
    drops = []
    for _ in range(REPEATS):
        Xp = X.copy()
        perm = rng.permutation(len(Xp))
        Xp[:, idx] = Xp[perm][:, idx]          # shuffle the group together
        drops.append(base_auc - roc_auc_score(y, predict(Xp)))
    rows.append({"attribute": name, "auc_drop_mean": round(np.mean(drops), 4),
                 "auc_drop_std": round(np.std(drops), 4)})
imp = pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False)
imp["relative_importance_%"] = (imp["auc_drop_mean"].clip(lower=0)
                                / imp["auc_drop_mean"].clip(lower=0).sum() * 100).round(1)
imp.to_csv(HERE / "feature_importance.csv", index=False)
print(imp.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4.8))
d = imp.iloc[::-1]
ax.barh(d["attribute"], d["auc_drop_mean"], xerr=d["auc_drop_std"], color="#3b6ea5")
ax.set_xlabel("Drop in test ROC AUC when attribute is shuffled")
ax.set_title("What the ANN relies on to predict churn (permutation importance)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(HERE / "feature_importance.png", dpi=130); plt.close()

# ------------------------------------------------------------------
# 2. Effect profiles: tenure and monthly charges
# ------------------------------------------------------------------
# NOTE: Segment dummies are derived from tenure + charges. When we vary tenure
# or charges we also reassign the segment with the Stage 2 K-Means model so the
# inputs stay internally consistent.
kmeans = joblib.load(REPO / "Clustering_Analysis" / "kmeans_model.joblib")
full = pd.read_csv(REPO / "Clustering_Analysis" / "clustered_dataset.csv")
# Stage 2 K-Means was fitted on full-data z-scores
mu_t, sd_t = full["tenure"].mean(), full["tenure"].std(ddof=0)
mu_m, sd_m = full["MonthlyCharges"].mean(), full["MonthlyCharges"].std(ddof=0)
seg_idx = [FEATURES.index(f"Segment_{k}") for k in range(4)]
ti, mi = FEATURES.index("tenure_scaled"), FEATURES.index("MonthlyCharges_scaled")


def set_raw(Xc, tenure=None, charges=None):
    t = test["tenure"].values if tenure is None else np.full(len(Xc), tenure)
    m = test["MonthlyCharges"].values if charges is None else np.full(len(Xc), charges)
    z = scaler.transform(pd.DataFrame({"tenure": t, "MonthlyCharges": m}))
    Xc[:, ti], Xc[:, mi] = z[:, 0], z[:, 1]
    seg = kmeans.predict(np.column_stack([(t - mu_t) / sd_t, (m - mu_m) / sd_m]))
    Xc[:, seg_idx] = 0
    Xc[np.arange(len(Xc)), np.array(seg_idx)[seg]] = 1
    return Xc


t_grid = np.arange(0, 73, 3)
m_grid = np.arange(20, 121, 5)
pd_t = [predict(set_raw(X.copy(), tenure=t)).mean() for t in t_grid]
pd_m = [predict(set_raw(X.copy(), charges=m)).mean() for m in m_grid]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(t_grid, np.array(pd_t) * 100, marker="o", ms=3, color="#d1495b")
ax[0].set_xlabel("Tenure (months)"); ax[0].set_ylabel("Avg predicted churn probability (%)")
ax[0].set_title("Churn risk falls as tenure grows")
ax[1].plot(m_grid, np.array(pd_m) * 100, marker="o", ms=3, color="#3b6ea5")
ax[1].set_xlabel("Monthly charges ($)")
ax[1].set_title("Churn risk rises with monthly charges")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(HERE / "effect_tenure_charges.png", dpi=130); plt.close()
pd.DataFrame({"tenure": t_grid, "avg_pred_prob": np.round(pd_t, 4)}).to_csv(
    HERE / "effect_tenure.csv", index=False)
pd.DataFrame({"monthly_charges": m_grid, "avg_pred_prob": np.round(pd_m, 4)}).to_csv(
    HERE / "effect_charges.csv", index=False)

# Binary attribute effects: set attribute to 0 vs 1 for everyone, compare
BIN = {"Senior citizen": "SeniorCitizen", "Has dependents": "Dependents_enc",
       "Fiber optic (vs DSL)": "InternetService_enc", "Phone service": "PhoneService_enc",
       "Multiple lines": "MultipleLines_enc", "Male (vs female)": "gender_enc"}
eff = []
for name, col in BIN.items():
    i = FEATURES.index(col)
    X0, X1 = X.copy(), X.copy(); X0[:, i] = 0; X1[:, i] = 1
    p0, p1 = float(predict(X0).mean()), float(predict(X1).mean())
    eff.append({"attribute": name, "pred_prob_if_no_%": round(p0 * 100, 1),
                "pred_prob_if_yes_%": round(p1 * 100, 1),
                "difference_pts": round((p1 - p0) * 100, 1)})
ci = [FEATURES.index("Contract_One year"), FEATURES.index("Contract_Two year")]
for label, v in [("Month-to-month", (0, 0)), ("One year", (1, 0)), ("Two year", (0, 1))]:
    Xc = X.copy(); Xc[:, ci[0]], Xc[:, ci[1]] = v
    eff.append({"attribute": f"Contract = {label}", "pred_prob_if_no_%": None,
                "pred_prob_if_yes_%": round(float(predict(Xc).mean()) * 100, 1),
                "difference_pts": None})
eff = pd.DataFrame(eff)
eff.to_csv(HERE / "attribute_effects.csv", index=False)
print("\n", eff.to_string(index=False))

# ------------------------------------------------------------------
# 3. Observed churn rates in the raw data (sanity check)
# ------------------------------------------------------------------
raw = pd.read_csv(REPO / "Dataset_ATS_v2.csv")
raw["churn"] = (raw["Churn"] == "Yes").astype(int)
raw["SeniorCitizen"] = raw["SeniorCitizen"].map({0: "No", 1: "Yes"})
raw["Tenure band"] = pd.cut(raw["tenure"], [-1, 6, 12, 24, 48, 72],
                            labels=["0-6 m", "7-12 m", "13-24 m", "25-48 m", "49-72 m"])
raw["Charges band"] = pd.cut(raw["MonthlyCharges"], [0, 30, 50, 70, 90, 120],
                             labels=["<$30", "$30-50", "$50-70", "$70-90", "$90+"])
out = []
for col in ["Tenure band", "Charges band", "SeniorCitizen", "Dependents", "Contract",
            "InternetService", "PhoneService", "MultipleLines", "gender"]:
    g = raw.groupby(col, observed=True)["churn"].agg(["size", "mean"]).reset_index()
    for _, r in g.iterrows():
        out.append({"attribute": col, "value": r[col], "customers": int(r["size"]),
                    "churn_rate_%": round(r["mean"] * 100, 1)})
obs = pd.DataFrame(out)
obs.to_csv(HERE / "churn_rate_by_attribute.csv", index=False)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for a, col in zip(axes, ["Tenure band", "Charges band", "SeniorCitizen", "Dependents"]):
    s = obs[obs["attribute"] == col]
    bars = a.bar(s["value"].astype(str), s["churn_rate_%"], color="#3b6ea5")
    a.bar_label(bars, fmt="%.0f%%", fontsize=8)
    a.set_title(col if col != "SeniorCitizen" else "Senior citizen")
    a.axhline(raw["churn"].mean() * 100, ls="--", color="grey", lw=1)
    a.tick_params(axis="x", rotation=30); a.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Observed churn rate (%)")
plt.suptitle("Observed churn rate by attribute (dashed line = overall 26.5%)", y=1.02)
plt.tight_layout(); plt.savefig(HERE / "churn_rate_by_attribute.png", dpi=130,
                                bbox_inches="tight"); plt.close()
print("Saved all outputs")
