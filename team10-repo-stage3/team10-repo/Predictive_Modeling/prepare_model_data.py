"""
Stage 3 - Data Engineering for the ANN model
Owner: Pruthal (Data Engineer)
ACS WILDA - Customer Churn Analysis for Telecommunications Company (Team 10)

What this script does
---------------------
Step 1 of 4. Takes the Stage 2 outputs and turns them into model-ready files for the ANN:

1. Loads ../Clustering_Analysis/clustered_dataset.csv (raw columns + encodings
   + the K-Means Segment from Stage 2).
2. Rebuilds the SAME stratified 80/20 train/test split used in Stage 2
   (random_state=42), so the test set is identical to Data_Preparation/test_set.csv.
3. Carves a 15% validation set out of the training data (stratified) for
   early stopping and decision-threshold tuning - the test set is never touched
   during training.
4. Re-fits StandardScaler on the TRAINING rows only and applies it to val/test.
   (Stage 2 fitted the scaler on the full dataset, which leaks a small amount of
   test information into training. Fitting on train only fixes that.)
5. Saves model_train.csv / model_val.csv / model_test.csv, the fitted scaler,
   the feature list and a data dictionary.

Run:
    cd Predictive_Modeling
    python prepare_model_data.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC = REPO / "Clustering_Analysis" / "clustered_dataset.csv"
RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------------
df = pd.read_csv(SRC)
print("Loaded:", SRC.name, df.shape)

# Data quality checks
assert df.isnull().sum().sum() == 0, "Unexpected missing values"
print("Missing values: 0")
print("Exact duplicate rows (raw columns):",
      df[["gender", "SeniorCitizen", "Dependents", "tenure", "PhoneService",
          "MultipleLines", "InternetService", "MonthlyCharges", "Churn",
          "Contract_One year", "Contract_Two year"]]
      .duplicated().sum())

# ------------------------------------------------------------------
# 2. Feature set
# ------------------------------------------------------------------
NUMERIC = ["tenure", "MonthlyCharges"]                  # scaled below
BINARY = ["SeniorCitizen", "gender_enc", "Dependents_enc", "PhoneService_enc",
          "MultipleLines_enc", "InternetService_enc",
          "Contract_One year", "Contract_Two year"]
SEGMENT = ["Segment_0", "Segment_1", "Segment_2", "Segment_3"]  # from Stage 2
FEATURES = [f"{c}_scaled" for c in NUMERIC] + BINARY + SEGMENT
TARGET = "Churn_enc"

for c in BINARY + SEGMENT:
    df[c] = df[c].astype(int)          # True/False -> 1/0

# ------------------------------------------------------------------
# 3. Split: same 80/20 test split as Stage 2, then 85/15 train/val
# ------------------------------------------------------------------
idx_trainval, idx_test = train_test_split(
    df.index, test_size=0.20, random_state=RANDOM_STATE, stratify=df[TARGET])
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.15, random_state=RANDOM_STATE,
    stratify=df.loc[idx_trainval, TARGET])

train, val, test = (df.loc[idx_train].copy(), df.loc[idx_val].copy(),
                    df.loc[idx_test].copy())

# ------------------------------------------------------------------
# 4. Scale numeric features - fit on TRAIN only
# ------------------------------------------------------------------
scaler = StandardScaler().fit(train[NUMERIC])
for part in (train, val, test):
    part[[f"{c}_scaled" for c in NUMERIC]] = scaler.transform(part[NUMERIC])

# ------------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------------
# Keep the readable columns alongside the features so later analyses
# (segments, factors) can join predictions back to customer attributes.
KEEP = ["gender", "SeniorCitizen", "Dependents", "tenure", "PhoneService",
        "MultipleLines", "InternetService", "MonthlyCharges", "Segment_Label"]
extra = [c for c in KEEP if c not in FEATURES]
for name, part in (("train", train), ("val", val), ("test", test)):
    out = part[FEATURES + [TARGET] + extra].copy()
    out.insert(0, "row_id", part.index)
    out.to_csv(HERE / f"model_{name}.csv", index=False)

joblib.dump(scaler, HERE / "scaler.joblib")
(HERE / "feature_columns.json").write_text(json.dumps(FEATURES, indent=2))

summary = {
    "total_rows": len(df),
    "train": {"rows": len(train), "churn_rate": round(train[TARGET].mean(), 4)},
    "validation": {"rows": len(val), "churn_rate": round(val[TARGET].mean(), 4)},
    "test": {"rows": len(test), "churn_rate": round(test[TARGET].mean(), 4)},
    "scaler_mean": dict(zip(NUMERIC, scaler.mean_.round(4).tolist())),
    "scaler_std": dict(zip(NUMERIC, scaler.scale_.round(4).tolist())),
    "n_features": len(FEATURES),
}
(HERE / "split_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

# Sanity check: test rows match Stage 2 test set
s2_test = pd.read_csv(REPO / "Data_Preparation" / "test_set.csv")
assert len(s2_test) == len(test) and s2_test["Churn"].sum() == test[TARGET].sum()
print("Test set matches Stage 2 test_set.csv (rows and churn count).")
