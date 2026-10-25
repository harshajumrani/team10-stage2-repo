"""
Data Preparation - Stage 2 Deliverable
ACS WILDA - Customer Churn Analysis for Telecommunications Company (Team 10)

Produces:
- preprocessed_dataset.csv (missing data handled, categorical variables encoded)
- train_set.csv / test_set.csv (train/test split for model validation)
- scaling applied and documented (see Scaling_Techniques.docx)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

RAW_PATH = "../Dataset_ATS_v2.csv"
RANDOM_STATE = 42

# ============================================================
# 1. LOAD & INSPECT
# ============================================================
df = pd.read_csv(RAW_PATH)
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nChurn balance:\n", df["Churn"].value_counts(normalize=True).round(3))

# ============================================================
# 2. HANDLE MISSING DATA POINTS
# ============================================================
# No missing values were found in this dataset during profiling, but this
# step is included to keep the pipeline robust for any future data refresh.
num_cols_raw = ["tenure", "MonthlyCharges"]
df[num_cols_raw] = df[num_cols_raw].apply(pd.to_numeric, errors="coerce")
df[num_cols_raw] = df[num_cols_raw].fillna(df[num_cols_raw].median())

cat_cols = ["gender", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "Contract"]
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# ============================================================
# 3. ENCODE CATEGORICAL VARIABLES
# ============================================================
binary_cols = ["gender", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", "Churn"]
label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

print("\nLabel encoding map:")
for col, mapping in label_encoders.items():
    print(f"  {col}: {mapping}")

# Contract (3 categories) -> one-hot encoding
df = pd.get_dummies(df, columns=["Contract"], prefix="Contract", drop_first=True)

# ============================================================
# 4. FEATURE SCALING / NORMALISATION
# ============================================================
# StandardScaler standardises tenure and MonthlyCharges to zero mean /
# unit variance, which improves convergence for distance-based methods
# (K-Means clustering) and gradient-based models (the ANN).
scale_cols = ["tenure", "MonthlyCharges"]
scaler = StandardScaler()
df[[c + "_scaled" for c in scale_cols]] = scaler.fit_transform(df[scale_cols])

# ============================================================
# 5. BUILD FINAL MODELLING FEATURE SET
# ============================================================
model_feature_cols = (
    ["SeniorCitizen", "tenure_scaled", "MonthlyCharges_scaled",
     "gender_enc", "Dependents_enc", "PhoneService_enc",
     "MultipleLines_enc", "InternetService_enc"]
    + [c for c in df.columns if c.startswith("Contract_")]
)
print("\nFinal feature columns for modelling:", model_feature_cols)

# ============================================================
# 6. TRAIN / TEST SPLIT (for model validation)
# ============================================================
X = df[model_feature_cols]
y = df["Churn_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.3f}, Test churn rate: {y_test.mean():.3f}")

# ============================================================
# 7. SAVE DELIVERABLES
# ============================================================
df.to_csv("preprocessed_dataset.csv", index=False)
print("\nSaved preprocessed_dataset.csv")

X_train.assign(Churn=y_train.values).to_csv("train_set.csv", index=False)
X_test.assign(Churn=y_test.values).to_csv("test_set.csv", index=False)
print("Saved train_set.csv and test_set.csv")

with open("model_feature_cols.txt", "w") as f:
    f.write("\n".join(model_feature_cols))
print("Saved model_feature_cols.txt")

# Save a small composition summary for documentation
with open("train_test_composition.txt", "w") as f:
    f.write(f"Total records: {len(df)}\n")
    f.write(f"Train set: {len(X_train)} records ({len(X_train)/len(df)*100:.1f}%), churn rate {y_train.mean()*100:.1f}%\n")
    f.write(f"Test set: {len(X_test)} records ({len(X_test)/len(df)*100:.1f}%), churn rate {y_test.mean()*100:.1f}%\n")
    f.write(f"Split method: stratified 80/20 split (random_state={RANDOM_STATE}) preserving churn ratio\n")
print("Saved train_test_composition.txt")
