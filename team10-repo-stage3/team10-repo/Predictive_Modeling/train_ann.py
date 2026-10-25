"""
Stage 3 - ANN churn prediction model: architecture, training and evaluation
Owner: Harsha Jumrani (Data Analyst - Predictive Modelling)
ACS WILDA - Customer Churn Analysis for Telecommunications Company (Team 10)

Step 2 of 4 (run prepare_model_data.py first)

Inputs  : model_{train,val,test}.csv, feature_columns.json
Outputs : churn_ann_model.keras          trained model (load with keras.models.load_model)
          architecture_search.csv        validation results for the candidate architectures
          training_history.png           loss / AUC curves
          confusion_matrix.png, roc_pr_curves.png
          evaluation_metrics.json        test-set metrics, baseline, cross-validation
          test_predictions.csv           churn probability + prediction per test customer
          model_summary.txt              layer-by-layer summary

Run:
    pip install tensorflow pandas numpy scikit-learn matplotlib
    cd Predictive_Modeling
    python train_ann.py
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

HERE = Path(__file__).resolve().parent
DATA = HERE   # outputs of prepare_model_data.py live in this folder
SEED = 42
TARGET = "Churn_enc"

keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

FEATURES = json.loads((DATA / "feature_columns.json").read_text())
train = pd.read_csv(DATA / "model_train.csv")
val = pd.read_csv(DATA / "model_val.csv")
test = pd.read_csv(DATA / "model_test.csv")
X_tr, y_tr = train[FEATURES].values.astype("float32"), train[TARGET].values
X_va, y_va = val[FEATURES].values.astype("float32"), val[TARGET].values
X_te, y_te = test[FEATURES].values.astype("float32"), test[TARGET].values
print(f"Train {X_tr.shape}  Val {X_va.shape}  Test {X_te.shape}")

# Class imbalance (~26.5% churn): weight the minority class more heavily so the
# network does not just learn to predict "No churn" for everyone.
cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
CLASS_WEIGHT = {0: float(cw[0]), 1: float(cw[1])}
print("Class weights:", CLASS_WEIGHT)


# ------------------------------------------------------------------
# 1. ARCHITECTURE
# ------------------------------------------------------------------
def build_model(hidden=(32, 16), dropout=0.3, l2=1e-4, lr=1e-3, n_inputs=None):
    """Feed-forward ANN: Input -> [Dense(ReLU) -> Dropout] x N -> Dense(1, sigmoid)."""
    inputs = keras.Input(shape=(n_inputs or len(FEATURES),), name="customer_features")
    x = inputs
    for i, units in enumerate(hidden, start=1):
        x = keras.layers.Dense(units, activation="relu",
                               kernel_initializer="he_normal",
                               kernel_regularizer=keras.regularizers.l2(l2),
                               name=f"hidden_{i}")(x)
        x = keras.layers.Dropout(dropout, name=f"dropout_{i}")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="churn_probability")(x)
    model = keras.Model(inputs, outputs, name="churn_ann")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"),
                           keras.metrics.BinaryAccuracy(name="accuracy")])
    return model


def fit(model, X, y, Xv, yv, verbose=0):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=20,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=8, min_lr=1e-5),
    ]
    return model.fit(X, y, validation_data=(Xv, yv), epochs=200, batch_size=32,
                     class_weight=CLASS_WEIGHT, callbacks=callbacks,
                     verbose=verbose)


# ------------------------------------------------------------------
# 2. ARCHITECTURE SEARCH (validation set only - test set untouched)
# ------------------------------------------------------------------
CANDIDATES = {
    "A: 1 hidden layer (16)":          dict(hidden=(16,), dropout=0.2),
    "B: 2 hidden layers (32-16)":      dict(hidden=(32, 16), dropout=0.3),
    "C: 2 hidden layers (64-32)":      dict(hidden=(64, 32), dropout=0.3),
    "D: 3 hidden layers (64-32-16)":   dict(hidden=(64, 32, 16), dropout=0.3),
}
rows = []
for name, cfg in CANDIDATES.items():
    keras.utils.set_random_seed(SEED)
    m = build_model(**cfg)
    h = fit(m, X_tr, y_tr, X_va, y_va)
    p = m.predict(X_va, verbose=0).ravel()
    rows.append({"architecture": name, "params": m.count_params(),
                 "epochs_trained": len(h.history["loss"]),
                 "val_auc": round(roc_auc_score(y_va, p), 4),
                 "val_pr_auc": round(average_precision_score(y_va, p), 4)})
    print(rows[-1])
search = pd.DataFrame(rows)
search.to_csv(HERE / "architecture_search.csv", index=False)

# Pick the best validation AUC; if within 0.002 of a simpler model, prefer simpler.
search = search.sort_values("params").reset_index(drop=True)
best_auc = search["val_auc"].max()
chosen = search[search["val_auc"] >= best_auc - 0.002].iloc[0]["architecture"]
CFG = CANDIDATES[chosen]
print(f"\nChosen architecture: {chosen}  {CFG}")

# ------------------------------------------------------------------
# 3. TRAIN FINAL MODEL
# ------------------------------------------------------------------
keras.utils.set_random_seed(SEED)
model = build_model(**CFG)
history = fit(model, X_tr, y_tr, X_va, y_va, verbose=0)
print(f"Trained for {len(history.history['loss'])} epochs "
      f"(best weights restored by early stopping)")
model.save(HERE / "churn_ann_model.keras")

with open(HERE / "model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda s, **k: f.write(s + "\n"))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="validation")
ax[0].set_title("Loss (binary cross-entropy)"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["auc"], label="train")
ax[1].plot(history.history["val_auc"], label="validation")
ax[1].set_title("ROC AUC"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig(HERE / "training_history.png", dpi=130); plt.close()

# ------------------------------------------------------------------
# 4. DECISION THRESHOLD (tuned on validation, maximise F1)
# ------------------------------------------------------------------
p_va = model.predict(X_va, verbose=0).ravel()
grid = np.round(np.arange(0.20, 0.81, 0.01), 2)
f1s = [f1_score(y_va, (p_va >= t).astype(int)) for t in grid]
THRESHOLD = float(grid[int(np.argmax(f1s))])
print("Decision threshold (max validation F1):", THRESHOLD)

# ------------------------------------------------------------------
# 5. EVALUATE ON THE HELD-OUT TEST SET
# ------------------------------------------------------------------
def metrics(y, p, t):
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"threshold": t,
            "accuracy": round(accuracy_score(y, yhat), 4),
            "precision": round(precision_score(y, yhat), 4),
            "recall": round(recall_score(y, yhat), 4),
            "f1": round(f1_score(y, yhat), 4),
            "specificity": round(tn / (tn + fp), 4),
            "roc_auc": round(roc_auc_score(y, p), 4),
            "pr_auc": round(average_precision_score(y, p), 4),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp),
                                 "FN": int(fn), "TP": int(tp)}}

p_te = model.predict(X_te, verbose=0).ravel()
ann_test = metrics(y_te, p_te, THRESHOLD)
ann_test_05 = metrics(y_te, p_te, 0.5)

# Baselines for context
majority = {"accuracy": round(1 - y_te.mean(), 4), "recall": 0.0,
            "note": "always predict 'No churn'"}
lr_model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr, y_tr)
p_lr_va = lr_model.predict_proba(X_va)[:, 1]
t_lr = float(grid[int(np.argmax([f1_score(y_va, (p_lr_va >= t).astype(int)) for t in grid]))])
lr_test = metrics(y_te, lr_model.predict_proba(X_te)[:, 1], t_lr)

# ------------------------------------------------------------------
# 6. ROBUSTNESS: 5-fold stratified cross-validation on train+val
# ------------------------------------------------------------------
X_all = np.vstack([X_tr, X_va]); y_all = np.concatenate([y_tr, y_va])
cv_auc, cv_f1, cv_rec = [], [], []
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_all, y_all)):
    keras.utils.set_random_seed(SEED + k)
    m = build_model(**CFG)
    fit(m, X_all[a], y_all[a], X_all[b], y_all[b])
    p = m.predict(X_all[b], verbose=0).ravel()
    cv_auc.append(roc_auc_score(y_all[b], p))
    cv_f1.append(f1_score(y_all[b], (p >= THRESHOLD).astype(int)))
    cv_rec.append(recall_score(y_all[b], (p >= THRESHOLD).astype(int)))
cv = {"roc_auc_mean": round(np.mean(cv_auc), 4), "roc_auc_std": round(np.std(cv_auc), 4),
      "f1_mean": round(np.mean(cv_f1), 4), "f1_std": round(np.std(cv_f1), 4),
      "recall_mean": round(np.mean(cv_rec), 4), "recall_std": round(np.std(cv_rec), 4),
      "fold_auc": [round(x, 4) for x in cv_auc]}

results = {"chosen_architecture": chosen, "config": {**CFG, "hidden": list(CFG["hidden"])},
           "epochs_trained": len(history.history["loss"]),
           "class_weight": CLASS_WEIGHT,
           "ann_test": ann_test, "ann_test_threshold_0.5": ann_test_05,
           "logistic_regression_test": lr_test, "majority_baseline_test": majority,
           "cross_validation_5fold": cv}
(HERE / "evaluation_metrics.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))

# ------------------------------------------------------------------
# 7. PLOTS + PREDICTIONS FILE
# ------------------------------------------------------------------
cm = np.array([[ann_test["confusion_matrix"]["TN"], ann_test["confusion_matrix"]["FP"]],
               [ann_test["confusion_matrix"]["FN"], ann_test["confusion_matrix"]["TP"]]])
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["Stay", "Churn"]); ax.set_yticks([0, 1], ["Stay", "Churn"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"ANN confusion matrix (test, t={THRESHOLD})")
plt.tight_layout(); plt.savefig(HERE / "confusion_matrix.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
fpr, tpr, _ = roc_curve(y_te, p_te)
ax[0].plot(fpr, tpr, label=f"ANN (AUC={ann_test['roc_auc']:.3f})")
ax[0].plot([0, 1], [0, 1], "--", color="grey", label="Random")
ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
ax[0].set_title("ROC curve (test set)"); ax[0].legend()
prec, rec, _ = precision_recall_curve(y_te, p_te)
ax[1].plot(rec, prec, label=f"ANN (AP={ann_test['pr_auc']:.3f})")
ax[1].axhline(y_te.mean(), ls="--", color="grey", label="Base rate")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[1].set_title("Precision-recall curve (test set)"); ax[1].legend()
plt.tight_layout(); plt.savefig(HERE / "roc_pr_curves.png", dpi=130); plt.close()

pred = test.drop(columns=FEATURES).copy()
pred["churn_probability"] = p_te.round(4)
pred["predicted_churn"] = (p_te >= THRESHOLD).astype(int)
pred.to_csv(HERE / "test_predictions.csv", index=False)
print("Saved model, metrics, plots and test_predictions.csv")
