# Predictive Modeling Deliverables

## Files in this folder

| File | Description | Owner |
|---|---|---|
| `prepare_model_data.py` | Step 1 — builds model-ready train / validation / test sets from the Stage 2 clustered dataset | Pruthal |
| `model_train.csv`, `model_val.csv`, `model_test.csv` | Training (4,788), validation (846) and test (1,409) sets with the 14 model features | Pruthal |
| `scaler.joblib` | StandardScaler for tenure and monthly charges, fitted on training rows only | Pruthal |
| `feature_columns.json`, `split_summary.json` | Feature list and split composition | Pruthal |
| `Model_Data_Pipeline.docx` | Documentation of the model data pipeline | Pruthal |
| `train_ann.py` | Step 2 — defines, optimises, trains and evaluates the ANN | Harsha |
| `ANN_Architecture.docx` | **Defined architecture of the ANN model** (input, hidden and output layers), training set-up and evaluation | Harsha |
| `churn_ann_model.keras` | **Trained ANN model** (load with `keras.models.load_model("churn_ann_model.keras")`) | Harsha |
| `model_summary.txt` | Layer-by-layer model summary | Harsha |
| `architecture_search.csv` | Validation results for the four candidate architectures | Harsha |
| `training_history.png` | Training vs validation loss and ROC AUC by epoch | Harsha |
| `evaluation_metrics.json` | Test-set metrics, baselines and 5-fold cross-validation | Harsha |
| `confusion_matrix.png`, `roc_pr_curves.png` | Test-set confusion matrix, ROC and precision-recall curves | Harsha |
| `test_predictions.csv` | Churn probability and prediction for every test customer | Harsha |
| `segment_churn_analysis.py` | Step 3 — ANN predictions broken down by Stage 2 customer segment | Mitul |
| `segment_prediction_summary.csv` | Actual vs predicted churn, recall and precision per segment | Mitul |
| `risk_tier_by_segment.csv`, `risk_tier_actual_churn.csv` | Low / Medium / High risk tiers per segment and their real churn rates | Mitul |
| `segment_actual_vs_predicted.png`, `risk_tier_by_segment.png` | Segment-level charts | Mitul |
| `churn_factors_analysis.py` | Step 4 — factors contributing to churn and retention | Ashrafun |
| `feature_importance.csv` / `.png` | Permutation importance of each customer attribute | Ashrafun |
| `effect_tenure_charges.png`, `effect_tenure.csv`, `effect_charges.csv` | How predicted churn changes with tenure and monthly charges | Ashrafun |
| `attribute_effects.csv` | Effect of each yes/no attribute and contract type on predicted churn | Ashrafun |
| `churn_rate_by_attribute.csv` / `.png` | Observed churn rate by attribute in the raw data | Ashrafun |

## ANN Architecture

| Layer | Units | Activation | Details |
|---|---|---|---|
| Input | 14 | – | Scaled tenure & monthly charges, 8 binary/one-hot fields, 4 segment flags |
| Hidden 1 | 64 | ReLU | He-normal init, L2 = 0.0001, Dropout 0.3 |
| Hidden 2 | 32 | ReLU | He-normal init, L2 = 0.0001, Dropout 0.3 |
| Output | 1 | Sigmoid | Churn probability |

3,073 trainable parameters. Optimiser Adam (lr 0.001), binary cross-entropy loss, balanced class weights (churn = 26.5%), early stopping on validation AUC, ReduceLROnPlateau.

## Architecture Selection

Four candidate architectures were compared on the validation set (the test set was not used):

| Architecture | Parameters | Validation ROC AUC |
|---|---|---|
| 1 hidden layer (16) | 257 | 0.811 |
| 2 hidden layers (32-16) | 1,025 | 0.808 |
| **2 hidden layers (64-32)** | **3,073** | **0.814 (best)** |
| 3 hidden layers (64-32-16) | 3,585 | 0.813 |

**64-32** was selected. Decision threshold **0.66** was chosen on the validation set to maximise F1.

## Model Performance (test set, 1,409 unseen customers)

| Metric | ANN (t = 0.66) | ANN (t = 0.50) | Logistic regression | Always "No churn" |
|---|---|---|---|---|
| ROC AUC | **0.805** | 0.805 | 0.805 | 0.5 |
| Accuracy | **77.2%** | 71.8% | 78.6% | 73.5% |
| Recall | **57.8%** | 73.8% | 50.8% | 0% |
| Precision | **56.8%** | 47.9% | 61.9% | – |
| F1 | **0.573** | 0.581 | 0.558 | 0 |

5-fold cross-validated ROC AUC: **0.824 ± 0.005**, so results are stable across data splits.

## Segment Results

| Segment | Actual churn | Share of all churners | Churners caught by ANN |
|---|---|---|---|
| New, High-Spend (At-Risk) | 50.0% | **59.4%** | 77.9% |
| New, Low-Spend | 23.3% | 21.7% | 48.1% |
| Loyal, High-Spend | 14.7% | 16.0% | 6.7% |
| Loyal, Low-Spend (Stable) | 5.2% | 2.9% | 0% |

| ANN risk tier | Customers | Actual churn rate |
|---|---|---|
| High (≥ 66%) | 380 | **56.8%** |
| Medium (30–66%) | 506 | 25.1% |
| Low (< 30%) | 523 | 5.9% |

## Key Insight

**Tenure (63.5%) and monthly charges (30.5%)** account for about 94% of the ANN's predictive power. Predicted churn falls from about 75% at month 0 to 11% at 6 years, and rises from about 19% at $20/month to 69% at $120/month. Senior citizens and customers without dependents are also at higher risk; contract type, internet service type and gender make little difference in this dataset. Customers the ANN rates high-risk churn at nearly **10x** the rate of low-risk customers (56.8% vs 5.9%), so the model is useful for targeting retention effort.

Note: the scaler is re-fitted on training rows only (Stage 2 fitted it on all rows), and the 1,409-customer test set is identical to `Data_Preparation/test_set.csv`.

## How to Run

```bash
cd Predictive_Modeling
pip install -r ../requirements.txt
python prepare_model_data.py
python train_ann.py              # ~3-5 min on CPU
python segment_churn_analysis.py
python churn_factors_analysis.py
```

Requires `../Clustering_Analysis/clustered_dataset.csv` and `../Clustering_Analysis/kmeans_model.joblib` to exist first (run the Stage 2 scripts before this). Results may differ very slightly on other TensorFlow versions or hardware; seeds are fixed and the trained model file is included so the reported numbers can be reproduced from it.
