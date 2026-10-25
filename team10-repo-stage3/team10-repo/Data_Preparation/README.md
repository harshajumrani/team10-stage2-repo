# Data Preparation Deliverables

## Files in this folder

| File | Description |
|---|---|
| `data_preparation.py` | Script that produces all deliverables below from the raw dataset |
| `preprocessed_dataset.csv` | Full dataset with missing data handled, categorical variables encoded, and numerical variables scaled |
| `train_set.csv` | Training set (80% of records, stratified by churn) |
| `test_set.csv` | Testing set (20% of records, stratified by churn) |
| `train_test_composition.txt` | Size and churn-rate composition of the train/test split |
| `model_feature_cols.txt` | List of final feature columns used for modelling |
| `Scaling_Techniques.docx` | Documentation of the scaling techniques applied, with code snippets |

## Dataset Summary

- **Source:** Dataset_ATS_v2.csv — 7,043 customer records, 10 original variables
- **Missing values:** None found during profiling
- **Duplicate rows:** 302 exact duplicate rows identified; retained and flagged rather than removed, since the dataset has no unique customer identifier to confirm they represent genuine duplicate customers
- **Target variable:** `Churn` (5,174 No / 1,869 Yes — approx. 73.5% / 26.5%, indicating class imbalance)

## Train / Test Set Composition

| Set | Records | % of Total | Churn Rate |
|---|---|---|---|
| Train | 5,634 | 80.0% | 26.5% |
| Test | 1,409 | 20.0% | 26.5% |

A stratified 80/20 split (`random_state=42`) was used so both sets preserve the same churn ratio as the full dataset, supporting reliable model validation.

## Preprocessing Steps Applied

1. **Missing data handling** — none found, but pipeline includes fill logic (median for numeric, mode for categorical) for robustness against future data refreshes
2. **Categorical encoding** — binary fields (gender, Dependents, PhoneService, MultipleLines, InternetService, Churn) label-encoded as 0/1; Contract (3 categories) one-hot encoded
3. **Feature scaling** — `tenure` and `MonthlyCharges` standardised using `StandardScaler` (see `Scaling_Techniques.docx` for full explanation and code)
4. **Train/test split** — stratified 80/20 split preserving churn ratio

## How to Run

```bash
cd Data_Preparation
pip install pandas numpy scikit-learn
python data_preparation.py
```
