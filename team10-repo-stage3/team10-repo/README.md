# Team 10 — Customer Churn Analysis: Stage 3 Deliverables

ACS WILDA Internship | Team 10: Bhavin (Project Manager), Harsha Jumrani (Data Analyst – Predictive Modelling), Mitul (Cluster Analyst), Pruthal (Data Engineer), Ashrafun (Business Analyst)

## Repository Structure

```
.
├── Dataset_ATS_v2.csv          # Original raw dataset
├── requirements.txt            # Python packages for all stages
├── Data_Preparation/           # Stage 2
│   ├── data_preparation.py
│   ├── preprocessed_dataset.csv
│   ├── train_set.csv
│   ├── test_set.csv
│   ├── train_test_composition.txt
│   ├── model_feature_cols.txt
│   ├── Scaling_Techniques.docx
│   └── README.md
├── Clustering_Analysis/        # Stage 2
│   ├── clustering_analysis.py
│   ├── cluster_selection.png
│   ├── kmeans_model.joblib
│   ├── segments_scatter.png
│   ├── churn_rate_by_segment.png
│   ├── segment_profiles.csv
│   ├── clustered_dataset.csv
│   └── README.md
├── Predictive_Modeling/        # Stage 3
│   ├── prepare_model_data.py           (Pruthal)
│   ├── model_train.csv / model_val.csv / model_test.csv
│   ├── scaler.joblib, feature_columns.json, split_summary.json
│   ├── Model_Data_Pipeline.docx
│   ├── train_ann.py                    (Harsha)
│   ├── ANN_Architecture.docx           <- defined ANN architecture
│   ├── churn_ann_model.keras           <- trained ANN model
│   ├── model_summary.txt, architecture_search.csv
│   ├── evaluation_metrics.json, test_predictions.csv
│   ├── training_history.png, confusion_matrix.png, roc_pr_curves.png
│   ├── segment_churn_analysis.py       (Mitul)
│   ├── segment_prediction_summary.csv, risk_tier_by_segment.csv, risk_tier_actual_churn.csv
│   ├── segment_actual_vs_predicted.png, risk_tier_by_segment.png
│   ├── churn_factors_analysis.py       (Ashrafun)
│   ├── feature_importance.csv/.png, attribute_effects.csv, effect_tenure.csv, effect_charges.csv
│   ├── effect_tenure_charges.png, churn_rate_by_attribute.csv/.png
│   └── README.md
├── Final_Report/               # Stage 3
│   ├── Final_Report.docx               <- submit via LMS
│   ├── 1_Summary_of_Key_Findings.docx  (Mitul)
│   ├── 2_Churn_and_Retention_Factors.docx (Ashrafun)
│   ├── 3_Retention_Strategy_Recommendations.docx (Ashrafun)
│   ├── 4_Limitations_and_Proposed_Solutions.docx (Bhavin)
│   └── README.md
└── Video/                      # Stage 3
    ├── Video_Script_and_Plan.docx
    └── README.md
```

## Stage 3 Headline Results

| Metric (test set, 1,409 unseen customers) | ANN |
|---|---|
| ROC AUC | 0.805 (5-fold CV 0.824 ± 0.005) |
| Accuracy | 77.2% |
| Recall (churners caught) | 57.8% |
| Precision | 56.8% |

ANN architecture: Input (14) → Dense 64 ReLU → Dropout 0.3 → Dense 32 ReLU → Dropout 0.3 → Dense 1 Sigmoid. Customers rated high-risk churn at 56.8% vs 5.9% for low-risk. Tenure and monthly charges are the main drivers of churn. See each folder's README for details.

## How to Run Everything

```bash
pip install -r requirements.txt
cd Data_Preparation && python data_preparation.py && cd ..
cd Clustering_Analysis && python clustering_analysis.py && cd ..
cd Predictive_Modeling
python prepare_model_data.py
python train_ann.py
python segment_churn_analysis.py
python churn_factors_analysis.py
```

All outputs are already committed, so the scripts only need to be re-run to reproduce them.

## How to Push Stage 3 to the Existing GitHub/GitLab Repository

The submission instructions say to keep using the Stage 2 repository. Copy the new Stage 3 folders and files into your local copy of that repository, then open a terminal inside it and run:

```bash
git add .
git commit -m "Stage 3: Predictive Modeling, Final Report and video plan"
git push
```

If you are starting from this folder on a machine where the repository was never cloned, run `git clone <your-repo-url>`, copy these files into the cloned folder, then run the commands above.

Check the repository is still **Public** (or shared with the project instructor), then resubmit the repository URL.

## What's Included vs. What You Still Need to Do

This repo covers everything for Stage 3 **except the video demonstration**, which the team needs to record (10–15 minutes). `Video/Video_Script_and_Plan.docx` gives each member's section, timing and talking points.

Submission checklist:
1. Resubmit the repository link.
2. Submit `Final_Report/Final_Report.docx` via the LMS.
3. Upload the video demonstration.

Remember: only the **Project Manager** submits on behalf of the team, so pass the repository link, final report and video to Bhavin rather than submitting individually.
