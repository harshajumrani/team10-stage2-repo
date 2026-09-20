# Team 10 — Customer Churn Analysis: Stage 2 Deliverables

ACS WILDA Internship | Data Analyst – Predictive Modelling (Harsha Jumrani)

## Repository Structure

```
.
├── Dataset_ATS_v2.csv          # Original raw dataset
├── Data_Preparation/
│   ├── data_preparation.py
│   ├── preprocessed_dataset.csv
│   ├── train_set.csv
│   ├── test_set.csv
│   ├── train_test_composition.txt
│   ├── model_feature_cols.txt
│   ├── Scaling_Techniques.docx
│   └── README.md
└── Clustering_Analysis/
    ├── clustering_analysis.py
    ├── cluster_selection.png
    ├── kmeans_model.joblib
    ├── segments_scatter.png
    ├── churn_rate_by_segment.png
    ├── segment_profiles.csv
    ├── clustered_dataset.csv
    └── README.md
```

## How to Push This to GitHub/GitLab

1. Create a new **public** repository on GitHub (e.g. `team10-customer-churn-stage2`) — do not initialise it with a README, since you already have one here.
2. On your own machine, download this whole folder (all files above), then open a terminal inside it and run:

```bash
git init
git add .
git commit -m "Stage 2: Data Preparation and Clustering Analysis deliverables"
git branch -M main
git remote add origin https://github.com/<your-username>/team10-customer-churn-stage2.git
git push -u origin main
```

3. Confirm the repository is set to **Public** (or shared with the project instructor) under the repository's Settings → General → Danger Zone → Change visibility.
4. Copy the repository URL — that's what gets submitted as the "GitHub/GitLab link" for Stage 2.

## What's Included vs. What You Still Need to Do

This repo covers everything **except the video demonstration**, which needs to be recorded by you personally (walking through the preprocessed dataset, train/test sets, scaling techniques, optimal cluster count, trained K-Means model, and segment visualisations — see each folder's README for the specific numbers/results to reference).

Remember: per the submission instructions, only the **Project Manager** submits the final links on behalf of the team — pass the repository link and video to Bhavin rather than submitting individually.
