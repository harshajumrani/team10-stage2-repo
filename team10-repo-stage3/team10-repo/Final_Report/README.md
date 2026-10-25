# Final Report Deliverables

## Files in this folder

| File | Description | Owner |
|---|---|---|
| `Final_Report.docx` | **Combined final report — submit this via the LMS.** Includes all sections below plus the ANN architecture, training and evaluation | Bhavin (compiled) |
| `1_Summary_of_Key_Findings.docx` | a. Summary of key findings from customer segmentation and ANN churn prediction | Mitul |
| `2_Churn_and_Retention_Factors.docx` | b. Identification and analysis of factors contributing to churn and retention | Ashrafun |
| `3_Retention_Strategy_Recommendations.docx` | c. Recommendations for targeted retention strategies | Ashrafun |
| `4_Limitations_and_Proposed_Solutions.docx` | d. Limitations, challenges and proposed solutions | Bhavin |

## Summary of Key Findings

- The ANN reached **ROC AUC 0.805, accuracy 77.2%, recall 57.8% and precision 56.8%** on 1,409 unseen test customers (5-fold CV AUC 0.824 ± 0.005).
- The **New, High-Spend (At-Risk)** segment is 31% of customers but about **59% of churners**; half of them churn.
- Customers the ANN rates high-risk churn at **56.8%**, vs **5.9%** for low-risk customers.
- **Loyal, High-Spend** customers are a blind spot: 16% of churners, but the model catches only 6.7% of them.

## Factors Contributing to Churn and Retention

| Churn drivers | Retention drivers | Little or no effect |
|---|---|---|
| Short tenure (52.9% churn in first 6 months) | Long tenure (9.5% churn after 4 years) | Contract type (25–27% for all three) |
| High monthly charges (33–37% churn at $70+) | Low monthly charges (10.1% under $30) | Internet service type |
| Senior citizen (41.7% vs 23.6%) | Having dependents (15.5% vs 31.3%) | Gender, multiple lines |

## Top Retention Recommendations

1. First-year onboarding programme (check-ins at 30, 60 and 90 days)
2. Plan-fit reviews and early loyalty offers for the New, High-Spend segment
3. Monthly ANN scoring with tiered outreach (high / medium / low risk) measured against a control group
4. Price-lock and bundle offers for customers paying $70+ per month
5. Senior customer support, household bundles, and relationship check-ins for Loyal, High-Spend customers

## Main Limitations

Only 10 customer variables (the ANN performs no better than logistic regression), class imbalance, uncalibrated probabilities, an unusual contract-type result that should be checked against full company data, duplicate rows with no customer ID, and a blind spot in the Loyal, High-Spend segment. Proposed solutions for each are in `4_Limitations_and_Proposed_Solutions.docx`.
