# Telco Customer Churn Prediction & Retention Insights

Predicting which customers are likely to churn (leave) a telecom provider,
and surfacing the business drivers behind churn so retention teams can act on them.

> Live demo: _add your deployed Streamlit link here once Phase 6 is done_

## Business Problem

Customer acquisition costs 5–25x more than retention. This project builds a
model that flags at-risk customers early and explains *why* they're at risk,
so a business can intervene (offers, outreach, plan changes) before losing them.

## Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- ~7,043 customers, 21 features (demographics, account info, services subscribed)
- Target: `Churn` (Yes/No)

## Project Structure

```
telco-churn-prediction/
├── data/
│   ├── raw/            # original, untouched dataset
│   └── processed/      # cleaned/engineered data
├── notebooks/           # exploratory analysis
├── src/                 # reusable preprocessing & training code
├── app/                 # Streamlit app
├── models/               # saved trained models
├── reports/figures/      # charts and visuals for README/report
├── requirements.txt
└── README.md
```

## Approach

1. **EDA** — understand churn distribution and drivers (tenure, contract type, charges, services)
2. **Preprocessing** — handle missing values, encode categoricals, scale numerics
3. **Modeling** — Logistic Regression baseline, then Random Forest / XGBoost
4. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC (recall matters most here — missing a churner is costlier than a false alarm)
5. **Explainability** — feature importance / SHAP to explain *why* a customer is flagged
6. **Deployment** — Streamlit app for interactive prediction + dashboard

## Results

_Fill in once modeling is done:_
- Best model: —
- Accuracy: — | Recall: — | F1: — | ROC-AUC: —
- Top churn drivers: —

## How to Run

```bash
# Clone
git clone https://github.com/<your-username>/telco-churn-prediction.git
cd telco-churn-prediction

# Set up environment
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit · matplotlib/seaborn

## Author

Built by [Your Name] — [LinkedIn] · [Portfolio]

## Status
Project scaffold pushed to GitHub. Model + Streamlit app in progress.
