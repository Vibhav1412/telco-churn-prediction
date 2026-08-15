"""
Model training script. We'll build this out together in Phase 4.
Skeleton for now so the repo structure is complete from commit 1.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


def train_baseline(data_path: str = "data/processed/telco_churn_processed.csv"):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")

    joblib.dump(model, "models/baseline_logistic.pkl")
    return model


if __name__ == "__main__":
    train_baseline()
