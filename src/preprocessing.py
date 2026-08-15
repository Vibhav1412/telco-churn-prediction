"""
Data loading and preprocessing utilities for the Telco Churn project.
Fill these in as you go through Phases 2-3.
"""

import pandas as pd


def load_raw_data(path: str = "data/raw/telco_churn.csv") -> pd.DataFrame:
    """Load the raw Kaggle Telco Churn CSV."""
    df = pd.read_csv(path)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle the known quirks of this dataset:
    - TotalCharges is read as a string with some blank values
    - customerID is not a useful feature
    """
    df = df.copy()

    # TotalCharges has some blank strings for customers with 0 tenure
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns. Start simple with one-hot encoding,
    revisit if you want ordinal encoding for contract length etc.
    """
    target = None
    if "Churn" in df.columns:
        target = df["Churn"].map({"Yes": 1, "No": 0})
        df = df.drop(columns=["Churn"])

    df_encoded = pd.get_dummies(df, drop_first=True)

    if target is not None:
        df_encoded["Churn"] = target

    return df_encoded


if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = basic_clean(raw)
    encoded = encode_features(cleaned)
    encoded.to_csv("data/processed/telco_churn_processed.csv", index=False)
    print(f"Processed shape: {encoded.shape}")
