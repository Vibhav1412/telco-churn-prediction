import pandas as pd

def load_and_clean_data(filepath="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """Load the raw Telco churn dataset and apply basic cleaning."""
    df = pd.read_csv(filepath)

    # Fix TotalCharges: blank strings -> 0 (these are new customers, tenure=0)
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # Drop customerID - it's just an identifier, not useful for modeling
    df = df.drop(columns=["customerID"])

    # Convert Churn to binary (1 = Yes, 0 = No) - easier for modeling later
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


if __name__ == "__main__":
    df = load_and_clean_data()
    print("Cleaned data shape:", df.shape)
    print("\nTotalCharges dtype now:", df["TotalCharges"].dtype)
    print("\nChurn value counts:")
    print(df["Churn"].value_counts())
    print("\nAny missing values left?")
    print(df.isnull().sum().sum())
