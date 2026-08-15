import pandas as pd

# Load the raw dataset
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 1. Shape and column names
print("=" * 50)
print("SHAPE (rows, columns):", df.shape)
print("=" * 50)
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# 2. Data types
print("\n" + "=" * 50)
print("DATA TYPES:")
print(df.dtypes)

# 3. Missing values
print("\n" + "=" * 50)
print("MISSING VALUES PER COLUMN:")
print(df.isnull().sum())

# 4. Preview first rows
print("\n" + "=" * 50)
print("FIRST 5 ROWS:")
print(df.head())

# 5. Check the TotalCharges issue
print("\n" + "=" * 50)
print("TotalCharges dtype:", df["TotalCharges"].dtype)
print("Sample TotalCharges values:", df["TotalCharges"].head(10).tolist())
# Investigate the TotalCharges blank-string issue
print("\n" + "=" * 50)
print("ROWS WHERE TotalCharges IS BLANK/WHITESPACE:")
blank_mask = df["TotalCharges"].str.strip() == ""
print("Number of blank rows:", blank_mask.sum())
print(df[blank_mask][["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])