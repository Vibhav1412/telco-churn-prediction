import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_and_clean_data

df = load_and_clean_data()

# Overall churn rate
churn_rate = df["Churn"].mean() * 100
print(f"Overall churn rate: {churn_rate:.2f}%")

# Churn rate by Contract type
print("\nChurn rate by Contract type:")
print(df.groupby("Contract")["Churn"].mean() * 100)

# Plot it
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="Contract", y="Churn", errorbar=None)
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig("reports/figures/churn_by_contract.png")
print("\nChart saved to reports/figures/churn_by_contract.png")
plt.show()

# Churn by tenure
print("\n" + "=" * 50)
print("Average tenure: Churned vs Not Churned")
print(df.groupby("Churn")["tenure"].mean())

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30)
plt.title("Tenure Distribution: Churned vs Retained Customers")
plt.xlabel("Tenure (months)")
plt.tight_layout()
plt.savefig("reports/figures/churn_by_tenure.png")
print("\nChart saved to reports/figures/churn_by_tenure.png")
plt.show()