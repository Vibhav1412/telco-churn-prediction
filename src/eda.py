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