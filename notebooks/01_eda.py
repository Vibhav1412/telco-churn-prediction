# %% [markdown]
# # Phase 2: Exploratory Data Analysis
# Run this in Jupyter/VS Code (cell by cell, using the `# %%` markers) or
# paste sections into Google Colab. Fill in after you've downloaded the
# dataset into data/raw/telco_churn.csv

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("../data/raw/telco_churn.csv")  # use full path if running in Colab
df.head()

# %%
df.info()

# %%
df.isnull().sum()

# %% [markdown]
# ## Churn distribution — how imbalanced is the target?

# %%
churn_rate = df["Churn"].value_counts(normalize=True) * 100
print(churn_rate)

sns.countplot(data=df, x="Churn")
plt.title("Churn Distribution")
plt.show()

# %% [markdown]
# ## Churn by contract type — this is usually one of the strongest drivers

# %%
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.show()

# %% [markdown]
# ## Churn by tenure — do newer customers churn more?

# %%
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30)
plt.title("Tenure Distribution by Churn")
plt.show()

# %% [markdown]
# ## TODO — keep exploring:
# - MonthlyCharges vs Churn
# - Which services (InternetService, TechSupport, etc.) correlate with churn?
# - Payment method vs churn
# - Write 3-5 bullet point insights below — these become your README "Results" section
