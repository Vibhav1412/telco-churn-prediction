import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from preprocessing import load_and_clean_data

# Load cleaned data
df = load_and_clean_data()

# One-hot encode all categorical (text) columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features (X) from target (y)
X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (helps some models, doesn't hurt XGBoost either)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Calculate class imbalance ratio to weight the minority class (churners) more heavily
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)
# Predict on test set
y_pred = model.predict(X_test_scaled)

# Evaluate
print("\n" + "=" * 50)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))