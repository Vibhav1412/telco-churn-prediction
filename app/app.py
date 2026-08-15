"""
Streamlit app for the Telco Churn project.
We'll build this out in Phase 6 once the model is trained.
Running this now will show a placeholder page - that's expected.
"""

import streamlit as st

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉")

st.title("📉 Telco Customer Churn Predictor")
st.write(
    "This app will let you enter a customer's details and predict "
    "their likelihood of churning, once the model is trained."
)

st.info("🚧 Model not trained yet — this is a placeholder. Come back after Phase 4-5!")

# TODO (Phase 6):
# 1. Load trained model with joblib.load("models/....pkl")
# 2. Build input widgets for customer features (st.selectbox, st.slider, etc.)
# 3. Run prediction on submit, show churn probability + top contributing factors
