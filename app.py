import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("credit_risk_model.pkl")

st.title("Credit Risk Prediction (Single Entry)")

# Take inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job Type", options=[0, 1, 2, 3])
credit_amount = st.number_input("Credit Amount", min_value=100, max_value=100000, value=2000)
duration = st.number_input("Duration (Months)", min_value=4, max_value=72, value=12)

sex = st.selectbox("Sex", ["male", "female"])
housing = st.selectbox("Housing", ["own", "free", "rent"])
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich", "NA"])
checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "NA"])
purpose = st.selectbox("Purpose", ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"])

# Encode into dataframe
input_dict = {
    "Age": age,
    "Job": job,
    "Credit amount": credit_amount,
    "Duration": duration,
    "Sex": sex,
    "Housing": housing,
    "Saving accounts": saving_accounts if saving_accounts != "NA" else None,
    "Checking account": checking_account if checking_account != "NA" else None,
    "Purpose": purpose
}
input_df = pd.DataFrame([input_dict])

# One-hot encoding
input_df_encoded = pd.get_dummies(input_df)

# Align with model features
model_features = model.feature_names_in_
for col in model_features:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0  # add missing columns

input_df_encoded = input_df_encoded[model_features]

# Prediction
if st.button("Predict Risk"):
    prediction = model.predict(input_df_encoded)[0]
    prob = model.predict_proba(input_df_encoded)[0][1]
    st.write(f"### Prediction: **{prediction}**")
    st.write(f"### Risk Probability: **{prob:.2f}**")
