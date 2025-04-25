import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and feature names
model = joblib.load("knn_model.pkl")

# Load the feature names used during training
try:
    expected_columns = joblib.load("model_features.pkl")
except FileNotFoundError:
    st.error("Feature list file (model_features.pkl) is missing. Ensure it is available in the same directory as this app.")
    st.stop()

# Define the input fields for user input
def user_input_features():
    st.title("Credit Risk Prediction")

    st.write("Enter the details of the applicant to assess credit risk:")

    # Input fields
    age = st.number_input("Age (e.g., 35)", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.number_input("Job type (e.g., 0, 1, 2, 3)", min_value=0, max_value=3, value=1, step=1)
    housing = st.selectbox("Housing", ["own", "free", "rent"])
    saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich", "unknown"])
    checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "unknown"])
    credit_amount = st.number_input("Credit Amount (e.g., 1000)", min_value=0, value=1000, step=100)
    duration = st.number_input("Duration (in months, e.g., 12)", min_value=1, value=12, step=1)
    purpose = st.selectbox(
        "Purpose", 
        ["business", "car", "domestic appliances", "education", "furniture/equipment", 
         "radio/TV", "repairs", "vacation/others"]
    )

    # Collect inputs into a dictionary
    data = {
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }

    return pd.DataFrame([data])

# Preprocess the user input to match the trained model
def preprocess_input(input_df):
    # Match preprocessing from training (e.g., one-hot encoding, scaling)
    categorical_columns = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    
    # One-hot encode categorical columns
    one_hot_encoded = pd.get_dummies(input_df[categorical_columns], drop_first=False)
    
    # Drop original categorical columns and add one-hot-encoded columns
    input_df = input_df.drop(categorical_columns, axis=1).join(one_hot_encoded)
    
    # Add missing columns with zeros if necessary
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Ensure column order matches the expected order
    input_df = input_df[expected_columns]
    
    return input_df

# Main Streamlit App
def main():
    input_df = user_input_features()

    if st.button("Predict Risk"):
        # Preprocess user input
        processed_input = preprocess_input(input_df)

        # Debugging: Show processed input
        st.write("Processed Input:")
        st.dataframe(processed_input)

        try:
            # Make prediction
            prediction = model.predict(processed_input)
            prediction_label = "Good Risk" if prediction[0] == 0 else "Bad Risk"

            # Display prediction
            st.write(f"### Prediction: {prediction_label}")
        except ValueError as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
