import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("knn_model.pkl")

# Define the input fields for user input
def user_input_features():
    st.title("Credit Risk Prediction")

    st.write("Enter the features of a single record below:")

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
    # Note: The below preprocessing should replicate the same transformations used during training.
    
    # Categorical Columns
    categorical_columns = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    
    # One-hot encode categorical columns
    one_hot_encoded = pd.get_dummies(input_df[categorical_columns], drop_first=False)
    
    # Drop original categorical columns and add one-hot-encoded columns
    input_df = input_df.drop(categorical_columns, axis=1).join(one_hot_encoded)
    
    # Ensure column order matches the model's training data
    # Note: Replace `expected_columns` with the actual column order used during training
    expected_columns = [
        "Age", "Job", "Credit amount", "Duration", 
        "Sex_female", "Sex_male", "Housing_free", "Housing_own", "Housing_rent",
        "Saving accounts_little", "Saving accounts_moderate", "Saving accounts_rich", 
        "Saving accounts_quite rich", "Saving accounts_unknown",
        "Checking account_little", "Checking account_moderate",
        "Checking account_rich", "Checking account_unknown",
        "Purpose_business", "Purpose_car", "Purpose_domestic appliances",
        "Purpose_education", "Purpose_furniture/equipment", 
        "Purpose_radio/TV", "Purpose_repairs", "Purpose_vacation/others"
    ]
    
    # Add missing columns with zeros if necessary
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match the expected order
    input_df = input_df[expected_columns]
    
    return input_df

# Main Streamlit App
def main():
    input_df = user_input_features()

    if st.button("Predict Risk"):
        # Preprocess user input
        processed_input = preprocess_input(input_df)

        # Make prediction
        prediction = model.predict(processed_input)
        prediction_label = "Good Risk" if prediction[0] == 0 else "Bad Risk"

        # Display prediction
        st.write(f"### Prediction: {prediction_label}")

if __name__ == "__main__":
    main()
