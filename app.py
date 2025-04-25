import streamlit as st
import pandas as pd
import cloudpickle
import numpy as np

# Load with verification
try:
    with open("credit_risk_model.pkl", "rb") as f:
        model = cloudpickle.load(f)
        
    if isinstance(model, np.ndarray):
        st.error("ERROR: The model file contains a NumPy array, not a trained model.")
        st.error("Please retrain your model and save the model object, not predictions.")
        st.stop()
        
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

st.title("🏦 Credit Risk Prediction")
st.markdown("Enter details of the applicant to assess **credit risk**.")

# Input form
with st.form("credit_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job Type (0=Unskilled, 3=Highly Skilled)", options=[0, 1, 2, 3])
    credit_amount = st.number_input("Credit Amount", min_value=100, max_value=100000, value=2000)
    duration = st.number_input("Loan Duration (Months)", min_value=4, max_value=72, value=12)

    sex = st.selectbox("Sex", ["male", "female"])
    housing = st.selectbox("Housing", ["own", "free", "rent"])
    saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich", "NA"])
    checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "NA"])
    purpose = st.selectbox("Purpose", [
        "radio/TV", "education", "furniture/equipment", "car", "business",
        "domestic appliances", "repairs", "vacation/others"
    ])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "Age": age,
        "Job": job,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Sex": sex,
        "Housing": housing,
        "Saving accounts": saving_accounts if saving_accounts != "NA" else np.nan,
        "Checking account": checking_account if checking_account != "NA" else np.nan,
        "Purpose": purpose,
    }
    
    input_df = pd.DataFrame([input_dict])
    
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        if prediction == 1:
            st.success(f"✅ Approved (Risk probability: {probability:.1%})")
        else:
            st.error(f"❌ Denied (Risk probability: {probability:.1%})")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Debug Info:")
        st.write("Input features:", input_df.columns.tolist())
        if hasattr(model, 'feature_names_in_'):
            st.write("Model expects:", model.feature_names_in_)
