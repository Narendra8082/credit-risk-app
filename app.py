import streamlit as st
import pandas as pd
import cloudpickle
import numpy as np

st.title("🏦 Credit Risk Prediction")
st.markdown("Enter details of the applicant to assess **credit risk**.")

# Step 1: Try to load the model
model = None
try:
    with open("credit_risk_model.pkl", "rb") as f:
        loaded_obj = cloudpickle.load(f)

    if hasattr(loaded_obj, "predict"):
        model = loaded_obj
    elif isinstance(loaded_obj, dict) and hasattr(loaded_obj.get("model", None), "predict"):
        model = loaded_obj["model"]
    else:
        st.error("❌ Loaded object is not a valid model. It may be a NumPy array or unrelated data.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Step 2: Input form
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

# Step 3: Make prediction
if submitted:
    input_dict = {
        "Age": age,
        "Job": job,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Sex": sex,
        "Housing": housing,
        "Saving accounts": None if saving_accounts == "NA" else saving_accounts,
        "Checking account": None if checking_account == "NA" else checking_account,
        "Purpose": purpose,
    }

    input_df = pd.DataFrame([input_dict])

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"✅ **Approved** - Probability of risk: {probability:.2f}")
        else:
            st.error(f"❌ **Denied** - Probability of risk: {probability:.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.write("📋 Input DataFrame:")
        st.dataframe(input_df)
