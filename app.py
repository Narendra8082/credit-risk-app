import streamlit as st
 import pandas as pd
 import cloudpickle
 from sklearn.preprocessing import StandardScaler
 import numpy as np
 
 # Load the trained pipeline model
 with open("credit_risk_model.pkl", "rb") as f:
     model = cloudpickle.load(f)
 
 st.title("🏦 Credit Risk Prediction")
 
 st.markdown("Enter details of the applicant to assess **credit risk**.")
 
 # Input form
 @@ -30,26 +31,40 @@
 
 # Handle input
 if submitted:
     # Convert inputs into a dataframe
     # Create input dictionary with consistent naming
     input_dict = {
         "Age": age,
         "Job": job,
         "Credit amount": credit_amount,
         "Duration": duration,
         "Sex": sex,
         "Housing": housing,
         "Saving accounts": None if saving_accounts == "NA" else saving_accounts,
         "Checking account": None if checking_account == "NA" else checking_account,
         "Saving accounts": saving_accounts if saving_accounts != "NA" else np.nan,
         "Checking account": checking_account if checking_account != "NA" else np.nan,
         "Purpose": purpose,
     }
 
     # Convert to DataFrame
     input_df = pd.DataFrame([input_dict])
 
     # Predict
     prediction = model.predict(input_df)[0]
     probability = model.predict_proba(input_df)[0][1]
 
     if prediction == 1:
         st.success(f"✅ **Approved** - Probability of risk: {probability:.2f}")
     else:
         st.error(f"❌ **Denied** - Probability of risk: {probability:.2f}")
     
     # Ensure consistent column ordering (match training data)
     expected_columns = [
         'Age', 'Job', 'Credit amount', 'Duration', 'Sex', 'Housing',
         'Saving accounts', 'Checking account', 'Purpose'
     ]
     input_df = input_df[expected_columns]
     
     try:
         # Predict
         prediction = model.predict(input_df)[0]
         probability = model.predict_proba(input_df)[0][1]
         
         if prediction == 1:
             st.success(f"✅ **Approved** - Probability of risk: {probability:.2f}")
         else:
             st.error(f"❌ **Denied** - Probability of risk: {probability:.2f}")
             
     except Exception as e:
         st.error(f"Prediction failed: {str(e)}")
         st.write("Input DataFrame:")
         st.write(input_df)
