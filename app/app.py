import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

# Load the trained model pipeline
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "models", "churn_model.pkl")
    return joblib.load(model_path)

model = load_model()

st.title("📊 Customer Churn Prediction")
st.markdown("Enter customer details below to predict if they are likely to churn.")

with st.form("prediction_form"):
    st.header("Customer Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
    with col2:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    st.header("Services Subscribed")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        
    with col5:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        
    with col6:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.header("Charges")
    col7, col8 = st.columns(2)
    with col7:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    with col8:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Build raw input dictionary representing the customer profile
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Convert input to DataFrame (retains original format and column names)
    df_input = pd.DataFrame([input_data])
    
    # Predict directly using the loaded pipeline (avoids manual scaling and preprocessing leakage)
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]
    
    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 **High Risk of Churn!** (Probability: {probability:.2%})")
        st.write("This customer exhibits behaviors and characteristics that strongly suggest they might cancel their subscription. Consider reaching out with retention offers.")
    else:
        st.success(f"✅ **Low Risk of Churn.** (Probability: {probability:.2%})")
        st.write("This customer is likely to stay.")
