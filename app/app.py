import streamlit as st
import pandas as pd
import numpy as np
import joblib

import os

# Load the trained model
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "models", "churn_model.pkl")
    return joblib.load(model_path)

model = load_model()

# Constants for scaling (computed from the training dataset)
MEANS = {
    'tenure': 32.371149,
    'MonthlyCharges': 64.761692,
    'TotalCharges': 2281.916928
}
STDS = {
    'tenure': 24.557737,
    'MonthlyCharges': 30.087911,
    'TotalCharges': 2265.109576
}

# The exact columns expected by the model in the correct order
EXPECTED_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

def preprocess_input(data):
    # Initialize a dictionary with all expected columns set to 0
    processed = {col: 0 for col in EXPECTED_COLUMNS}
    
    # 1. Label Encoded Variables
    processed['gender'] = 1 if data['gender'] == 'Male' else 0
    processed['Partner'] = 1 if data['Partner'] == 'Yes' else 0
    processed['Dependents'] = 1 if data['Dependents'] == 'Yes' else 0
    processed['PhoneService'] = 1 if data['PhoneService'] == 'Yes' else 0
    processed['PaperlessBilling'] = 1 if data['PaperlessBilling'] == 'Yes' else 0
    processed['SeniorCitizen'] = data['SeniorCitizen']
    
    # 2. Scaled Variables
    processed['tenure'] = (data['tenure'] - MEANS['tenure']) / STDS['tenure']
    processed['MonthlyCharges'] = (data['MonthlyCharges'] - MEANS['MonthlyCharges']) / STDS['MonthlyCharges']
    processed['TotalCharges'] = (data['TotalCharges'] - MEANS['TotalCharges']) / STDS['TotalCharges']
    
    # 3. One-Hot Encoded Variables
    if f"MultipleLines_{data['MultipleLines']}" in processed:
        processed[f"MultipleLines_{data['MultipleLines']}"] = 1
        
    if f"InternetService_{data['InternetService']}" in processed:
        processed[f"InternetService_{data['InternetService']}"] = 1
        
    if f"OnlineSecurity_{data['OnlineSecurity']}" in processed:
        processed[f"OnlineSecurity_{data['OnlineSecurity']}"] = 1
        
    if f"OnlineBackup_{data['OnlineBackup']}" in processed:
        processed[f"OnlineBackup_{data['OnlineBackup']}"] = 1
        
    if f"DeviceProtection_{data['DeviceProtection']}" in processed:
        processed[f"DeviceProtection_{data['DeviceProtection']}"] = 1
        
    if f"TechSupport_{data['TechSupport']}" in processed:
        processed[f"TechSupport_{data['TechSupport']}"] = 1
        
    if f"StreamingTV_{data['StreamingTV']}" in processed:
        processed[f"StreamingTV_{data['StreamingTV']}"] = 1
        
    if f"StreamingMovies_{data['StreamingMovies']}" in processed:
        processed[f"StreamingMovies_{data['StreamingMovies']}"] = 1
        
    if f"Contract_{data['Contract']}" in processed:
        processed[f"Contract_{data['Contract']}"] = 1
        
    if f"PaymentMethod_{data['PaymentMethod']}" in processed:
        processed[f"PaymentMethod_{data['PaymentMethod']}"] = 1
        
    # Convert to DataFrame with single row
    return pd.DataFrame([processed], columns=EXPECTED_COLUMNS)

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

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
    
    # Preprocess
    df_processed = preprocess_input(input_data)
    
    # Predict
    prediction = model.predict(df_processed)[0]
    probability = model.predict_proba(df_processed)[0][1]
    
    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 **High Risk of Churn!** (Probability: {probability:.2%})")
        st.write("This customer exhibits behaviors and characteristics that strongly suggest they might cancel their subscription. Consider reaching out with retention offers.")
    else:
        st.success(f"✅ **Low Risk of Churn.** (Probability: {probability:.2%})")
        st.write("This customer is likely to stay.")
