import streamlit as st
import pandas as pd
from joblib import load

# -------------------------------
# Cache model and scaler loading
# -------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = load("logreg_model.joblib")
    scaler = load("scaler.joblib")
    return model, scaler

model, scaler = load_model_and_scaler()

# -------------------------------
# App Title
# -------------------------------
st.title("Diabetic Retinopathy Prediction")
st.write("Enter patient data below:")

# -------------------------------
# Input fields
# -------------------------------
age = st.number_input("Age", min_value=0, max_value=120, value=30)
systolic_bp = st.number_input("Systolic Blood Pressure", min_value=50, max_value=250, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=30, max_value=150, value=80)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)

# -------------------------------
# Predict button
# -------------------------------
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame([[age, systolic_bp, diastolic_bp, cholesterol]],
                              columns=['age','systolic_bp','diastolic_bp','cholesterol'])
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {prediction} (0 = No Retinopathy, 1 = Retinopathy)")
    st.write(f"**Probability of Retinopathy:** {probability:.2f}")
    
    # Display the input data for reference
    st.write("### Entered Patient Data")
    st.write(input_data)