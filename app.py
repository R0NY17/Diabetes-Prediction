import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.markdown("""
Welcome to the **Diabetes Prediction App**!  
This tool uses a machine learning model trained on medical data to predict whether a person is likely to have diabetes.

---

### 📌 How It Works
1. Enter the patient's health information in the form below.
2. Click **Predict**.
3. The app will tell you if the person is likely diabetic or not — based on the model's analysis.

**Note**: This is a demo built with public data and is not intended for medical diagnosis.

---
""")

# Load model and scaler
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# App title
st.title("Diabetes Prediction App")
st.markdown("Enter patient details to predict diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction button
if st.button("Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    if prediction[0] == 1:
        st.error("⚠️ The model predicts the person has **diabetes**.")
    else:
        st.success("✅ The model predicts the person is **not diabetic**.")