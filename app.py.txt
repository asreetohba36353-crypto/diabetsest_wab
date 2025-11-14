import streamlit as st
import joblib
import numpy as np

model = joblib.load("diabetes_best_model.pkl")

st.title("🔍 Diabetes Prediction Web App")

glucose = st.number_input("Glucose", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
age = st.number_input("Age", min_value=0)
blood = st.number_input("BloodPressure", min_value=0.0)
insulin = st.number_input("Insulin", min_value=0.0)
ped = st.number_input("DiabetesPedigreeFunction", min_value=0.0)
skin = st.number_input("SkinThickness", min_value=0.0)

if st.button("Predict"):
    data = np.array([[glucose, bmi, age, blood, insulin, ped, skin]])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if pred == 1:
        st.error(f"ผลทำนาย: เสี่ยงเป็นเบาหวาน (โอกาส {prob*100:.2f}%)")
    else:
        st.success(f"ผลทำนาย: ไม่เสี่ยง (โอกาส {prob*100:.2f}%)")
