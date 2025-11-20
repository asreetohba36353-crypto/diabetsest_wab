import streamlit as st
import joblib
import numpy as np

model = joblib.load("Diabetset.pkl")

st.title("🩺 ระบบพยากรณ์โรคเบาหวาน")

# ===== INPUTS =====
glucose = st.number_input("ระดับน้ำตาลในเลือด (mg/dL)", min_value=30, max_value=300)

weight = st.number_input("น้ำหนัก (kg)", min_value=20.0, max_value=300.0)
height = st.number_input("ส่วนสูง (cm)", min_value=100.0, max_value=250.0)

if height > 0:
    bmi = weight / ((height/100)**2)
else:
    bmi = 0

st.write("ค่า BMI =", round(bmi, 2))

age = st.number_input("อายุ", min_value=10, max_value=100)
blood = st.number_input("ความดันโลหิต (mmHg)", min_value=20, max_value=200)
insulin = st.number_input("อินซูลิน", min_value=10, max_value=900)
skin = st.number_input("ความหนาผิวหนัง (mm)", min_value=1, max_value=110)

dpf = st.number_input("ค่าความเสี่ยงทางพันธุกรรม (0.0 - 3.0)", min_value=0.0, max_value=3.0, format="%.2f")

# ===== PREDICT =====
if st.button("ทำนายผล"):
    data = np.array([[glucose, bmi, age, blood, insulin, dpf, skin]])
    pred = model.predict(data)[0]

    st.subheader("ผลการทำนาย:")
    if pred == 1:
        st.error("⚠ มีความเสี่ยงเป็นโรคเบาหวาน")
    else:
        st.success("✔ ไม่เสี่ยงเป็นโรคเบาหวาน")






