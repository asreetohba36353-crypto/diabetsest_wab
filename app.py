DPF_MAP = {
    "ไม่มีประวัติในครอบครัว": 0.05,
    "ญาติห่าง (เช่น ป้า/น้า/อา) เป็น": 0.5,
    "พ่อหรือแม่เป็น": 1.0,
    "พ่อแม่ + พี่น้องเป็น": 2.0,
    "หลายคนในครอบครัวเป็น": 2.5,
}
import streamlit as st
import numpy as np
import joblib

# โหลดโมเดล
model = joblib.load("Diabetset.pkl")

# Mapping DPF
DPF_MAP = {
    "ไม่มีประวัติในครอบครัว": 0.05,
    "ญาติห่าง (เช่น ป้า/น้า/อา) เป็น": 0.5,
    "พ่อหรือแม่เป็น": 1.0,
    "พ่อแม่ + พี่น้องเป็น": 2.0,
    "หลายคนในครอบครัวเป็น": 2.5,
}

st.title("🩺 ระบบพยากรณ์ความเสี่ยงโรคเบาหวาน")

st.write("กรอกข้อมูลด้านล่างให้ถูกต้องตามช่วงจริงของมนุษย์ ระบบจะช่วยตรวจสอบความผิดปกติให้อัตโนมัติ")

# ====== INPUT FORM ======
glucose = st.number_input("ระดับน้ำตาลในเลือด (mg/dL)", min_value=40, max_value=300)
if glucose < 40 or glucose > 300:
    st.warning("⚠ ค่า Glucose อยู่เกินช่วงจริง 40–300 mg/dL")

age = st.number_input("อายุ (ปี)", min_value=10, max_value=100)
blood = st.number_input("ความดันโลหิต (mmHg)", min_value=40, max_value=140)
skin = st.number_input("ความหนา Skin Thickness (mm)", min_value=5, max_value=80)
insulin = st.number_input("ระดับอินซูลิน (μU/mL)", min_value=10, max_value=400)

# BMI คำนวณอัตโนมัติจาก นน./ส่วนสูง
weight = st.number_input("น้ำหนัก (kg)", min_value=20.0, max_value=250.0)
height = st.number_input("ส่วนสูง (cm)", min_value=100.0, max_value=220.0)

if height > 0:
    bmi = weight / ((height/100)**2)
else:
    bmi = 0

bmi = round(bmi, 2)
st.write(f"ค่า BMI = **{bmi}**")
if bmi < 10 or bmi > 60:
    st.warning("⚠ ค่า BMI อยู่นอกช่วงปกติ 10–60")

# DPF dropdown
dpf_label = st.selectbox("ประวัติโรคเบาหวานในครอบครัว", list(DPF_MAP.keys()))
dpf = DPF_MAP[dpf_label]

# ส่งค่าเข้าโมเดล
if st.button("ทำนายผล"):
    # ตรวจสอบค่าไม่ปกติ → clip ให้โมเดลรับได้
    glucose = np.clip(glucose, 40, 300)
    blood = np.clip(blood, 40, 140)
    skin = np.clip(skin, 5, 80)
    insulin = np.clip(insulin, 10, 400)
    bmi = np.clip(bmi, 10, 60)

    data = np.array([[glucose, bmi, age, blood, insulin, dpf, skin]])

    pred = model.predict(data)[0]

    st.subheader("ผลการประเมินจากโมเดล:")
    if pred == 1:
        st.error("⚠ มีความเสี่ยงสูงที่จะเป็นโรคเบาหวาน")
    else:
        st.success("✔ ความเสี่ยงต่ำ ไม่เข้าข่ายโรคเบาหวาน")

    st.info("หมายเหตุ: ผลลัพธ์นี้เป็นการประเมินเชิงสถิติ ไม่ใช่การวินิจฉัยทางการแพทย์")







