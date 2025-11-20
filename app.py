import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

# =========================
# Load model and dataset
# =========================
model = pickle.load(open("Diabetset.pkl", "rb"))
df = pd.read_csv("diabetes.csv")

# ค่าเฉลี่ยของคอลัมน์ต่าง ๆ
AVG_GLUCOSE = df["Glucose"].mean()
AVG_INSULIN = df["Insulin"].mean()
AVG_SKIN = df["SkinThickness"].mean()

# =========================
# Helper: mapping DPF
# =========================
def get_family_history_dpf(choice):
    mapping = {
        "ไม่มีใครเป็น": 0.1,
        "มีคนในครอบครัว 1 คน": 0.3,
        "มี 2 คน": 0.5,
        "มีมากกว่า 2 คน": 1.0
    }
    return mapping.get(choice, 0.3)


# =========================
# Input UI
# =========================
st.title("🩺 ระบบประเมินโอกาสเป็นเบาหวาน")

st.subheader("ข้อมูลหลัก")

age = st.number_input("อายุ (ปี)", 10, 100, 30)

# BMI = นน./(ส่วนสูง^2)
weight = st.number_input("น้ำหนัก (kg)", 20, 200, 60)
height = st.number_input("ส่วนสูง (cm)", 120, 220, 165)
bmi = weight / ((height / 100) ** 2)

st.write(f"**BMI = {bmi:.2f}**")

# Glucose
st.subheader("ระดับน้ำตาลในเลือด (Glucose)")
unknown_glucose = st.checkbox("ไม่รู้ค่า Glucose")

if unknown_glucose:
    glucose = AVG_GLUCOSE
    st.info(f"ใช้ค่าเฉลี่ยจากข้อมูล = {glucose:.1f}")
else:
    glucose = st.number_input("กรอก Glucose", 40, 300, 120)

# Blood Pressure
blood = st.number_input("ค่าความดัน (Diastolic)", 40, 140, 70)

# Skin Thickness
st.subheader("ความหนาผิวหนัง (SkinThickness)")
unknown_skin = st.checkbox("ไม่รู้ค่า SkinThickness")

if unknown_skin:
    skin = AVG_SKIN
    st.info(f"ใช้ค่าเฉลี่ยจากข้อมูล = {skin:.1f}")
else:
    skin = st.number_input("SkinThickness", 5, 80, 20)

# Insulin
st.subheader("ระดับ Insulin")
unknown_insulin = st.checkbox("ไม่รู้ค่า Insulin")

if unknown_insulin:
    insulin = AVG_INSULIN
    st.info(f"ใช้ค่าเฉลี่ย = {insulin:.1f}")
else:
    insulin = st.number_input("Insulin", 10, 400, 85)

# Diabetes Pedigree Function
st.subheader("ประวัติครอบครัวเป็นเบาหวาน")
family = st.selectbox("เลือก", ["ไม่มีใครเป็น", "มีคนในครอบครัว 1 คน", "มี 2 คน", "มีมากกว่า 2 คน"])
dpf = get_family_history_dpf(family)

preg = st.number_input("จำนวนครั้งที่ตั้งครรภ์", 0, 20, 1)


# =========================
# Prediction
# =========================
if st.button("ประเมินความเสี่ยง"):
    features = np.array([[preg, glucose, blood, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.subheader("ผลการประเมิน")

    if prediction == 1:
        st.error(f"⚠ มีความเสี่ยง **สูง** ({prob*100:.2f}%)")

        st.warning("""
        🔶 **คำแนะนำเบื้องต้น**
        - ควรไปพบแพทย์เพื่อตรวจน้ำตาลในเลือด
        - ออกกำลังกายอย่างน้อย 150 นาทีต่อสัปดาห์
        - เลี่ยงอาหารหวาน มัน เค็ม
        - ควบคุมน้ำหนักให้อยู่ในเกณฑ์ปกติ
        """)

        if bmi > 27:
            st.info("💡 BMI สูงกว่าปกติ → ควรลดน้ำหนักลง 5–10%")
        if glucose > 160:
            st.info("💡 ค่าน้ำตาลสูงกว่าเกณฑ์ → เสี่ยงมากขึ้น")
    else:
        st.success(f"✓ ความเสี่ยงต่ำ ({prob*100:.2f}%)")

        st.info("""
        💚 **คำแนะนำ**
        - ดูแลสุขภาพให้ดีต่อเนื่อง
        - ทานอาหารครบ 5 หมู่
        - ออกกำลังกายสม่ำเสมอ
        """)












