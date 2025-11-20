import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล
model = joblib.load("Diabetset.pkl")

# ค่าเฉลี่ยจาก dataset (โบรเอามาจาก df.describe())
AVG_GLUCOSE = 120
AVG_INSULIN = 100
AVG_SKIN = 25

st.title("ระบบพยากรณ์โรคเบาหวาน")

st.subheader("ข้อมูลพื้นฐาน")

# อายุ
age = st.number_input("อายุ (ปี)", min_value=10, max_value=100, value=30)

# น้ำหนัก + ส่วนสูง = คำนวณ BMI
weight = st.number_input("น้ำหนัก (kg)", min_value=20.0, max_value=200.0, value=60.0)
height = st.number_input("ส่วนสูง (cm)", min_value=120.0, max_value=220.0, value=170.0)
bmi = weight / ((height/100)**2)
st.write(f"**BMI ของคุณ:** {bmi:.2f}")

# Glucose
st.subheader("ระดับน้ำตาล (Glucose)")

glucose_unknown = st.checkbox("ไม่รู้ค่าระดับน้ำตาล (ใช้ค่าเฉลี่ยอัตโนมัติ)")

if glucose_unknown:
    glucose = AVG_GLUCOSE
    st.info(f"ใช้ค่าเฉลี่ยแทน: {glucose}")
else:
    glucose = st.number_input(
        "กรอกรระดับ Glucose",
        min_value=40,
        max_value=300,
        value=100
    )

# Blood Pressure
bp = st.number_input(
    "ความดันโลหิต (Diastolic Blood Pressure)",
    min_value=40,
    max_value=140,
    value=80
)

# SkinThickness
st.subheader("ความหนาผิวหนัง (Skin Thickness)")

skin_unknown = st.checkbox("ไม่รู้ค่า Skin Thickness (ใช้ค่าเฉลี่ย)")

if skin_unknown:
    skin = AVG_SKIN
    st.info(f"ใช้ค่าเฉลี่ยแทน: {skin}")
else:
    skin = st.number_input(
        "Skin Thickness (mm)",
        min_value=5,
        max_value=80,
        value=20
    )

# Insulin
st.subheader("ระดับอินซูลิน (Insulin)")

insulin_unknown = st.checkbox("ไม่รู้ค่า Insulin (ใช้ค่าเฉลี่ย)")

if insulin_unknown:
    insulin = AVG_INSULIN
    st.info(f"ใช้ค่าเฉลี่ยแทน: {insulin}")
else:
    insulin = st.number_input(
        "ระดับ Insulin",
        min_value=10,
        max_value=400,
        value=100
    )

# DPF → mapping จากประวัติครอบครัว
st.subheader("ประวัติครอบครัวเกี่ยวกับโรคเบาหวาน")

family = st.selectbox(
    "เลือกสถานะ",
    [
        "ไม่มีในครอบครัว",
        "มีคนในครอบครัวสายตรง (พ่อ/แม่)",
        "มีหลายคนในครอบครัว",
        "พันธุกรรมแรงมาก"
    ]
)

DPF_MAP = {
    "ไม่มีในครอบครัว": 0.1,
    "มีคนในครอบครัวสายตรง (พ่อ/แม่)": 0.5,
    "มีหลายคนในครอบครัว": 1.0,
    "พันธุกรรมแรงมาก": 2.0
}

dpf = DPF_MAP[family]

# ปุ่มพยากรณ์
if st.button("คำนวณความเสี่ยง"):

    features = np.array([[glucose, bmi, age, bp, insulin, dpf, skin]])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.subheader("ผลการประเมิน")

    if prediction == 1:
        st.error(f"⚠ คุณมีความเสี่ยงสูงเป็นเบาหวาน ({prob*100:.2f}%)")

        st.warning("""
### 🔥 คำแนะนำเบื้องต้น
- ควรพบแพทย์ตรวจน้ำตาลในเลือด (FBS / HbA1c)
- ลดอาหารหวานและแป้งขัดสี
- ควรออกกำลังกายสม่ำเสมอ
- ควรตรวจซ้ำภายใน 3 เดือน
        """)
    else:
        st.success(f"คุณมีความเสี่ยงต่ำ ({prob*100:.2f}%)")
        st.info("""
### ✅ แนะนำเพิ่มเติม
- ควบคุมน้ำหนักให้อยู่ในเกณฑ์
- ออกกำลังกาย 3–5 วันต่อสัปดาห์
- ลดน้ำตาลและอาหารทอด
        """)











