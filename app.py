import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# Load trained model
model = joblib.load("heart_disease_rf_model.pkl")

# App title
st.title("❤️ Heart Disease Risk Assessment")
st.write("Enter patient health details to predict heart disease risk.")

# Sidebar inputs
st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", 1, 100, 45)

sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0

cp = st.sidebar.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG Result (0–2)", [0, 1, 2])
thalach = st.sidebar.number_input("Maximum Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thal (0 = Normal, 1 = Fixed, 2 = Reversible)", [0, 1, 2])

# Description
st.markdown("""
This application predicts the **risk of heart disease**
using a **Random Forest Machine Learning model**.
""")

# Predict button
if st.button("Predict Risk"):

    # Prepare input
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                             restecg, thalach, exang, oldpeak,
                             slope, ca, thal]])

    # Prediction
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1] * 100

    # Result message
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease\n\nRisk Probability: {prob:.2f}%")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n\nRisk Probability: {prob:.2f}%")

    # Gauge visualization (reduced size)
    st.subheader("📊 Risk Meter")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={"text": "Heart Disease Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red"},
            "steps": [
                {"range": [0, 40], "color": "lightgreen"},
                {"range": [40, 70], "color": "yellow"},
                {"range": [70, 100], "color": "pink"}
            ],
        }
    ))

    st.plotly_chart(fig)


    # Medical advice for high risk
    if prob >= 70:
        st.warning(
            "⚕️ Medical Advice:\n\n"
            "This indicates a high risk of heart disease. "
            "Please consult a qualified cardiologist as soon as possible "
            "for proper medical evaluation and guidance."
        )

# Footer
st.markdown("---")
st.caption("Developed by Harishmitha Uk J | CodeClause Internship Project")
