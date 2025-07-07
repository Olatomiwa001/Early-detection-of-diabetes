# diabetes_app.py

import streamlit as st
import joblib
import numpy as np

# Load model and features
model = joblib.load('diabetes_model.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Mellitus Prediction System")

st.markdown("""
Enter the following health details to assess the likelihood of having diabetes.  
The prediction is based on a machine learning model trained on the Pima Indians dataset.
""")

# Collect input
user_data = {}

user_data["Pregnancies"] = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
user_data["Glucose"] = st.slider("Glucose Level (mg/dL)", 0, 200, 100)
user_data["BloodPressure"] = st.slider("Blood Pressure (mm Hg)", 0, 140, 70)
user_data["SkinThickness"] = st.slider("Skin Thickness (mm)", 0, 100, 20)
user_data["Insulin"] = st.slider("Insulin Level", 0, 900, 85)
user_data["BMI"] = st.slider("BMI", 0.0, 70.0, 25.0)
user_data["DiabetesPedigreeFunction"] = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
user_data["Age"] = st.slider("Age", 10, 100, 30)

if st.button("🔍 Predict"):
    input_data = np.array([user_data[feat] for feat in feature_names]).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1] * 100

    st.subheader("🩺 Prediction Result")
    if prediction == 1:
        st.error(f"❌ High Risk of Diabetes ({proba:.2f}%)")
        st.markdown("Immediate medical consultation is strongly recommended.")
    else:
        st.success(f"✅ Low Risk of Diabetes ({100 - proba:.2f}%)")
        st.markdown("Maintain a healthy lifestyle and continue regular checkups.")
