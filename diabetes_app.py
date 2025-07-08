# diabetes_app.py
# Early Detection of Diabetes Mellitus Using Feature Selection and Ensemble Models

import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Load model and features
model = joblib.load('diabetes_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Page configuration
st.set_page_config(
    page_title="Early Diabetes Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"  # Better for mobile
)

# Mobile-optimized CSS with high contrast colors
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling - High contrast, mobile-first */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: #ffffff;
        color: #1a1a1a;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: #0f0f0f;
            color: #ffffff;
        }
        
        .main-header {
            background: #1a1a1a !important;
            border: 2px solid #333333 !important;
        }
        
        .input-section, .result-section, .metric-card {
            background: #1a1a1a !important;
            border: 1px solid #333333 !important;
            color: #ffffff !important;
        }
        
        .stSelectbox > div > div, .stSlider > div > div {
            background: #1a1a1a !important;
            border: 1px solid #333333 !important;
        }
    }
    
    /* Main container - Mobile optimized */
    .main-header {
        background: #000000;
        color: #ffffff;
        padding: 1.5rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
        border: 2px solid #000000;
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .main-header p {
        color: #ffffff;
        font-size: 0.9rem;
        margin-bottom: 0;
        opacity: 0.9;
    }
    
    /* Input section - Mobile first */
    .input-section {
        background: #ffffff;
        color: #1a1a1a;
        padding: 1.5rem 1rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .input-section h2 {
        color: #1a1a1a;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        border-bottom: 2px solid #000000;
        padding-bottom: 0.5rem;
    }
    
    .input-section h3 {
        color: #1a1a1a;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        font-size: 1.1rem;
    }
    
    /* Results section */
    .result-section {
        background: #ffffff;
        color: #1a1a1a;
        padding: 1.5rem 1rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        margin-top: 1rem;
    }
    
    .result-section h2 {
        color: #1a1a1a;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    
    /* Risk level styling - High contrast */
    .risk-high {
        background: #dc2626;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 700;
        font-size: 1.1rem;
        border: 2px solid #dc2626;
    }
    
    .risk-low {
        background: #16a34a;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 700;
        font-size: 1.1rem;
        border: 2px solid #16a34a;
    }
    
    /* Metric cards - Mobile optimized */
    .metric-card {
        background: #ffffff;
        color: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        opacity: 0.8;
    }
    
    /* Form styling */
    .stForm {
        background: transparent;
        border: none;
        padding: 0;
    }
    
    .stFormSubmitButton > button {
        background: #000000 !important;
        color: #ffffff !important;
        border: 2px solid #000000 !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        margin-top: 1rem !important;
    }
    
    .stFormSubmitButton > button:hover {
        background: #333333 !important;
        border-color: #333333 !important;
    }
    
    /* Sidebar optimization */
    .sidebar .sidebar-content {
        background: #f8f9fa;
        color: #1a1a1a;
        padding: 1rem;
    }
    
    /* Info sections */
    .info-section {
        background: #000000;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 2px solid #000000;
    }
    
    .info-section h4 {
        color: #ffffff;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .info-section ul {
        margin-left: 1rem;
        color: #ffffff;
    }
    
    .info-section li {
        margin-bottom: 0.5rem;
    }
    
    /* Recommendations styling */
    .recommendation-high {
        background: #dc2626;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 2px solid #dc2626;
    }
    
    .recommendation-low {
        background: #16a34a;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 2px solid #16a34a;
    }
    
    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem;
        }
        
        .main-header p {
            font-size: 0.8rem;
        }
        
        .input-section, .result-section {
            padding: 1rem 0.75rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        .risk-high, .risk-low {
            padding: 1rem;
            font-size: 1rem;
        }
    }
    
    /* Streamlit element overrides */
    .stSelectbox > div > div {
        background: #ffffff;
        color: #1a1a1a;
        border: 1px solid #d1d5db;
    }
    
    .stSlider > div > div {
        color: #1a1a1a;
    }
    
    .stSlider .st-bb {
        background: #000000;
    }
    
    .stNumberInput > div > div {
        background: #ffffff;
        color: #1a1a1a;
        border: 1px solid #d1d5db;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 1rem;
        background: #f8f9fa;
        color: #1a1a1a;
        margin-top: 2rem;
        border-top: 2px solid #e0e0e0;
    }
    
    .footer p {
        margin: 0.5rem 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🩺 Diabetes Detection System</h1>
    <p>AI-Powered Early Screening & Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# Check if mobile view (simplified layout)
is_mobile = st.sidebar.checkbox("📱 Mobile View", value=True, help="Optimized for mobile devices")

if is_mobile:
    # Mobile-optimized single column layout
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("## 📝 Health Information")
    
    # Create input form
    with st.form("diabetes_prediction_form"):
        st.markdown("### 🔍 Key Health Metrics")
        
        # Single column for mobile
        pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.slider("🍯 Glucose Level (mg/dL)", 0, 200, 100)
        blood_pressure = st.slider("❤️ Blood Pressure (mm Hg)", 0, 140, 70)
        skin_thickness = st.slider("📏 Skin Thickness (mm)", 0, 100, 20)
        insulin = st.slider("💉 Insulin (μU/mL)", 0, 900, 85)
        bmi = st.slider("⚖️ BMI", 0.0, 70.0, 25.0)
        diabetes_pedigree = st.slider("🧬 Family History Score", 0.0, 2.5, 0.5)
        age = st.slider("🎂 Age", 10, 100, 30)
        
        # Submit button
        predict_button = st.form_submit_button("🧠 Analyze Risk")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Real-time metrics display
    st.markdown("## 📊 Health Indicators")
    
    # BMI Status
    if bmi < 18.5:
        bmi_status, bmi_color = "Underweight", "#3b82f6"
    elif bmi < 25:
        bmi_status, bmi_color = "Normal", "#16a34a"
    elif bmi < 30:
        bmi_status, bmi_color = "Overweight", "#f59e0b"
    else:
        bmi_status, bmi_color = "Obese", "#dc2626"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {bmi_color};">{bmi:.1f}</div>
        <div class="metric-label">BMI: {bmi_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Glucose Status
    if glucose < 100:
        glucose_status, glucose_color = "Normal", "#16a34a"
    elif glucose < 126:
        glucose_status, glucose_color = "Pre-diabetic", "#f59e0b"
    else:
        glucose_status, glucose_color = "High Risk", "#dc2626"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {glucose_color};">{glucose} mg/dL</div>
        <div class="metric-label">Glucose: {glucose_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Blood Pressure Status
    if blood_pressure < 80:
        bp_status, bp_color = "Normal", "#16a34a"
    elif blood_pressure < 90:
        bp_status, bp_color = "Elevated", "#f59e0b"
    else:
        bp_status, bp_color = "High", "#dc2626"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {bp_color};">{blood_pressure} mmHg</div>
        <div class="metric-label">BP: {bp_status}</div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Desktop layout (original two-column design)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("## 📝 Health Parameter Input")
        
        with st.form("diabetes_prediction_form"):
            st.markdown("### 🩺 Primary Health Indicators")
            col_a, col_b = st.columns(2)
            
            with col_a:
                pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, value=1)
                glucose = st.slider("🍯 Glucose (mg/dL)", 0, 200, 100)
                blood_pressure = st.slider("❤️ Blood Pressure (mm Hg)", 0, 140, 70)
                skin_thickness = st.slider("📏 Skin Thickness (mm)", 0, 100, 20)
            
            with col_b:
                insulin = st.slider("💉 Insulin (μU/mL)", 0, 900, 85)
                bmi = st.slider("⚖️ BMI", 0.0, 70.0, 25.0)
                diabetes_pedigree = st.slider("🧬 Family History", 0.0, 2.5, 0.5)
                age = st.slider("🎂 Age", 10, 100, 30)
            
            predict_button = st.form_submit_button("🧠 Analyze Risk")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("## 📊 Health Metrics")
        
        # Similar metric cards as mobile version
        if bmi < 18.5:
            bmi_status, bmi_color = "Underweight", "#3b82f6"
        elif bmi < 25:
            bmi_status, bmi_color = "Normal", "#16a34a"
        elif bmi < 30:
            bmi_status, bmi_color = "Overweight", "#f59e0b"
        else:
            bmi_status, bmi_color = "Obese", "#dc2626"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {bmi_color};">{bmi:.1f}</div>
            <div class="metric-label">BMI: {bmi_status}</div>
        </div>
        """, unsafe_allow_html=True)

# Prediction results
if predict_button:
    # Collect user data
    user_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }
    
    # Make prediction
    input_data = np.array([user_data[feat] for feat in feature_names]).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1] * 100
    
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.markdown("## 🎯 Risk Assessment Results")
    
    # Results display
    if prediction == 1:
        st.markdown(f"""
        <div class="risk-high">
            ⚠️ HIGH DIABETES RISK<br>
            <strong style="font-size: 1.5rem;">{proba:.1f}%</strong><br>
            Risk Probability
        </div>
        """, unsafe_allow_html=True)
        
        # High risk recommendations
        st.markdown("""
        <div class="recommendation-high">
            <h4>🚨 Immediate Action Required</h4>
            <ul>
                <li><strong>Schedule doctor consultation immediately</strong></li>
                <li>Request comprehensive diabetes screening (HbA1c, fasting glucose)</li>
                <li>Begin lifestyle modifications (diet, exercise)</li>
                <li>Monitor blood glucose regularly</li>
                <li>Consider family history discussion with healthcare provider</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown(f"""
        <div class="risk-low">
            ✅ LOW DIABETES RISK<br>
            <strong style="font-size: 1.5rem;">{100-proba:.1f}%</strong><br>
            Healthy Status
        </div>
        """, unsafe_allow_html=True)
        
        # Low risk recommendations
        st.markdown("""
        <div class="recommendation-low">
            <h4>✅ Preventive Health Measures</h4>
            <ul>
                <li>Maintain healthy diet and regular exercise</li>
                <li>Annual health check-ups and glucose screening</li>
                <li>Weight management (target BMI 18.5-24.9)</li>
                <li>Blood pressure monitoring</li>
                <li>Continue healthy lifestyle habits</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk meter visualization
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = proba,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score (%)", 'font': {'size': 16, 'color': '#1a1a1a'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickfont': {'color': '#1a1a1a'}},
            'bar': {'color': "#1a1a1a"},
            'steps': [
                {'range': [0, 30], 'color': "#16a34a"},
                {'range': [30, 70], 'color': "#f59e0b"},
                {'range': [70, 100], 'color': "#dc2626"}
            ],
            'threshold': {
                'line': {'color': "#1a1a1a", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color='#1a1a1a')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical information
    st.markdown("## 🔬 Model Information")
    st.markdown("""
    <div class="info-section">
        <h4>🤖 AI Model Details</h4>
        <p><strong>Algorithm:</strong> Ensemble Machine Learning (Random Forest + AdaBoost + XGBoost)</p>
        <p><strong>Dataset:</strong> PIMA Indian Diabetes Dataset (768 samples)</p>
        <p><strong>Feature Selection:</strong> Recursive Feature Elimination (RFE)</p>
        <p><strong>Validation:</strong> 5-fold cross-validation</p>
        <p><strong>Performance:</strong> Optimized for medical screening accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("## 📋 About This System")
    st.markdown("""
    **Early Detection of Diabetes Mellitus**
    
    This AI system uses ensemble machine learning to assess diabetes risk based on key health indicators.
    """)
    
    st.markdown("## 🎯 Key Features")
    st.markdown("""
    - **Ensemble ML**: Multiple algorithms for accuracy
    - **Feature Selection**: Optimized input processing
    - **Real-time Analysis**: Instant risk assessment
    - **Mobile Optimized**: Works on all devices
    """)
    
    st.markdown("## ⚠️ Important Notice")
    st.markdown("""
    This tool is for **screening purposes only** and should not replace professional medical advice. 
    
    Always consult with healthcare providers for proper diagnosis and treatment.
    """)

# Footer
st.markdown("""
<div class="footer">
    <p><strong>🩺 Diabetes Detection System | AI-Powered Health Screening</strong></p>
    <p>Research Project: Machine Learning for Early Disease Detection</p>
    <p>Last Updated: """ + datetime.now().strftime("%B %d, %Y") + """</p>
</div>
""", unsafe_allow_html=True)