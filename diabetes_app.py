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
    initial_sidebar_state="collapsed"
)

# Clean, mobile-optimized CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f2f6;
    }
    
    /* Metric cards */
    .metric-card {
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid #e1e5e9;
        background: #fafafa;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.7;
        font-weight: 500;
    }
    
    /* Risk result styling */
    .risk-result {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .risk-high {
        background: #fee2e2;
        color: #dc2626;
        border: 2px solid #fecaca;
    }
    
    .risk-low {
        background: #dcfce7;
        color: #16a34a;
        border: 2px solid #bbf7d0;
    }
    
    /* Recommendation boxes */
    .recommendation {
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .recommendation-high {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        color: #7f1d1d;
    }
    
    .recommendation-low {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        color: #14532d;
    }
    
    .recommendation h4 {
        margin: 0 0 1rem 0;
        font-weight: 600;
    }
    
    .recommendation ul {
        margin: 0;
        padding-left: 1.2rem;
    }
    
    .recommendation li {
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    /* Info section */
    .info-section {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .info-section h4 {
        margin: 0 0 1rem 0;
        font-weight: 600;
        color: #1e293b;
    }
    
    .info-section p {
        margin: 0.5rem 0;
        line-height: 1.5;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 1rem;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
        color: #64748b;
    }
    
    .footer p {
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    /* Mobile optimizations */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .main-header {
            padding: 1.5rem 1rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.6rem;
        }
        
        .risk-result {
            padding: 1.5rem;
            font-size: 1.1rem;
        }
        
        .recommendation, .info-section {
            padding: 1rem;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: #1f2937;
            border-color: #374151;
            color: #f9fafb;
        }
        
        .risk-high {
            background: #7f1d1d;
            color: #fecaca;
            border-color: #991b1b;
        }
        
        .risk-low {
            background: #14532d;
            color: #bbf7d0;
            border-color: #166534;
        }
        
        .recommendation-high {
            background: #1f2937;
            border-left-color: #ef4444;
            color: #fca5a5;
        }
        
        .recommendation-low {
            background: #1f2937;
            border-left-color: #22c55e;
            color: #86efac;
        }
        
        .info-section {
            background: #1f2937;
            border-color: #374151;
            color: #f9fafb;
        }
        
        .info-section h4 {
            color: #f9fafb;
        }
        
        .section-header {
            color: #f9fafb;
            border-bottom-color: #374151;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🩺 Detection of Diabetes System</h1>
    <p>Early Detection of Diabetes Mellitus using Feature Selection and Ensemble Models</p>
</div>
""", unsafe_allow_html=True)

# Check if mobile view (simplified layout)
is_mobile = st.sidebar.checkbox("📱 Mobile View", value=True, help="Optimized for mobile devices")

if is_mobile:
    # Mobile-optimized single column layout
    st.markdown('<h2 class="section-header">📝 Health Information</h2>', unsafe_allow_html=True)
    
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
    
    # Real-time metrics display
    st.markdown('<h2 class="section-header">📊 Health Indicators</h2>', unsafe_allow_html=True)
    
    # BMI Status
    if bmi < 18.5:
        bmi_status, bmi_color = "Underweight", "#3b82f6"
    elif bmi < 25:
        bmi_status, bmi_color = "Normal", "#10b981"
    elif bmi < 30:
        bmi_status, bmi_color = "Overweight", "#f59e0b"
    else:
        bmi_status, bmi_color = "Obese", "#ef4444"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {bmi_color};">{bmi:.1f}</div>
        <div class="metric-label">BMI: {bmi_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Glucose Status
    if glucose < 100:
        glucose_status, glucose_color = "Normal", "#10b981"
    elif glucose < 126:
        glucose_status, glucose_color = "Pre-diabetic", "#f59e0b"
    else:
        glucose_status, glucose_color = "High Risk", "#ef4444"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {glucose_color};">{glucose} mg/dL</div>
        <div class="metric-label">Glucose: {glucose_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Blood Pressure Status
    if blood_pressure < 80:
        bp_status, bp_color = "Normal", "#10b981"
    elif blood_pressure < 90:
        bp_status, bp_color = "Elevated", "#f59e0b"
    else:
        bp_status, bp_color = "High", "#ef4444"
    
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
        st.markdown('<h2 class="section-header">📝 Health Parameter Input</h2>', unsafe_allow_html=True)
        
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
    
    with col2:
        st.markdown('<h2 class="section-header">📊 Health Metrics</h2>', unsafe_allow_html=True)
        
        # Similar metric cards as mobile version
        if bmi < 18.5:
            bmi_status, bmi_color = "Underweight", "#3b82f6"
        elif bmi < 25:
            bmi_status, bmi_color = "Normal", "#10b981"
        elif bmi < 30:
            bmi_status, bmi_color = "Overweight", "#f59e0b"
        else:
            bmi_status, bmi_color = "Obese", "#ef4444"
        
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
    
    st.markdown('<h2 class="section-header">🎯 Risk Assessment Results</h2>', unsafe_allow_html=True)
    
    # Results display
    if prediction == 1:
        st.markdown(f"""
        <div class="risk-result risk-high">
            ⚠️ HIGH DIABETES RISK<br>
            <strong style="font-size: 1.8rem;">{proba:.1f}%</strong><br>
            Risk Probability
        </div>
        """, unsafe_allow_html=True)
        
        # High risk recommendations
        st.markdown("""
        <div class="recommendation recommendation-high">
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
        <div class="risk-result risk-low">
            ✅ LOW DIABETES RISK<br>
            <strong style="font-size: 1.8rem;">{100-proba:.1f}%</strong><br>
            Healthy Status
        </div>
        """, unsafe_allow_html=True)
        
        # Low risk recommendations
        st.markdown("""
        <div class="recommendation recommendation-low">
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
    
    # Enhanced risk meter visualization - mobile optimized
    # Get theme colors for better visibility
    theme_colors = {
        'background': 'rgba(0,0,0,0)',
        'text': '#262730',
        'grid': '#e1e5e9'
    }
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = proba,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': "Diabetes Risk Score (%)", 
            'font': {'size': 18, 'color': theme_colors['text'], 'family': 'Inter'}
        },
        number = {
            'font': {'size': 32, 'color': theme_colors['text'], 'family': 'Inter'},
            'suffix': '%'
        },
        gauge = {
            'axis': {
                'range': [None, 100], 
                'tickwidth': 2,
                'tickcolor': theme_colors['grid'],
                'tickfont': {'color': theme_colors['text'], 'size': 14}
            },
            'bar': {'color': "#667eea", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': theme_colors['grid'],
            'steps': [
                {'range': [0, 30], 'color': "#dcfce7"},
                {'range': [30, 70], 'color': "#fef3c7"},
                {'range': [70, 100], 'color': "#fee2e2"}
            ],
            'threshold': {
                'line': {'color': theme_colors['text'], 'width': 4},
                'thickness': 0.8,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor=theme_colors['background'],
        plot_bgcolor=theme_colors['background'],
        font=dict(color=theme_colors['text'], family='Inter'),
        autosize=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add risk interpretation
    st.markdown("### 📊 Risk Interpretation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #dcfce7; border-radius: 8px; margin: 0.5rem 0;">
            <strong style="color: #166534;">Low Risk</strong><br>
            <span style="color: #166534; font-size: 0.9rem;">0-30%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #fef3c7; border-radius: 8px; margin: 0.5rem 0;">
            <strong style="color: #92400e;">Moderate Risk</strong><br>
            <span style="color: #92400e; font-size: 0.9rem;">30-70%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #fee2e2; border-radius: 8px; margin: 0.5rem 0;">
            <strong style="color: #dc2626;">High Risk</strong><br>
            <span style="color: #dc2626; font-size: 0.9rem;">70-100%</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical information
    st.markdown('<h2 class="section-header">🔬 Model Information</h2>', unsafe_allow_html=True)
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