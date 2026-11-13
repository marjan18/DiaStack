import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go



# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ✅ FIX: Initialize session state BEFORE anything else
if 'prediction_threshold' not in st.session_state:
    st.session_state.prediction_threshold = 0.4


if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False



# ✅ ENHANCED UI: Modern, Eye-catching CSS
st.markdown("""
    <style>
    /* Main theme */
    :root {
        --primary-color: #2E7D32;
        --danger-color: #D32F2F;
        --warning-color: #F57C00;
        --info-color: #1976D2;
    }
    
    .main {
        padding: 20px;
    }
    
    /* Risk Score Card - BIG and Eye-catching */
    .risk-score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    
    .risk-score-value {
        font-size: 48px;
        font-weight: bold;
        margin: 15px 0;
    }
    
    .risk-score-label {
        font-size: 18px;
        opacity: 0.9;
        margin-bottom: 10px;
    }
    
    /* High Risk Alert */
    .high-risk-alert {
        background: linear-gradient(135deg, #D32F2F 0%, #B71C1C 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(211, 47, 47, 0.3);
        margin: 15px 0;
    }
    
    .high-risk-text {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    /* Low Risk Success */
    .low-risk-success {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(46, 125, 50, 0.3);
        margin: 15px 0;
    }
    
    .low-risk-text {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    /* Enhanced Metrics */
    .metric-card {
        background: #f5f5f5;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border-left: 5px solid;
    }
    
    .metric-card.primary {
        border-left-color: #1976D2;
    }
    
    .metric-card.success {
        border-left-color: #2E7D32;
    }
    
    .metric-card.danger {
        border-left-color: #D32F2F;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #333;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 13px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Risk Category Badge */
    .risk-category {
        display: inline-block;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 16px;
        margin: 10px 5px;
    }
    
    .risk-category.high {
        background-color: #FFEBEE;
        color: #D32F2F;
        border: 2px solid #D32F2F;
    }
    
    .risk-category.low {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 2px solid #2E7D32;
    }
    
    /* Input section styling */
    .input-section {
        background: #F9F9F9;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    .input-title {
        font-size: 18px;
        font-weight: bold;
        color: #333;
        margin-bottom: 15px;
        border-bottom: 3px solid #1976D2;
        padding-bottom: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 40px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #333;
        margin-top: 20px;
    }
    
    h1 {
        border-bottom: 4px solid #667eea;
        padding-bottom: 15px;
    }
    
    </style>
    """, unsafe_allow_html=True)



# Load model
@st.cache_resource
def load_model():
    with open('diabetes_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data



# ✅ ENHANCED: Modern Title with Icons
st.markdown("""
<div style='text-align: center; padding: 30px 0;'>
    <h1 style='margin: 0;'>🏥 Diabetes Risk Predictor</h1>
    <p style='color: #666; font-size: 16px; margin: 10px 0;'>
        Advanced AI-Powered Health Assessment System
    </p>
</div>
""", unsafe_allow_html=True)


try:
    model_data = load_model()
    model = model_data['model']
    scaler = model_data['scaler_train']
    label_encoders = model_data['label_encoders']
    feature_columns = model_data['feature_columns']
    
    st.success("✅ System Ready - Model Loaded Successfully")
    
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()



# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Assessment", "📊 Model Info", "⚙️ Settings", "❓ Help"])



# ============================================================================
# TAB 3: SETTINGS (Place FIRST so threshold is set before Prediction tab)
# ============================================================================
with tab3:
    st.header("Sensitivity Settings")
    
    st.markdown("""
    ### Risk Detection Sensitivity
    Adjust how sensitive the system is when detecting diabetes risk.
    """)
    
    st.markdown("""
    - **🔴 Very Sensitive (0.2-0.3):** Catches maximum cases (more alerts)
    - **🟡 Balanced (0.35-0.4):** Recommended for screening
    - **🟢 Conservative (0.45-0.9):** Only high confidence cases
    """)
    
    # ✅ FIX: Slider updates session state directly
    st.session_state.prediction_threshold = st.slider(
        "Sensitivity Level",
        min_value=0.1,
        max_value=0.9,
        value=st.session_state.prediction_threshold,
        step=0.05,
        help="Lower = More Sensitive, Higher = More Conservative",
        key="threshold_slider"
    )
    
    # Display threshold info
    col_threshold_info = st.columns(3)
    
    with col_threshold_info[0]:
        st.metric("Current Setting", f"{st.session_state.prediction_threshold:.2f}")
    
    with col_threshold_info[1]:
        if st.session_state.prediction_threshold <= 0.3:
            st.warning("🔴 Very Sensitive")
        elif st.session_state.prediction_threshold <= 0.4:
            st.info("🟡 Balanced")
        else:
            st.success("🟢 Conservative")
    
    st.divider()
    
    st.markdown("""
    ### How It Works
    
    The system analyzes patient data and calculates a risk score (0-100%).
    
    **Example:** If risk score is 40%
    - Sensitivity 0.3 (30%):  **ALERT** → Risk ≥ 30%
    - Sensitivity 0.4 (40%):  **ALERT** → Risk ≥ 40%
    - Sensitivity 0.5 (50%):  **No Alert** → Risk < 50%
    """)



# ============================================================================
# TAB 1: ASSESSMENT (PREDICTION)
# ============================================================================
with tab1:
    st.header("Patient Assessment")
    
    # ✅ FIX: Get threshold from session state
    threshold = st.session_state.prediction_threshold
    
    st.info(f"⚙️ Current Sensitivity: **{threshold:.0%}** | [Adjust in Settings]()")
    
    # ✅ ENHANCED: Two-column layout with better organization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Basic Information")
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=45,
            step=1,
            help="Patient's age in years"
        )
        
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
            help="Patient's gender"
        )
        
        st.markdown("#### 💓 Cardiovascular Metrics")
        systolic_bp = st.number_input(
            "Systolic BP (mmHg)",
            min_value=70,
            max_value=250,
            value=120,
            step=1
        )
        
        diastolic_bp = st.number_input(
            "Diastolic BP (mmHg)",
            min_value=40,
            max_value=150,
            value=80,
            step=1
        )
        
        pulse_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=40,
            max_value=200,
            value=72,
            step=1
        )
    
    with col2:
        st.markdown("#### 🧬 Metabolic Indicators")
        glucose = st.number_input(
            "Glucose (mmol/L)",
            min_value=2.5,
            max_value=25.0,
            value=5.5,
            step=0.1
        )
        
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=10.0,
            max_value=60.0,
            value=24.0,
            step=0.1
        )
        
        st.markdown("#### ⚠️ Risk Factors")
        family_diabetes = st.checkbox(
            "Family History of Diabetes",
            value=False
        )
        
        hypertensive = st.checkbox(
            "Hypertension Diagnosis",
            value=False
        )
        
        family_hypertension = st.checkbox(
            "Family History of Hypertension",
            value=False
        )
        
        cardiovascular_disease = st.checkbox(
            "Cardiovascular Disease",
            value=False
        )
        
        stroke = st.checkbox(
            "History of Stroke",
            value=False
        )
    
    # ✅ ENHANCED: Large, centered predict button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        predict_button = st.button(
            "🔍 ANALYZE RISK",
            use_container_width=True,
            key="predict_btn"
        )
    
    if predict_button:
        try:
            # Create input data
            input_data = {
                'age': age,
                'gender': gender,
                'pulse_rate': pulse_rate,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'glucose': glucose,
                'bmi': bmi,
                'family_diabetes': 1 if family_diabetes else 0,
                'hypertensive': 1 if hypertensive else 0,
                'family_hypertension': 1 if family_hypertension else 0,
                'cardiovascular_disease': 1 if cardiovascular_disease else 0,
                'stroke': 1 if stroke else 0
            }
            
            df_input = pd.DataFrame([input_data])
            df_input = df_input[feature_columns]
            df_input['gender'] = label_encoders['gender'].transform(df_input['gender'])
            df_scaled = scaler.transform(df_input)
            df_prediction = pd.DataFrame(df_scaled, columns=feature_columns)
            
            # ✅ FIX: Get probabilities and apply threshold correctly
            prediction_proba = model.predict_proba(df_prediction)[0]
            
            # Apply custom threshold from session state
            prediction = 1 if prediction_proba[1] >= threshold else 0
            
            risk_percentage = prediction_proba[1] * 100
            
            # ✅ ENHANCED: Eye-catching results display
            st.markdown("---")
            st.markdown("<h2 style='text-align: center; margin: 30px 0;'>📋 ASSESSMENT RESULTS</h2>", unsafe_allow_html=True)
            
            # ✅ BIG RISK SCORE CARD
            if prediction == 1:
                st.markdown(f"""
                <div class='high-risk-alert'>
                    <div class='risk-score-label'>⚠️ RISK ASSESSMENT</div>
                    <div class='high-risk-text'>HIGH RISK DETECTED</div>
                    <div style='font-size: 24px; margin-top: 15px;'>Risk Level: {risk_percentage:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='low-risk-success'>
                    <div class='risk-score-label'>✅ RISK ASSESSMENT</div>
                    <div class='low-risk-text'>LOW RISK</div>
                    <div style='font-size: 24px; margin-top: 15px;'>Risk Level: {risk_percentage:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ✅ ENHANCED: Three key metrics instead of confidence
            st.markdown("<h3 style='text-align: center; margin: 30px 0;'>Key Metrics</h3>", unsafe_allow_html=True)
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown(f"""
                <div class='metric-card primary'>
                    <div class='metric-label'>Risk Score</div>
                    <div class='metric-value'>{risk_percentage:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                sensitivity_pct = threshold * 100
                st.markdown(f"""
                <div class='metric-card primary'>
                    <div class='metric-label'>Detection Level</div>
                    <div class='metric-value'>{sensitivity_pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                certainty = max(prediction_proba) * 100
                st.markdown(f"""
                <div class='metric-card primary'>
                    <div class='metric-label'>Certainty</div>
                    <div class='metric-value'>{certainty:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ✅ Decision logic
            st.markdown("---")
            st.markdown("### Decision Analysis")
            
            decision_text = f"""
            **Risk Score:** {risk_percentage:.1f}%  
            **Detection Threshold:** {threshold*100:.0f}%  
            **Comparison:** {risk_percentage:.1f}% {'≥' if prediction == 1 else '<'} {threshold*100:.0f}%  
            **Status:** {'⚠️ **ALERT - Recommend Medical Consultation**' if prediction == 1 else '✅ **Within Safe Range**'}
            """
            
            if prediction == 1:
                st.warning(decision_text)
            else:
                st.success(decision_text)
            
            # ✅ ENHANCED: Better visualization
            st.markdown("---")
            st.markdown("### Risk Distribution")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['No Diabetes Risk', 'Diabetes Risk'],
                    y=[prediction_proba[0]*100, prediction_proba[1]*100],
                    marker=dict(
                        color=['#2E7D32', '#D32F2F'],
                        line=dict(color='rgba(0,0,0,0.3)', width=2)
                    ),
                    text=[f'{v:.1f}%' for v in [prediction_proba[0]*100, prediction_proba[1]*100]],
                    textposition='auto',
                    textfont=dict(size=14, color='white'),
                )
            ])
            
            # Add threshold line
            fig.add_hline(
                y=threshold*100, 
                line_dash="dash", 
                line_color="orange",
                line_width=3,
                annotation_text=f"Detection Threshold: {threshold*100:.0f}%",
                annotation_position="right",
                annotation_font_size=12,
                annotation_font_color="orange"
            )
            
            fig.update_layout(
                title="Risk Probability Analysis",
                xaxis_title="Risk Category",
                yaxis_title="Probability (%)",
                height=450,
                showlegend=False,
                yaxis=dict(range=[0, 100]),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(240,240,240,1)',
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ✅ Patient summary
            st.markdown("---")
            st.markdown("### Patient Data Summary")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.markdown(f"""
                **Demographics**
                - Age: {age} years
                - Gender: {gender}
                - BMI: {bmi:.1f} kg/m²
                """)
            
            with summary_col2:
                st.markdown(f"""
                **Vitals**
                - Glucose: {glucose:.1f} mmol/L
                - Systolic BP: {systolic_bp} mmHg
                - Diastolic BP: {diastolic_bp} mmHg
                """)
            
            with summary_col3:
                risk_factors_count = sum([family_diabetes, hypertensive, family_hypertension, cardiovascular_disease, stroke])
                st.markdown(f"""
                **Risk Factors**
                - Count: {risk_factors_count}/5
                - Family Diabetes: {'Yes' if family_diabetes else 'No'}
                - Hypertension: {'Yes' if hypertensive else 'No'}
                """)
            
            st.session_state.prediction_made = True
            
        except Exception as e:
            st.error(f"❌ Analysis Error: {e}")
            import traceback
            st.error(traceback.format_exc())



# ============================================================================
# TAB 2: MODEL INFO
# ============================================================================
with tab2:
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Architecture")
        st.info("""
        **Ensemble Learning**
        
        - Random Forest (50 trees)
        - Gradient Boosting (50 rounds)
        - Support Vector Machine
        - Logistic Regression (meta-learner)
        
        **Validation:** 5-Fold Cross-Validation
        """)
    
    with col2:
        st.markdown("### Performance Metrics")
        st.success("""
        **Accuracy:** 93.76%
        
        **Class 0 (No Diabetes)**
        - Precision: 94%
        - Recall: 99%
        
        **Class 1 (Diabetes)**
        - Precision: 71%
        - Recall: 22%
        """)
    
    st.warning("""
    ⚠️ **Important Notes:**
    - Dataset Imbalance: 95% healthy, 5% diabetic
    - Adjust sensitivity in Settings for your use case
    - This is a screening tool, not a diagnosis
    - Always consult healthcare professionals
    """)
    
    st.subheader("Input Features (12 Total)")
    
    feature_info = {
        'Feature': feature_columns,
        'Type': ['Age', 'Gender', 'Heart Rate', 'Systolic BP', 
                'Diastolic BP', 'Glucose', 'BMI', 'Family Diabetes', 
                'Hypertension', 'Family Hypertension', 'Cardiovascular Disease', 'Stroke']
    }
    
    st.dataframe(pd.DataFrame(feature_info), use_container_width=True, hide_index=True)



# ============================================================================
# TAB 4: HELP
# ============================================================================
with tab4:
    st.header("❓ Help & Guide")
    
    st.markdown("""
    ### What is this system?
    An AI-powered diabetes risk assessment tool that analyzes patient health metrics
    and provides personalized risk scores.
    
    ### How to use?
    1. **Enter patient information** - Fill all health metrics
    2. **Adjust sensitivity** (optional) - Settings tab controls detection level
    3. **Click ANALYZE RISK** - Get instant assessment
    4. **Review results** - Understand the risk level and metrics
    
    ### Understanding Risk Levels
    - **Risk Score:** 0-100% probability of diabetes
    - **Detection Level:** Sensitivity threshold (lower = catches more cases)
    - **Certainty:** How confident the system is in its prediction
    
    ### Key Metrics Explained
    | Metric | Meaning |
    |--------|---------|
    | Risk Score | Percentage chance of diabetes (0-100%) |
    | Detection Level | Current sensitivity setting |
    | Certainty | Confidence in the prediction |
    
    ### Sensitivity Guidelines
    - **20-30%** 🔴 Maximum sensitivity (catches most, more alerts)
    - **35-40%** 🟡 Recommended for screening (balanced)
    - **45-50%** 🟢 Conservative (fewer false alarms)
    
    ### Input Ranges
    - Age: 18-120 years
    - BP: 70-250 systolic, 40-150 diastolic
    - Heart Rate: 40-200 bpm
    - Glucose: 2.5-25.0 mmol/L
    - BMI: 10.0-60.0 kg/m²
    """)
    
    st.warning("""
    ⚠️ **Disclaimer:**
    - Educational and screening purposes only
    - Not a medical diagnosis tool
    - Always consult qualified healthcare professionals
    - This system complements, not replaces, medical advice
    """)



# ✅ ENHANCED: Modern footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <p style='margin: 5px;'>🏥 <b>Diabetes Risk Prediction System</b></p>
    <p style='margin: 5px; font-size: 12px;'>Built with Streamlit & Machine Learning</p>
    <p style='margin: 5px; font-size: 12px;'>✅ Enhanced UI | 🎯 Better Metrics | 📊 Eye-Catching Visuals</p>
</div>
""", unsafe_allow_html=True)
