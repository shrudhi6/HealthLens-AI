"""
HealthLens AI - Lab Report Analysis with Food & Fitness Recommendations
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
from modules.ai_recommendation import AIRecommendationEngine

# Set page config
st.set_page_config(
    page_title="HealthLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stDataFrame {
        width: 100% !important;
    }
    div[data-testid="stDataFrame"] > div {
        width: 100% !important;
        overflow-x: auto !important;
    }
    table {
        width: 100% !important;
        table-layout: fixed !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

# Import modules
@st.cache_resource
def initialize_modules():
    """Initialize OCR and ML modules"""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.getcwd(), 'modules'))
        
        from ocr_module import MedicalReportOCR, MedicalTextParser
        from ml_classifier import HealthConditionClassifier
        from ai_recommendation import AIRecommendationEngine

        
        ocr_engine = MedicalReportOCR()
        parser = MedicalTextParser()
        classifier = HealthConditionClassifier()
        rec_engine = AIRecommendationEngine()

        
        try:
            classifier.load_models()
        except:
            st.warning("⚠️ Pre-trained models not found. Train them first.")
        
        return {
            'ocr': ocr_engine,
            'parser': parser,
            'classifier': classifier,
            'recommender': rec_engine,
            'available': True
        }
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return {'available': False}

def main():
    st.markdown('<h1 class="main-header">🔬 HealthLens AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Lab Report Analysis with AI-Powered Recommendations</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔬 HealthLens")  # Fixed icon
        st.title("Navigation")
        page = st.radio("Go to", [
            "🏠 Home",
            "📤 Upload Report",
            "📊 Analysis Dashboard",
            "🍎 Nutrition Plan",
            "💪 Fitness Plan",
            "📅 Reminders"  # Added back
        ])
        st.divider()
        st.info("💡 Upload your lab report to get started!")
    
    # Route pages
    if page == "🏠 Home":
        show_home()
    elif page == "📤 Upload Report":
        show_upload()
    elif page == "📊 Analysis Dashboard":
        show_analysis()
    elif page == "🍎 Nutrition Plan":
        show_nutrition()
    elif page == "💪 Fitness Plan":
        show_fitness()
    elif page == "📅 Reminders":  # Added back
        show_reminders()
def show_home():
    st.header("Welcome to HealthLens AI! 👋")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 What We Do")
        st.write("""
        - 📄 Upload lab reports (PDF/Image)
        - 🔍 AI-powered analysis using Random Forest
        - 📊 Clear health insights
        - 🍎 Personalized diet plans
        - 💪 Custom fitness routines
        """)
    
    with col2:
        st.subheader("🏥 Conditions We Analyze")
        st.write("""
        - 🩸 Anemia
        - 🍬 Diabetes & Pre-diabetes
        - 💊 High Cholesterol
        """)

def show_upload():
    st.header("📤 Upload Your Lab Report")
    
    modules = initialize_modules()
    
    uploaded_file = st.file_uploader(
        "Choose your lab report",
        type=['pdf', 'jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            st.image(image, caption="Report Preview", use_column_width=True)
        
        if st.button("🔍 Extract Data from Report", type="primary"):
            if not modules['available']:
                st.error("❌ OCR not available. Use manual entry below.")
            else:
                with st.spinner("🔄 Processing..."):
                    try:
                        Path("uploads").mkdir(exist_ok=True)
                        temp_path = f"uploads/{uploaded_file.name}"
                        
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        ocr_engine = modules['ocr']
                        parser = modules['parser']
                        
                        extracted_text = ocr_engine.extract_text(temp_path)
                        
                        with st.expander("📄 View Extracted Text"):
                            st.text_area("Text", extracted_text, height=200)
                        
                        parsed_data = parser.parse_report(extracted_text)
                        patient_info = parser.extract_patient_info(extracted_text)
                        
                        if parsed_data:
                            st.success(f"✅ Extracted {len(parsed_data)} values!")
                            
                            data_rows = []
                            for test_name, test_data in parsed_data.items():
                                if isinstance(test_data, dict):
                                    data_rows.append({
                                        'Test': test_name.upper(),
                                        'Value': f"{test_data['value']} {test_data.get('unit', '')}",
                                        'Status': test_data.get('status', 'Unknown').upper()
                                    })
                            
                            if data_rows:
                                df = pd.DataFrame(data_rows)
                                st.table(df)
                                
                                st.session_state.analysis_results = {
                                    'parsed_values': parsed_data,
                                    'patient_info': patient_info,
                                    'timestamp': datetime.now()
                                }
                                
                                st.success("✅ Go to Analysis Dashboard!")
                        else:
                            st.warning("⚠️ No values found. Use manual entry.")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        st.divider()
    
    # Manual Entry
    st.subheader("Or Enter Values Manually")
    
    with st.form("manual_entry"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 0.0, 20.0, 13.5, 0.1)
            wbc = st.number_input("WBC (×10³/µL)", 0.0, 50.0, 7.5, 0.1)
            rbc = st.number_input("RBC (×10⁶/µL)", 0.0, 10.0, 5.0, 0.1)
        
        with col2:
            platelets = st.number_input("Platelets (×10³/µL)", 0, 1000, 250, 10)
            glucose = st.number_input("Glucose (mg/dL)", 0, 500, 95, 1)
            cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 400, 180, 1)
        
        with col3:
            hdl = st.number_input("HDL (mg/dL)", 0, 200, 50, 1)
            ldl = st.number_input("LDL (mg/dL)", 0, 300, 100, 1)
            triglycerides = st.number_input("Triglycerides (mg/dL)", 0, 500, 120, 1)
        
        if st.form_submit_button("🔍 Analyze", type="primary"):
            st.session_state.analysis_results = {
                'parsed_values': {
                    'hemoglobin': {'value': hemoglobin, 'unit': 'g/dL'},
                    'wbc': {'value': wbc, 'unit': '×10³/µL'},
                    'rbc': {'value': rbc, 'unit': '×10⁶/µL'},
                    'platelets': {'value': platelets, 'unit': '×10³/µL'},
                    'glucose': {'value': glucose, 'unit': 'mg/dL'},
                    'cholesterol': {'value': cholesterol, 'unit': 'mg/dL'},
                    'hdl': {'value': hdl, 'unit': 'mg/dL'},
                    'ldl': {'value': ldl, 'unit': 'mg/dL'},
                    'triglycerides': {'value': triglycerides, 'unit': 'mg/dL'}
                },
                'timestamp': datetime.now()
            }
            st.success("✅ Analysis complete!")
            st.balloons()

def show_analysis():
    st.header("📊 Analysis Dashboard")
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Upload a report first!")
        return
    
    results = st.session_state.analysis_results
    parsed_values = results.get('parsed_values', {})
    
    modules = initialize_modules()
    
    # Health Score
    test_results = {}
    for key, value in parsed_values.items():
        if isinstance(value, dict):
            test_results[key] = value['value']
        else:
            test_results[key] = value
    
    score = calculate_health_score(test_results)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.progress(score / 100)
        st.metric("Health Score", f"{score}/100")
    with col2:
        if score >= 80:
            st.success("✅ Excellent")
        elif score >= 60:
            st.warning("⚠️ Fair")
        else:
            st.error("🚨 Attention Needed")
    
    st.divider()
    
    # ML Predictions
    if modules['available']:
        st.subheader("🤖 AI Analysis (Random Forest)")
        
        try:
            classifier = modules['classifier']
            predictions = classifier.predict_conditions(test_results)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred = predictions.get('anemia', {})
                st.metric("ANEMIA", 
                         "DETECTED" if pred.get('prediction') else "NORMAL",
                         f"{pred.get('probability', 0):.1%}")
            
            with col2:
                pred = predictions.get('diabetes', {})
                st.metric("DIABETES",
                         "DETECTED" if pred.get('prediction') else "NORMAL",
                         f"{pred.get('probability', 0):.1%}")
            
            with col3:
                pred = predictions.get('cholesterol', {})
                st.metric("CHOLESTEROL",
                         "DETECTED" if pred.get('prediction') else "NORMAL",
                         f"{pred.get('probability', 0):.1%}")
            
            st.session_state.analysis_results['predictions'] = predictions
        
        except Exception as e:
            st.error(f"Prediction error: {e}")
    
    st.divider()
    
    # Test Results Table
    st.subheader("🔬 Test Results")
    
    data_rows = []
    for test_name, test_data in parsed_values.items():
        if isinstance(test_data, dict):
            value = test_data.get('value', 0)
            unit = test_data.get('unit', '')
            data_rows.append({
                'Test': test_name.upper(),
                'Value': f"{value} {unit}"
            })
    
    if data_rows:
        df = pd.DataFrame(data_rows)
        st.table(df)



def show_nutrition():
    st.header("🍎 Personalized Nutrition Plan (Gemini AI)")

    if not st.session_state.analysis_results:
        st.warning("⚠️ Please complete the health analysis first.")
        return

    predictions = st.session_state.analysis_results.get('predictions', {})
    if not predictions:
        st.info("💡 Run the AI health analysis first.")
        return

    detected_conditions = [c for c, d in predictions.items() if d.get('prediction', False)]
    if not detected_conditions:
        st.success("✅ All parameters look normal! Maintain a healthy balanced diet.")
        return

    # Initialize Gemini recommendation engine
    ai_engine = AIRecommendationEngine()

    st.info(f"🧠 Generating AI-powered diet plan for: {', '.join(detected_conditions)}...")

    with st.spinner("🍽️ Gemini is preparing your personalized plan..."):
        try:
            plan = ai_engine.generate_food_plan(detected_conditions, predictions)
            st.markdown(plan)
        except Exception as e:
            st.error(f"❌ Error generating AI recommendations: {e}")

def show_fitness():
    st.header("💪 Personalized Fitness Plan")
    
    if not st.session_state.analysis_results:
        st.warning("⚠️ Complete analysis first!")
        return
    
    st.success("🎉 Your fitness plan is ready!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Daily Activity", "30 min")
    with col2:
        st.metric("Weekly Goal", "150 min")
    with col3:
        st.metric("Intensity", "Moderate")
    
    st.divider()
    
    st.subheader("📅 Weekly Plan")
    
    schedule = pd.DataFrame({
        'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'Activity': ['Walking', 'Strength', 'Yoga', 'Cycling', 'Swimming', 'Strength', 'Rest'],
        'Duration': ['30 min'] * 7
    })
    
    st.table(schedule)

def calculate_health_score(results):
    score = 100
    
    hb = results.get('hemoglobin', 14)
    if hb < 13.0 or hb > 17.0:
        score -= 15
    
    glucose = results.get('glucose', 95)
    if glucose > 100:
        score -= 20
    elif glucose > 125:
        score -= 30
    
    cholesterol = results.get('cholesterol', 180)
    if cholesterol > 200:
        score -= 15
    elif cholesterol > 240:
        score -= 25
    
    return max(0, min(100, score))
def show_reminders():
    st.header("📅 Health Reminders & Appointments")
    
    st.info("🔔 Stay on top of your health with automated reminders!")
    
    # Reminder settings
    st.subheader("⚙️ Reminder Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        reminder_type = st.selectbox(
            "Reminder Type",
            ["Monthly Checkup", "Weekly Medication", "Daily Exercise", "Hydration Alert"]
        )
        reminder_time = st.time_input("Preferred Time", datetime.now().time())
    
    with col2:
        reminder_method = st.multiselect(
            "Notification Method",
            ["Email", "SMS", "App Notification"],
            default=["Email"]
        )
        if st.button("➕ Add Reminder"):
            st.success("✅ Reminder added successfully!")
    
    st.divider()
    
    # Upcoming appointments
    st.subheader("📋 Upcoming Appointments")
    
    appointments = pd.DataFrame({
        "Date": [
            (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        ],
        "Type": ["Follow-up", "Lab Test", "Consultation"],
        "Doctor": ["Dr. Smith", "Lab Center", "Dr. Johnson"],
        "Status": ["Scheduled", "Pending", "Scheduled"]
    })
    
    st.table(appointments)
    
    st.divider()
    
    # Next checkup
    st.subheader("📆 Next Lab Checkup")
    
    if st.session_state.analysis_results:
        last_test = st.session_state.analysis_results.get('timestamp', datetime.now())
        next_test = last_test + timedelta(days=30)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Last Checkup", last_test.strftime("%b %d, %Y"))
        with col2:
            st.metric("Next Checkup", next_test.strftime("%b %d, %Y"))
        with col3:
            days_until = (next_test - datetime.now()).days
            st.metric("Days Until", f"{days_until} days")
    else:
        st.info("💡 Upload a report to track your checkup schedule!")
if __name__ == "__main__":
    main()
   