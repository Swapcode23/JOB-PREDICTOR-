import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ─── Page Configuration ───
st.set_page_config(
    page_title="Placement Readiness & Job Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: bold; color: #1f4e79; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #555; text-align: center; margin-bottom: 2rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1rem; }
    .role-card { background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #667eea; margin-bottom: 0.8rem; }
    .gap-card { background: #fff3cd; padding: 0.8rem; border-radius: 8px; border-left: 4px solid #ffc107; margin-bottom: 0.5rem; }
    .strength-card { background: #d4edda; padding: 0.8rem; border-radius: 8px; border-left: 4px solid #28a745; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───
BRANCHES = ['CSE', 'Mechanical', 'Civil', 'Chemical', 'Metallurgy', 'Electrical', 'ECE', 'AI']
STATUSES = ['4th year student', 'Alumni (Graduated)']
TOOLS_LIST = [
    'Machine Learning / Deep Learning', 'Data Analysis (Python / Excel / Pandas)',
    'Data Visualization (PowerBI / Tableau)', 'CAD tools (SolidWorks / CATIA / AutoCAD)',
    'ANSYS / Simulation tools', 'MATLAB / Simulink', 'Embedded systems / microcontrollers', 'Circuit design tools'
]
PROJECT_COUNTS = ['0', '1-2', '3-4', '5+']
PROJECT_DOMAINS = ['Software Development', 'Data Science / AI', 'Core Engineering', 'Robotics / Embedded Systems', 'Mixed domains']
INTERNSHIPS = ['No internship', 'Software Development Internship', 'Data Science / AI Internship', 'Core Engineering Internship', 'Electronics / Embedded Internship']
PREP_DOMAINS = ['Software Development', 'Data Science / AI', 'Core Engineering', 'Embedded Systems / Electronics', 'Consulting / Management']

# ─── Load Model Artifacts ───
@st.cache_resource
def load_model_artifacts():
    artifacts = {}
    required_files = {
        'model': 'best_model.pkl',
        'scaler': 'scaler.pkl',
        'label_encoders': 'label_encoders.pkl',
        'target_encoder': 'target_encoder.pkl',
        'mlb': 'mlb_tools.pkl',
        'feature_columns': 'feature_columns.pkl'
    }
    for key, filename in required_files.items():
        filepath = os.path.join(os.getcwd(), filename)
        if not os.path.exists(filepath):
            st.error(f"Missing file: {filename}. Please ensure it is uploaded to your GitHub repository.")
            return None
        with open(filepath, 'rb') as f:
            artifacts[key] = pickle.load(f)
    return artifacts

# ─── Main App ───
def main():
    st.markdown('<div class="main-header">Placement Readiness & Job Role Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered career guidance for engineering students</div>', unsafe_allow_html=True)
    st.markdown("---")

    artifacts = load_model_artifacts()
    if artifacts is None:
        st.stop()

    st.header("Enter Your Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        branch = st.selectbox("1. Engineering Branch", BRANCHES)
        status = st.selectbox("2. Your Status", STATUSES)
        cgpa = st.slider("3. CGPA", 4.0, 10.0, 7.5, 0.1)

    with col2:
        st.markdown("**4. Programming Proficiency (1-5)**")
        python_prof = st.slider("Python", 1, 5, 3)
        cpp_prof = st.slider("C/C++", 1, 5, 3)
        java_prof = st.slider("Java", 1, 5, 2)
        matlab_prof = st.slider("MATLAB", 1, 5, 2)

    with col3:
        st.markdown("**5. Conceptual Understanding (1-5)**")
        dsa = st.slider("DSA", 1, 5, 3)
        sql = st.slider("Database / SQL", 1, 5, 3)
        oop = st.slider("OOP", 1, 5, 3)
        os_und = st.slider("Operating Systems", 1, 5, 3)

    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        tools = st.multiselect("6. Technical Tools", TOOLS_LIST)
        project_count = st.selectbox("7. Number of Projects", PROJECT_COUNTS, index=2)
        project_domain = st.selectbox("8. Project Domain", PROJECT_DOMAINS)

    with col5:
        internship = st.selectbox("9. Internship Experience", INTERNSHIPS)
        prep_domain = st.selectbox("10. Preparation Domain", PREP_DOMAINS)
        confidence = st.slider("11. Confidence Level (1-5)", 1, 5, 3)

    st.markdown("---")
    predict_clicked = st.button("Predict My Job Roles", type="primary", use_container_width=True)

    if predict_clicked:
        if not tools:
            st.warning("Please select at least one technical tool.")
            st.stop()
        
        # In a real deployment, the predict_roles() function logic goes here.
        # For layout purposes, we render a success box.
        st.success("Model loaded and UI successfully rendered! (Backend prediction logic executes here)")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.85rem;'>"
        "Applied Machine Learning (22MET921) | Dr. Gunjan Soni | Swapnil Acharya, Aditya Verma, Amit Kumar"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
