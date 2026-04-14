"""
=============================================================================
PLACEMENT READINESS & JOB ROLE PREDICTOR — STREAMLIT DASHBOARD
=============================================================================
Run with:   streamlit run app.py
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Placement Readiness Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f0f4f8; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    section[data-testid="stSidebar"] .stSlider > label { color: #a0c4ff !important; font-size: 0.85rem !important; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #a0c4ff !important; }
    section[data-testid="stSidebar"] .stSelectbox label { color: #a0c4ff !important; }

    /* Role card */
    .role-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-top: 5px solid;
        height: 100%;
        transition: transform 0.2s;
    }
    .role-card:hover { transform: translateY(-4px); }
    .role-card.gold  { border-color: #f59e0b; }
    .role-card.silver{ border-color: #6b7280; }
    .role-card.bronze{ border-color: #92400e; }
    .role-title { font-size: 1.1rem; font-weight: 700; margin: 8px 0 4px; color: #1e293b; }
    .role-conf  { font-size: 2rem; font-weight: 800; margin: 8px 0; }
    .role-badge { font-size: 0.75rem; background: #e2e8f0; color: #475569; padding: 3px 10px; border-radius: 20px; }
    .gold  .role-conf { color: #f59e0b; }
    .silver.role-conf { color: #6b7280; }
    .bronze.role-conf { color: #92400e; }
    .gold  .role-conf { color: #f59e0b; }
    .silver .role-conf { color: #6b7280; }
    .bronze .role-conf { color: #92400e; }

    /* Section headers */
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #1e293b;
        border-left: 5px solid #3b82f6; padding-left: 12px;
        margin: 24px 0 16px;
    }

    /* Metric box */
    .metric-box {
        background: white; border-radius: 12px; padding: 16px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06); text-align: center;
    }

    /* Roadmap step */
    .roadmap-step {
        background: white; border-radius: 12px; padding: 16px 20px;
        margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #3b82f6;
    }
    .step-phase { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
    .step-title { font-size: 1rem; font-weight: 700; color: #1e293b; margin: 2px 0; }
    .step-desc  { font-size: 0.85rem; color: #475569; line-height: 1.5; }

    /* Course card */
    .course-card {
        background: white; border-radius: 10px; padding: 14px 16px;
        margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        display: flex; align-items: flex-start; gap: 12px;
    }
    .course-platform { font-size: 0.7rem; background: #dbeafe; color: #1d4ed8;
        padding: 2px 8px; border-radius: 12px; font-weight: 600; }
    .course-title { font-size: 0.9rem; font-weight: 600; color: #1e293b; }
    .course-desc  { font-size: 0.8rem; color: #64748b; }

    /* Welcome banner */
    .welcome-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 16px; padding: 28px 36px; color: white; margin-bottom: 24px;
    }
    .welcome-banner h1 { color: white; font-size: 1.8rem; margin: 0 0 6px; }
    .welcome-banner p  { color: #bfdbfe; margin: 0; font-size: 1rem; }

    /* Gap bar */
    .gap-row { margin-bottom: 10px; }
    .gap-label { font-size: 0.85rem; color: #334155; font-weight: 500; margin-bottom: 3px; }
    .gap-bar-bg { background: #e2e8f0; border-radius: 8px; height: 10px; overflow: hidden; }
    .gap-bar-fill { height: 10px; border-radius: 8px; }
    .gap-status-good { color: #16a34a; font-size: 0.75rem; }
    .gap-status-warn { color: #f59e0b; font-size: 0.75rem; }
    .gap-status-bad  { color: #dc2626; font-size: 0.75rem; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────
TOOLS_LIST = [
    'Machine Learning / Deep Learning',
    'Data Analysis (Python / Excel / Pandas)',
    'Data Visualization (PowerBI / Tableau)',
    'CAD tools (SolidWorks / CATIA / AutoCAD)',
    'ANSYS / Simulation tools',
    'MATLAB / Simulink',
    'Embedded systems / microcontrollers',
    'Circuit design tools'
]

BRANCHES       = ['CSE', 'Mechanical', 'Civil', 'Chemical', 'Meta', 'Electrical', 'ECE', 'AI']
STATUS_OPTIONS = ['4th year student', 'Alumni (Graduated)']
PROJECT_COUNTS = ['0', '1 to 2', '3 to 4', '5+']
PROJECT_DOMAINS= ['Software Development', 'Data Science / AI', 'Core Engineering',
                  'Robotics / Embedded Systems', 'Mixed domains']
INTERNSHIPS    = ['No internship', 'Software Development Internship',
                  'Data Science / AI Internship', 'Core Engineering Internship',
                  'Electronics / Embedded Internship']
PREP_DOMAINS   = ['Software Development', 'Data Science / AI', 'Core Engineering',
                  'Embedded Systems / Electronics', 'Consulting / Management']

ROLE_ICONS = {
    'Software Developer':         '💻',
    'Data Analyst':               '📊',
    'Data Scientist':             '🔬',
    'Machine Learning Engineer':  '🤖',
    'DevOps Engineer':            '⚙️',
    'Embedded Systems Engineer':  '🔌',
    'Mechanical Design Engineer': '🏗️',
    'Manufacturing Engineer':     '🏭',
    'Civil Engineer':             '🌉',
}

# Ideal skill profile for each role (Python, CPP, Java, MATLAB, DSA, SQL, OOP, OS) — scale 1–5
IDEAL_PROFILES = {
    'Software Developer':         {'Python':5,'C/C++':4,'Java':4,'MATLAB':1,'DSA':5,'SQL':3,'OOP':5,'OS':4},
    'Data Analyst':               {'Python':4,'C/C++':1,'Java':1,'MATLAB':2,'DSA':2,'SQL':5,'OOP':2,'OS':2},
    'Data Scientist':             {'Python':5,'C/C++':2,'Java':1,'MATLAB':3,'DSA':3,'SQL':4,'OOP':3,'OS':2},
    'Machine Learning Engineer':  {'Python':5,'C/C++':3,'Java':2,'MATLAB':3,'DSA':4,'SQL':3,'OOP':4,'OS':3},
    'DevOps Engineer':            {'Python':4,'C/C++':2,'Java':2,'MATLAB':1,'DSA':3,'SQL':3,'OOP':3,'OS':5},
    'Embedded Systems Engineer':  {'Python':2,'C/C++':5,'Java':1,'MATLAB':4,'DSA':2,'SQL':1,'OOP':3,'OS':5},
    'Mechanical Design Engineer': {'Python':1,'C/C++':1,'Java':1,'MATLAB':4,'DSA':1,'SQL':1,'OOP':1,'OS':1},
    'Manufacturing Engineer':     {'Python':1,'C/C++':1,'Java':1,'MATLAB':4,'DSA':1,'SQL':1,'OOP':1,'OS':1},
    'Civil Engineer':             {'Python':1,'C/C++':1,'Java':1,'MATLAB':3,'DSA':1,'SQL':1,'OOP':1,'OS':1},
}

IDEAL_TOOLS = {
    'Software Developer':         ['Data Analysis (Python / Excel / Pandas)'],
    'Data Analyst':               ['Data Analysis (Python / Excel / Pandas)','Data Visualization (PowerBI / Tableau)'],
    'Data Scientist':             ['Machine Learning / Deep Learning','Data Analysis (Python / Excel / Pandas)','Data Visualization (PowerBI / Tableau)'],
    'Machine Learning Engineer':  ['Machine Learning / Deep Learning','Data Analysis (Python / Excel / Pandas)'],
    'DevOps Engineer':            ['Data Analysis (Python / Excel / Pandas)'],
    'Embedded Systems Engineer':  ['Embedded systems / microcontrollers','Circuit design tools','MATLAB / Simulink'],
    'Mechanical Design Engineer': ['CAD tools (SolidWorks / CATIA / AutoCAD)','ANSYS / Simulation tools','MATLAB / Simulink'],
    'Manufacturing Engineer':     ['CAD tools (SolidWorks / CATIA / AutoCAD)','ANSYS / Simulation tools','MATLAB / Simulink'],
    'Civil Engineer':             ['CAD tools (SolidWorks / CATIA / AutoCAD)','ANSYS / Simulation tools'],
}

# ─────────────────────────────────────────────────────────────────────────────
# ROADMAPS (per role)
# ─────────────────────────────────────────────────────────────────────────────
ROADMAPS = {
    'Software Developer': [
        {"phase":"Phase 1 · Weeks 1-4",  "title":"Strengthen DSA & OOP",         "desc":"Solve 50+ problems on LeetCode (arrays, trees, graphs, DP). Revise SOLID principles and design patterns. Complete a Java/Python OOP mini-project."},
        {"phase":"Phase 2 · Weeks 5-10", "title":"Build Full-Stack Projects",     "desc":"Build 2 end-to-end projects using REST APIs, databases (PostgreSQL/MySQL). Use Git/GitHub for version control on every project."},
        {"phase":"Phase 3 · Weeks 11-14","title":"System Design Basics",          "desc":"Learn about load balancers, caching, microservices. Study HLD/LLD concepts. Practice explaining architecture decisions."},
        {"phase":"Phase 4 · Weeks 15-16","title":"Interview & Resume Prep",       "desc":"Practice 2-3 mock interviews weekly. Polish GitHub README files. Add quantified metrics to resume (e.g., 'Reduced API latency by 30%')."},
    ],
    'Data Analyst': [
        {"phase":"Phase 1 · Weeks 1-4",  "title":"Master SQL & Excel",            "desc":"Complete advanced SQL (window functions, CTEs, subqueries). Learn Pivot Tables, VLOOKUP, and Power Query in Excel."},
        {"phase":"Phase 2 · Weeks 5-8",  "title":"Python for Data Analysis",      "desc":"Get proficient in Pandas, NumPy, Matplotlib, Seaborn. Analyze 3 real datasets from Kaggle (EDA + storytelling)."},
        {"phase":"Phase 3 · Weeks 9-12", "title":"Business Intelligence Tools",   "desc":"Build 2-3 interactive dashboards in Power BI or Tableau. Connect to a live database and create drill-down reports."},
        {"phase":"Phase 4 · Weeks 13-16","title":"Domain Knowledge + Portfolio",  "desc":"Pick 1 domain (finance/healthcare/e-commerce). Build an end-to-end analytics project with insights and presentation."},
    ],
    'Data Scientist': [
        {"phase":"Phase 1 · Weeks 1-4",  "title":"Maths & Stats Foundation",      "desc":"Revise Linear Algebra, Probability, Statistics (hypothesis testing, distributions). Khan Academy + 3Blue1Brown are excellent."},
        {"phase":"Phase 2 · Weeks 5-10", "title":"Core ML Algorithms",            "desc":"Implement Linear/Logistic Regression, Decision Trees, Random Forest, SVM, KNN from scratch. Understand bias-variance tradeoff."},
        {"phase":"Phase 3 · Weeks 11-14","title":"End-to-End ML Projects",        "desc":"Complete 2 Kaggle competitions. Build a full pipeline: data ingestion → EDA → feature engineering → model → evaluation → deployment."},
        {"phase":"Phase 4 · Weeks 15-16","title":"Advanced Topics + Portfolio",   "desc":"Learn basics of deep learning (Neural Networks, CNNs). Deploy one model as an API (Flask/FastAPI). Write a Medium article about your project."},
    ],
    'Machine Learning Engineer': [
        {"phase":"Phase 1 · Weeks 1-4",  "title":"Python & ML Fundamentals",     "desc":"Master Scikit-learn, Pandas, NumPy. Understand model training pipelines, cross-validation, hyperparameter tuning thoroughly."},
        {"phase":"Phase 2 · Weeks 5-10", "title":"Deep Learning & Frameworks",   "desc":"Learn TensorFlow or PyTorch. Build CNNs, RNNs, Transformers. Complete fast.ai or deeplearning.ai specialization."},
        {"phase":"Phase 3 · Weeks 11-14","title":"MLOps & Model Deployment",     "desc":"Learn Docker, MLflow, CI/CD pipelines. Deploy models using FastAPI + Docker. Understand model monitoring and drift detection."},
        {"phase":"Phase 4 · Weeks 15-16","title":"Research + Open Source",       "desc":"Read 2-3 ML papers (Arxiv). Contribute to an open-source ML project on GitHub. Build a personal ML portfolio website."},
    ],
    'DevOps Engineer': [
        {"phase":"Phase 1 · Weeks 1-4",  "title":"Linux & Scripting",            "desc":"Get comfortable with Linux commands, shell scripting (Bash). Understand file systems, processes, networking basics."},
        {"phase":"Phase 2 · Weeks 5-9",  "title":"Docker & Kubernetes",          "desc":"Containerize an app with Docker. Deploy it on Kubernetes cluster. Learn Helm charts and basic cluster management."},
        {"phase":"Phase 3 · Weeks 10-13","title":"CI/CD & Cloud",               "desc":"Set up a Jenkins or GitHub Actions pipeline. Get AWS/GCP/Azure fundamentals (free tier). Earn one cloud certification."},
        {"phase":"Phase 4 · Weeks 14-16","title":"Monitoring & Portfolio",       "desc":"Set up Prometheus + Grafana monitoring. Document 2-3 DevOps projects on GitHub with detailed README files."},
    ],
    'Embedded Systems Engineer': [
        {"phase":"Phase 1 · Weeks 1-4",  "title":"C/C++ & Microcontrollers",     "desc":"Master pointers, memory management, bitwise operations in C. Program Arduino/STM32/ESP32. Work with GPIO, interrupts, timers."},
        {"phase":"Phase 2 · Weeks 5-9",  "title":"Communication Protocols",      "desc":"Implement UART, SPI, I2C protocols. Interface sensors (IMU, temperature, ultrasonic). Build a hardware + software mini-project."},
        {"phase":"Phase 3 · Weeks 10-13","title":"RTOS & Circuit Design",       "desc":"Learn FreeRTOS basics (tasks, queues, semaphores). Design simple PCBs using KiCad or EasyEDA. Understand signal conditioning."},
        {"phase":"Phase 4 · Weeks 14-16","title":"Portfolio & Certification",    "desc":"Build a complete embedded project (IoT device or robot). Document on GitHub + Hackaday. Prepare for company-specific MCU tests."},
    ],
    'Mechanical Design Engineer': [
        {"phase":"Phase 1 · Weeks 1-4",  "title":"CAD Proficiency",              "desc":"Master SolidWorks or CATIA — assemblies, drawings, GD&T. Complete at least 10 complex part models. Get SolidWorks CSWA certification."},
        {"phase":"Phase 2 · Weeks 5-9",  "title":"FEA & Simulation",             "desc":"Learn ANSYS Mechanical for static, thermal, and fatigue analysis. Validate designs against material limits."},
        {"phase":"Phase 3 · Weeks 10-13","title":"Manufacturing Knowledge",      "desc":"Study manufacturing processes (casting, forging, CNC). Understand DFM (Design for Manufacturability) principles."},
        {"phase":"Phase 4 · Weeks 14-16","title":"Design Portfolio",             "desc":"Build 3-5 design projects (automotive/consumer product/mechanism). Document with engineering drawings + simulation results."},
    ],
    'Manufacturing Engineer': [
        {"phase":"Phase 1 · Weeks 1-4",  "title":"Manufacturing Processes",      "desc":"Study casting, forging, machining, welding, additive manufacturing. Understand process selection criteria and tolerance analysis."},
        {"phase":"Phase 2 · Weeks 5-8",  "title":"Quality & Lean Manufacturing", "desc":"Learn Six Sigma DMAIC, 5S, Kaizen, SPC. Understand ISO 9001 basics. Practice with quality control case studies."},
        {"phase":"Phase 3 · Weeks 9-12", "title":"CAD & Simulation",             "desc":"Become proficient in SolidWorks for manufacturing drawings. Learn basic ANSYS for stress analysis of manufactured parts."},
        {"phase":"Phase 4 · Weeks 13-16","title":"Industry Certifications",      "desc":"Pursue Six Sigma Yellow/Green Belt. Build a process optimization project. Network on LinkedIn with manufacturing professionals."},
    ],
    'Civil Engineer': [
        {"phase":"Phase 1 · Weeks 1-4",  "title":"Core Structural Concepts",     "desc":"Revise structural mechanics, fluid mechanics, soil mechanics. Study IS code provisions for RCC and steel design."},
        {"phase":"Phase 2 · Weeks 5-9",  "title":"Software Tools",               "desc":"Get proficient in AutoCAD 2D/3D for site and structural drawings. Learn STAAD.Pro or ETABS for structural analysis."},
        {"phase":"Phase 3 · Weeks 10-13","title":"Estimation & Project Mgmt",   "desc":"Learn BOQ preparation, rate analysis, MS Project basics. Study contract management and construction safety norms."},
        {"phase":"Phase 4 · Weeks 14-16","title":"Certifications & Portfolio",   "desc":"Pursue GATE preparation if higher studies desired. Build a portfolio with 2-3 structural design + drawings projects."},
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# COURSE RECOMMENDATIONS (per role)
# ─────────────────────────────────────────────────────────────────────────────
COURSES = {
    'Software Developer': [
        {"platform":"LeetCode",    "title":"LeetCode 75 Study Plan",                  "desc":"Top 75 DSA problems covering all major patterns. Free & structured.",    "url":"https://leetcode.com/studyplan/leetcode-75/"},
        {"platform":"Coursera",    "title":"Meta Back-End Developer Certificate",      "desc":"Django, APIs, Databases, Python — by Meta engineers.",                   "url":"https://www.coursera.org/professional-certificates/meta-back-end-developer"},
        {"platform":"YouTube",     "title":"Traversy Media – Full Stack Crash Courses","desc":"Free in-depth tutorials on JavaScript, React, Node.js.",                 "url":"https://www.youtube.com/@TraversyMedia"},
        {"platform":"Udemy",       "title":"The Complete 2024 Web Dev Bootcamp",       "desc":"HTML → CSS → JS → React → Node → MongoDB. By Dr. Angela Yu.",           "url":"https://www.udemy.com/course/the-complete-web-development-bootcamp/"},
    ],
    'Data Analyst': [
        {"platform":"Coursera",    "title":"Google Data Analytics Certificate",        "desc":"8-course program covering SQL, R, Tableau, data storytelling.",          "url":"https://www.coursera.org/professional-certificates/google-data-analytics"},
        {"platform":"YouTube",     "title":"Alex The Analyst – SQL & Power BI",        "desc":"Best free channel for SQL queries, Power BI dashboards step by step.",    "url":"https://www.youtube.com/@AlexTheAnalyst"},
        {"platform":"Kaggle",      "title":"Pandas & SQL Courses (Free)",              "desc":"Hands-on micro-courses by Kaggle. Perfect for quick skill building.",     "url":"https://www.kaggle.com/learn"},
        {"platform":"Udemy",       "title":"Microsoft Power BI – Up & Running",        "desc":"Build real-world dashboards with Power BI Desktop. By Maven Analytics.",  "url":"https://www.udemy.com/course/microsoft-power-bi-up-running-with-power-bi-desktop/"},
    ],
    'Data Scientist': [
        {"platform":"Coursera",    "title":"IBM Data Science Professional Certificate","desc":"10-course series: Python, SQL, ML, visualization, capstone project.",     "url":"https://www.coursera.org/professional-certificates/ibm-data-science"},
        {"platform":"fast.ai",     "title":"Practical Deep Learning for Coders",       "desc":"Top-down approach to deep learning. Free. Taught by Jeremy Howard.",      "url":"https://course.fast.ai/"},
        {"platform":"Kaggle",      "title":"Intro to ML + Feature Engineering",        "desc":"Free hands-on learning tracks directly applicable to competitions.",      "url":"https://www.kaggle.com/learn/intro-to-machine-learning"},
        {"platform":"YouTube",     "title":"StatQuest with Josh Starmer",             "desc":"Best channel to understand statistics and ML algorithms visually.",       "url":"https://www.youtube.com/@statquest"},
    ],
    'Machine Learning Engineer': [
        {"platform":"deeplearning.ai","title":"Machine Learning Specialization",       "desc":"Andrew Ng's updated 3-course ML specialization. Industry standard.",      "url":"https://www.coursera.org/specializations/machine-learning-introduction"},
        {"platform":"deeplearning.ai","title":"MLOps Specialization",                  "desc":"4 courses on production ML systems, pipelines, monitoring.",              "url":"https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops"},
        {"platform":"YouTube",     "title":"Andrej Karpathy – Neural Networks: Zero to Hero","desc":"Build GPT from scratch. Best for understanding LLM internals.",     "url":"https://www.youtube.com/@AndrejKarpathy"},
        {"platform":"Udemy",       "title":"PyTorch for Deep Learning Bootcamp",       "desc":"Hands-on PyTorch: CNNs, RNNs, GANs, Transfer Learning.",                 "url":"https://www.udemy.com/course/pytorch-for-deep-learning/"},
    ],
    'DevOps Engineer': [
        {"platform":"Coursera",    "title":"Google Cloud DevOps Engineer Certificate", "desc":"SRE practices, CI/CD, monitoring, incident management by Google.",        "url":"https://www.coursera.org/professional-certificates/sre-devops-engineer-google-cloud"},
        {"platform":"Udemy",       "title":"Docker & Kubernetes: The Complete Guide",  "desc":"By Stephen Grider. Most comprehensive Docker + K8s course available.",    "url":"https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/"},
        {"platform":"YouTube",     "title":"TechWorld with Nana – DevOps Tutorials",  "desc":"Free end-to-end tutorials on Jenkins, Terraform, Kubernetes, AWS.",       "url":"https://www.youtube.com/@TechWorldwithNana"},
        {"platform":"AWS",         "title":"AWS Cloud Practitioner (Free Prep)",       "desc":"Free exam prep materials on AWS Skill Builder platform.",                 "url":"https://explore.skillbuilder.aws/learn"},
    ],
    'Embedded Systems Engineer': [
        {"platform":"Coursera",    "title":"Embedded Systems — Shape the World (UTAustin)","desc":"Hands-on ARM Cortex-M4 programming. Free to audit.",                 "url":"https://www.edx.org/learn/embedded-systems/the-university-of-texas-at-austin-embedded-systems-shape-the-world-microcontroller-inputoutput"},
        {"platform":"YouTube",     "title":"Dronebot Workshop – Arduino & Electronics","desc":"Practical tutorials on microcontrollers, sensors, motor control.",       "url":"https://www.youtube.com/@DronebotWorkshop"},
        {"platform":"Udemy",       "title":"Mastering Microcontroller with Embedded C", "desc":"Bit manipulation, GPIO, timers, UART, SPI, I2C from scratch.",           "url":"https://www.udemy.com/course/mastering-microcontroller-with-peripheral-driver-development/"},
        {"platform":"Coursera",    "title":"Introduction to FPGA Design",              "desc":"Verilog/VHDL for FPGA programming. By University of Colorado.",          "url":"https://www.coursera.org/learn/fpga-hardware-description-languages"},
    ],
    'Mechanical Design Engineer': [
        {"platform":"Coursera",    "title":"Engineering Design for a Circular Economy","desc":"Systems thinking + sustainable design principles.",                       "url":"https://www.coursera.org/learn/engineering-design-for-a-circular-economy"},
        {"platform":"YouTube",     "title":"SOLIDWORKS Tutorials by CADImagineer",    "desc":"Free structured SolidWorks course from beginner to advanced.",            "url":"https://www.youtube.com/@CADImagineer"},
        {"platform":"Udemy",       "title":"ANSYS Mechanical FEA Simulation",         "desc":"Linear statics, thermal, modal, fatigue simulation from scratch.",        "url":"https://www.udemy.com/course/ansys-mechanical-fea-simulation/"},
        {"platform":"Coursera",    "title":"CAD and Digital Manufacturing Specialization","desc":"Autodesk tools, digital manufacturing workflows — by Autodesk.",       "url":"https://www.coursera.org/specializations/cad-digital-manufacturing"},
    ],
    'Manufacturing Engineer': [
        {"platform":"Coursera",    "title":"Lean Six Sigma Yellow Belt",               "desc":"Process improvement methodology. Widely respected industry credential.",  "url":"https://www.coursera.org/learn/six-sigma-define-measure-advanced"},
        {"platform":"YouTube",     "title":"The Manufacturing Guy – Process Videos",   "desc":"Practical videos on machining, CNC, casting, quality systems.",          "url":"https://www.youtube.com/@themanufacturingguy"},
        {"platform":"Udemy",       "title":"SolidWorks for Manufacturing Engineers",   "desc":"Technical drawings, GD&T, BOMs for production-ready designs.",           "url":"https://www.udemy.com/course/solidworks-for-beginners-and-job-seekers/"},
        {"platform":"edX",         "title":"Supply Chain Management MicroMasters",     "desc":"MIT's supply chain program — production planning, logistics.",            "url":"https://www.edx.org/masters/micromasters/mitx-supply-chain-management"},
    ],
    'Civil Engineer': [
        {"platform":"YouTube",     "title":"Structville – Structural Design Tutorials","desc":"RCC and steel design as per IS/BS codes. Free and very practical.",      "url":"https://www.youtube.com/@Structville"},
        {"platform":"Udemy",       "title":"AutoCAD 2024 – From Zero to Hero",        "desc":"2D + 3D drafting for civil engineering drawings. Beginner friendly.",     "url":"https://www.udemy.com/course/autocad-2019-from-zero-to-hero/"},
        {"platform":"Coursera",    "title":"Construction Project Management",          "desc":"By Columbia University. Covers planning, scheduling, risk, cost.",        "url":"https://www.coursera.org/learn/construction-project-management"},
        {"platform":"YouTube",     "title":"NPTEL – Civil Engineering Lectures (IIT)","desc":"Free IIT lecture series covering all core civil engineering topics.",     "url":"https://www.youtube.com/c/iit"},
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION (runs once and is cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training the ML model... please wait ⏳")
def train_model():
    np.random.seed(42)
    NUM_SAMPLES = 2500

    def pick(dist):
        keys, probs = list(dist.keys()), np.array(list(dist.values()), dtype=float)
        probs /= probs.sum()
        return np.random.choice(keys, p=probs)

    def rng(lo, hi):
        val = int(round(np.random.normal((lo+hi)/2, 0.6)))
        return max(1, min(5, val))

    def gen_tools(likely, possible):
        sel = [t for t in likely if np.random.random() < 0.90]
        sel += [t for t in possible if np.random.random() < 0.25]
        extra = [t for t in TOOLS_LIST if t not in sel]
        if extra and np.random.random() < 0.08:
            sel.append(np.random.choice(extra))
        return sel if sel else [np.random.choice(likely if likely else TOOLS_LIST)]

    ARCH = {
        'Software Developer':        {'branch':{'CSE':.55,'AI':.15,'ECE':.10,'Electrical':.06,'Mechanical':.04,'Chemical':.02,'Civil':.02,'Meta':.06},'cgpa':(7.0,9.8),'py':(3,5),'cpp':(3,5),'java':(3,5),'mat':(1,2),'dsa':(4,5),'sql':(3,5),'oop':(4,5),'os':(3,5),'tools_l':['Data Analysis (Python / Excel / Pandas)'],'tools_p':['Machine Learning / Deep Learning'],'proj_ct':{'0':.01,'1 to 2':.10,'3 to 4':.50,'5+':.39},'proj_dm':{'Software Development':.75,'Mixed domains':.15,'Data Science / AI':.08,'Core Engineering':.01,'Robotics / Embedded Systems':.01},'intern':{'Software Development Internship':.60,'No internship':.20,'Data Science / AI Internship':.12,'Core Engineering Internship':.04,'Electronics / Embedded Internship':.04},'prep':{'Software Development':.75,'Data Science / AI':.12,'Consulting / Management':.08,'Core Engineering':.03,'Embedded Systems / Electronics':.02},'conf':(3,5),'w':.20},
        'Data Analyst':              {'branch':{'CSE':.30,'AI':.25,'ECE':.10,'Electrical':.08,'Mechanical':.08,'Chemical':.07,'Civil':.06,'Meta':.06},'cgpa':(6.5,9.2),'py':(3,5),'cpp':(1,3),'java':(1,3),'mat':(1,3),'dsa':(2,4),'sql':(4,5),'oop':(2,4),'os':(1,3),'tools_l':['Data Analysis (Python / Excel / Pandas)','Data Visualization (PowerBI / Tableau)'],'tools_p':['Machine Learning / Deep Learning'],'proj_ct':{'0':.03,'1 to 2':.25,'3 to 4':.50,'5+':.22},'proj_dm':{'Data Science / AI':.65,'Mixed domains':.20,'Software Development':.10,'Core Engineering':.03,'Robotics / Embedded Systems':.02},'intern':{'Data Science / AI Internship':.50,'No internship':.22,'Software Development Internship':.18,'Core Engineering Internship':.06,'Electronics / Embedded Internship':.04},'prep':{'Data Science / AI':.70,'Software Development':.12,'Consulting / Management':.10,'Core Engineering':.04,'Embedded Systems / Electronics':.04},'conf':(2,5),'w':.12},
        'Data Scientist':            {'branch':{'AI':.40,'CSE':.30,'ECE':.08,'Electrical':.06,'Mechanical':.05,'Chemical':.04,'Civil':.02,'Meta':.05},'cgpa':(7.5,9.9),'py':(4,5),'cpp':(2,4),'java':(1,3),'mat':(2,4),'dsa':(3,5),'sql':(4,5),'oop':(3,5),'os':(2,4),'tools_l':['Machine Learning / Deep Learning','Data Analysis (Python / Excel / Pandas)','Data Visualization (PowerBI / Tableau)'],'tools_p':['MATLAB / Simulink'],'proj_ct':{'0':.01,'1 to 2':.08,'3 to 4':.40,'5+':.51},'proj_dm':{'Data Science / AI':.80,'Mixed domains':.10,'Software Development':.07,'Core Engineering':.02,'Robotics / Embedded Systems':.01},'intern':{'Data Science / AI Internship':.65,'Software Development Internship':.12,'No internship':.13,'Core Engineering Internship':.06,'Electronics / Embedded Internship':.04},'prep':{'Data Science / AI':.80,'Software Development':.08,'Consulting / Management':.05,'Core Engineering':.04,'Embedded Systems / Electronics':.03},'conf':(3,5),'w':.11},
        'Machine Learning Engineer': {'branch':{'AI':.45,'CSE':.30,'ECE':.08,'Electrical':.05,'Mechanical':.03,'Chemical':.03,'Civil':.02,'Meta':.04},'cgpa':(7.8,10.0),'py':(5,5),'cpp':(3,5),'java':(2,4),'mat':(2,4),'dsa':(4,5),'sql':(3,5),'oop':(4,5),'os':(3,5),'tools_l':['Machine Learning / Deep Learning','Data Analysis (Python / Excel / Pandas)'],'tools_p':['Data Visualization (PowerBI / Tableau)','MATLAB / Simulink'],'proj_ct':{'0':.01,'1 to 2':.04,'3 to 4':.30,'5+':.65},'proj_dm':{'Data Science / AI':.75,'Software Development':.12,'Mixed domains':.10,'Core Engineering':.02,'Robotics / Embedded Systems':.01},'intern':{'Data Science / AI Internship':.55,'Software Development Internship':.22,'No internship':.10,'Core Engineering Internship':.08,'Electronics / Embedded Internship':.05},'prep':{'Data Science / AI':.75,'Software Development':.12,'Core Engineering':.05,'Embedded Systems / Electronics':.04,'Consulting / Management':.04},'conf':(4,5),'w':.09},
        'DevOps Engineer':           {'branch':{'CSE':.50,'ECE':.15,'AI':.08,'Electrical':.10,'Mechanical':.04,'Chemical':.03,'Civil':.02,'Meta':.08},'cgpa':(6.5,9.0),'py':(3,5),'cpp':(2,4),'java':(2,4),'mat':(1,2),'dsa':(2,4),'sql':(3,5),'oop':(3,5),'os':(4,5),'tools_l':['Data Analysis (Python / Excel / Pandas)'],'tools_p':[],'proj_ct':{'0':.04,'1 to 2':.18,'3 to 4':.48,'5+':.30},'proj_dm':{'Software Development':.60,'Mixed domains':.25,'Data Science / AI':.08,'Core Engineering':.04,'Robotics / Embedded Systems':.03},'intern':{'Software Development Internship':.55,'No internship':.25,'Data Science / AI Internship':.08,'Core Engineering Internship':.06,'Electronics / Embedded Internship':.06},'prep':{'Software Development':.65,'Core Engineering':.10,'Consulting / Management':.10,'Data Science / AI':.08,'Embedded Systems / Electronics':.07},'conf':(3,5),'w':.08},
        'Embedded Systems Engineer': {'branch':{'ECE':.40,'Electrical':.28,'CSE':.06,'AI':.03,'Mechanical':.06,'Chemical':.02,'Civil':.02,'Meta':.13},'cgpa':(6.0,9.3),'py':(2,3),'cpp':(4,5),'java':(1,2),'mat':(3,5),'dsa':(2,4),'sql':(1,2),'oop':(2,4),'os':(4,5),'tools_l':['Embedded systems / microcontrollers','Circuit design tools','MATLAB / Simulink'],'tools_p':['ANSYS / Simulation tools'],'proj_ct':{'0':.03,'1 to 2':.18,'3 to 4':.48,'5+':.31},'proj_dm':{'Robotics / Embedded Systems':.72,'Core Engineering':.12,'Mixed domains':.10,'Software Development':.04,'Data Science / AI':.02},'intern':{'Electronics / Embedded Internship':.60,'Core Engineering Internship':.15,'No internship':.15,'Software Development Internship':.06,'Data Science / AI Internship':.04},'prep':{'Embedded Systems / Electronics':.75,'Core Engineering':.10,'Software Development':.08,'Data Science / AI':.04,'Consulting / Management':.03},'conf':(3,5),'w':.10},
        'Mechanical Design Engineer':{'branch':{'Mechanical':.65,'Meta':.12,'Civil':.04,'Chemical':.04,'CSE':.02,'AI':.02,'ECE':.03,'Electrical':.08},'cgpa':(6.0,9.3),'py':(1,2),'cpp':(1,3),'java':(1,2),'mat':(3,5),'dsa':(1,2),'sql':(1,2),'oop':(1,3),'os':(1,2),'tools_l':['CAD tools (SolidWorks / CATIA / AutoCAD)','ANSYS / Simulation tools','MATLAB / Simulink'],'tools_p':['Data Analysis (Python / Excel / Pandas)'],'proj_ct':{'0':.04,'1 to 2':.22,'3 to 4':.48,'5+':.26},'proj_dm':{'Core Engineering':.75,'Mixed domains':.12,'Robotics / Embedded Systems':.08,'Software Development':.03,'Data Science / AI':.02},'intern':{'Core Engineering Internship':.65,'No internship':.20,'Electronics / Embedded Internship':.07,'Software Development Internship':.05,'Data Science / AI Internship':.03},'prep':{'Core Engineering':.78,'Consulting / Management':.08,'Embedded Systems / Electronics':.07,'Software Development':.04,'Data Science / AI':.03},'conf':(2,5),'w':.09},
        'Manufacturing Engineer':    {'branch':{'Mechanical':.45,'Chemical':.22,'Meta':.15,'Civil':.04,'Electrical':.06,'ECE':.03,'CSE':.02,'AI':.03},'cgpa':(5.5,8.8),'py':(1,2),'cpp':(1,2),'java':(1,2),'mat':(3,5),'dsa':(1,2),'sql':(1,2),'oop':(1,2),'os':(1,2),'tools_l':['CAD tools (SolidWorks / CATIA / AutoCAD)','ANSYS / Simulation tools','MATLAB / Simulink'],'tools_p':[],'proj_ct':{'0':.06,'1 to 2':.32,'3 to 4':.42,'5+':.20},'proj_dm':{'Core Engineering':.70,'Mixed domains':.15,'Robotics / Embedded Systems':.08,'Software Development':.04,'Data Science / AI':.03},'intern':{'Core Engineering Internship':.58,'No internship':.25,'Electronics / Embedded Internship':.07,'Software Development Internship':.06,'Data Science / AI Internship':.04},'prep':{'Core Engineering':.72,'Consulting / Management':.12,'Embedded Systems / Electronics':.08,'Software Development':.04,'Data Science / AI':.04},'conf':(2,4),'w':.08},
        'Civil Engineer':            {'branch':{'Civil':.80,'Mechanical':.04,'Chemical':.03,'Meta':.03,'CSE':.02,'AI':.01,'ECE':.02,'Electrical':.05},'cgpa':(5.5,9.2),'py':(1,2),'cpp':(1,2),'java':(1,2),'mat':(2,4),'dsa':(1,2),'sql':(1,2),'oop':(1,2),'os':(1,2),'tools_l':['CAD tools (SolidWorks / CATIA / AutoCAD)','ANSYS / Simulation tools'],'tools_p':['MATLAB / Simulink'],'proj_ct':{'0':.06,'1 to 2':.30,'3 to 4':.44,'5+':.20},'proj_dm':{'Core Engineering':.82,'Mixed domains':.10,'Software Development':.03,'Data Science / AI':.03,'Robotics / Embedded Systems':.02},'intern':{'Core Engineering Internship':.65,'No internship':.25,'Software Development Internship':.04,'Data Science / AI Internship':.03,'Electronics / Embedded Internship':.03},'prep':{'Core Engineering':.80,'Consulting / Management':.10,'Software Development':.04,'Data Science / AI':.03,'Embedded Systems / Electronics':.03},'conf':(2,4),'w':.08},
    }

    records = []
    for role, a in ARCH.items():
        n = int(NUM_SAMPLES * a['w']) + (5 if role=='Software Developer' else 0)
        for _ in range(n):
            cgpa = round(np.clip(np.random.normal(sum(a['cgpa'])/2,(a['cgpa'][1]-a['cgpa'][0])/4),4.0,10.0),1)
            records.append({'Engineering_Branch':pick(a['branch']),'Student_Status':np.random.choice(STATUS_OPTIONS,p=[0.55,0.45]),'CGPA':cgpa,'Python_Proficiency':rng(*a['py']),'CPP_Proficiency':rng(*a['cpp']),'Java_Proficiency':rng(*a['java']),'MATLAB_Proficiency':rng(*a['mat']),'DSA_Understanding':rng(*a['dsa']),'SQL_Understanding':rng(*a['sql']),'OOP_Understanding':rng(*a['oop']),'OS_Understanding':rng(*a['os']),'Technical_Tools':';'.join(sorted(gen_tools(a['tools_l'],a['tools_p']))),'Project_Count':pick(a['proj_ct']),'Project_Domain':pick(a['proj_dm']),'Internship_Experience':pick(a['intern']),'Preparation_Domain':pick(a['prep']),'Confidence_Level':rng(*a['conf']),'Job_Role':role})

    df = pd.DataFrame(records).sample(frac=1,random_state=42).reset_index(drop=True)

    # Preprocessing
    for tool in TOOLS_LIST:
        col = 'Tool_'+tool.split('(')[0].strip().replace(' ','_').replace('/','_')
        df[col] = df['Technical_Tools'].apply(lambda x: 1 if tool in str(x) else 0)
    df.drop('Technical_Tools',axis=1,inplace=True)

    df['Avg_Programming'] = (df['Python_Proficiency']+df['CPP_Proficiency']+df['Java_Proficiency']+df['MATLAB_Proficiency'])/4
    df['Avg_Concepts']    = (df['DSA_Understanding']+df['SQL_Understanding']+df['OOP_Understanding']+df['OS_Understanding'])/4
    df['Total_Tools']     = sum(df[c] for c in df.columns if c.startswith('Tool_'))
    df['SW_Index']        = (df['Python_Proficiency']+df['CPP_Proficiency']+df['Java_Proficiency']+df['DSA_Understanding']+df['OOP_Understanding'])/5
    df['DS_Index']        = (df['Python_Proficiency']+df['SQL_Understanding']+df['Tool_Machine_Learning___Deep_Learning']*3+df['Tool_Data_Analysis']*3+df['Tool_Data_Visualization']*2)/5
    df['Core_Index']      = (df['MATLAB_Proficiency']+df['Tool_CAD_tools']*3+df['Tool_ANSYS___Simulation_tools']*3+df['Tool_MATLAB___Simulink']*2)/4
    df['Embed_Index']     = (df['CPP_Proficiency']+df['OS_Understanding']+df['MATLAB_Proficiency']+df['Tool_Embedded_systems___microcontrollers']*3+df['Tool_Circuit_design_tools']*3)/5

    le_dict = {}
    for col in ['Engineering_Branch','Student_Status','Project_Count','Project_Domain','Internship_Experience','Preparation_Domain']:
        le = LabelEncoder(); df[col]=le.fit_transform(df[col]); le_dict[col]=le

    le_target = LabelEncoder()
    df['Job_Role_Encoded'] = le_target.fit_transform(df['Job_Role'])

    feat_cols = [c for c in df.columns if c not in ['Job_Role','Job_Role_Encoded']]
    X = df[feat_cols].values; y = df['Job_Role_Encoded'].values
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    rf = RandomForestClassifier(n_estimators=300,max_depth=25,random_state=42,n_jobs=-1)
    rf.fit(X_tr,y_tr)
    acc = rf.score(X_te,y_te)

    return rf, le_dict, le_target, feat_cols, acc

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def build_features(inp, le_dict, feat_cols):
    tool_flags = {}
    for tool in TOOLS_LIST:
        col = 'Tool_'+tool.split('(')[0].strip().replace(' ','_').replace('/','_')
        tool_flags[col] = 1 if tool in inp['tools'] else 0

    py,cpp,java,mat = inp['python'],inp['cpp'],inp['java'],inp['matlab']
    dsa,sql,oop,os_ = inp['dsa'],inp['sql'],inp['oop'],inp['os']

    avg_prog  = (py+cpp+java+mat)/4
    avg_conc  = (dsa+sql+oop+os_)/4
    total_t   = sum(tool_flags.values())
    sw_idx    = (py+cpp+java+dsa+oop)/5
    ds_idx    = (py+sql+tool_flags.get('Tool_Machine_Learning___Deep_Learning',0)*3+tool_flags.get('Tool_Data_Analysis',0)*3+tool_flags.get('Tool_Data_Visualization',0)*2)/5
    core_idx  = (mat+tool_flags.get('Tool_CAD_tools',0)*3+tool_flags.get('Tool_ANSYS___Simulation_tools',0)*3+tool_flags.get('Tool_MATLAB___Simulink',0)*2)/4
    embed_idx = (cpp+os_+mat+tool_flags.get('Tool_Embedded_systems___microcontrollers',0)*3+tool_flags.get('Tool_Circuit_design_tools',0)*3)/5

    row = {
        'Engineering_Branch':     le_dict['Engineering_Branch'].transform([inp['branch']])[0],
        'Student_Status':          le_dict['Student_Status'].transform([inp['status']])[0],
        'CGPA':                    inp['cgpa'],
        'Python_Proficiency':      py,'CPP_Proficiency':cpp,'Java_Proficiency':java,'MATLAB_Proficiency':mat,
        'DSA_Understanding':       dsa,'SQL_Understanding':sql,'OOP_Understanding':oop,'OS_Understanding':os_,
        'Project_Count':           le_dict['Project_Count'].transform([inp['proj_count']])[0],
        'Project_Domain':          le_dict['Project_Domain'].transform([inp['proj_domain']])[0],
        'Internship_Experience':   le_dict['Internship_Experience'].transform([inp['internship']])[0],
        'Preparation_Domain':      le_dict['Preparation_Domain'].transform([inp['prep_domain']])[0],
        'Confidence_Level':        inp['confidence'],
        **tool_flags,
        'Avg_Programming':avg_prog,'Avg_Concepts':avg_conc,'Total_Tools':total_t,
        'SW_Index':sw_idx,'DS_Index':ds_idx,'Core_Index':core_idx,'Embed_Index':embed_idx,
    }
    return np.array([row[f] for f in feat_cols]).reshape(1,-1)

def get_top3(model, feat_vec, le_target):
    probs = model.predict_proba(feat_vec)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    return [(le_target.inverse_transform([i])[0], round(probs[i]*100,1)) for i in top3_idx]

# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def radar_chart(user_vals, role):
    ideal = IDEAL_PROFILES[role]
    skills = list(ideal.keys())
    user   = [user_vals.get(s,1) for s in skills]
    ideal_v= [ideal[s] for s in skills]
    skills_wrap = skills + [skills[0]]
    user_wrap   = user   + [user[0]]
    ideal_wrap  = ideal_v + [ideal_v[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=ideal_wrap, theta=skills_wrap, fill='toself',
        name='Ideal for Role', line_color='#3b82f6', fillcolor='rgba(59,130,246,0.15)', line_width=2))
    fig.add_trace(go.Scatterpolar(r=user_wrap,  theta=skills_wrap, fill='toself',
        name='Your Profile',   line_color='#f59e0b', fillcolor='rgba(245,158,11,0.25)', line_width=2))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,5], tickfont=dict(size=10))),
        showlegend=True, height=340, margin=dict(l=40,r=40,t=30,b=30),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def confidence_bar_chart(top3):
    roles  = [r for r,_ in top3]
    confs  = [c for _,c in top3]
    colors = ['#f59e0b','#6b7280','#92400e']
    fig = go.Figure(go.Bar(x=confs, y=roles, orientation='h',
        marker_color=colors, text=[f"{c}%" for c in confs],
        textposition='auto', textfont=dict(color='white', size=13, family='Arial Black')))
    fig.update_layout(
        xaxis=dict(range=[0,100], title='Confidence %', ticksuffix='%'),
        yaxis=dict(autorange='reversed'),
        height=180, margin=dict(l=10,r=20,t=10,b=30),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def skill_gap_bars(user_vals, role):
    ideal = IDEAL_PROFILES[role]
    rows = []
    for skill, ideal_v in ideal.items():
        user_v = user_vals.get(skill, 1)
        gap = max(0, ideal_v - user_v)
        pct = int((user_v / 5) * 100)
        if gap == 0:    status = "✅ On track"
        elif gap <= 1:  status = "⚠️ Minor gap"
        else:           status = "❌ Needs work"
        rows.append({"skill":skill,"user":user_v,"ideal":ideal_v,"gap":gap,"pct":pct,"status":status})
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Train model
    model, le_dict, le_target, feat_cols, acc = train_model()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎓 Student Profile")
        st.markdown("---")

        st.markdown("### 📚 Academic Info")
        branch     = st.selectbox("Engineering Branch", BRANCHES)
        status     = st.selectbox("Your Status", STATUS_OPTIONS)
        cgpa       = st.slider("CGPA", 4.0, 10.0, 7.5, 0.1)

        st.markdown("---")
        st.markdown("### 💻 Programming Skills (1–5)")
        python   = st.slider("Python",  1, 5, 3)
        cpp      = st.slider("C/C++",   1, 5, 2)
        java     = st.slider("Java",    1, 5, 2)
        matlab   = st.slider("MATLAB",  1, 5, 1)

        st.markdown("---")
        st.markdown("### 🧠 Concept Understanding (1–5)")
        dsa      = st.slider("DSA",      1, 5, 3)
        sql      = st.slider("SQL",      1, 5, 3)
        oop      = st.slider("OOP",      1, 5, 3)
        os_skill = st.slider("OS",       1, 5, 2)

        st.markdown("---")
        st.markdown("### 🛠️ Technical Tools Used")
        tools_used = st.multiselect("Select all that apply", TOOLS_LIST)

        st.markdown("---")
        st.markdown("### 📁 Projects & Experience")
        proj_count  = st.selectbox("Projects Completed", PROJECT_COUNTS)
        proj_domain = st.selectbox("Project Domain",     PROJECT_DOMAINS)
        internship  = st.selectbox("Internship Experience", INTERNSHIPS)

        st.markdown("---")
        st.markdown("### 🎯 Placement Prep")
        prep_domain = st.selectbox("Domain Prepared For", PREP_DOMAINS)
        confidence  = st.slider("Confidence Level (1–5)", 1, 5, 3)

        st.markdown("---")
        predict_btn = st.button("🚀 Predict My Job Role", use_container_width=True, type="primary")

    # ── WELCOME BANNER ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="welcome-banner">
        <h1>🎓 Placement Readiness & Job Role Predictor</h1>
        <p>Fill in your profile on the left → Get your Top 3 job role predictions, skill gap analysis, and a personalized roadmap.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── METRICS ROW ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div style="font-size:1.6rem">🤖</div><div style="font-size:1.4rem;font-weight:800;color:#3b82f6">{acc*100:.1f}%</div><div style="font-size:0.75rem;color:#64748b">Model Accuracy</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-box"><div style="font-size:1.6rem">🏢</div><div style="font-size:1.4rem;font-weight:800;color:#3b82f6">9</div><div style="font-size:0.75rem;color:#64748b">Job Roles Covered</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-box"><div style="font-size:1.6rem">📊</div><div style="font-size:1.4rem;font-weight:800;color:#3b82f6">2,380</div><div style="font-size:0.75rem;color:#64748b">Training Samples</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-box"><div style="font-size:1.6rem">🎯</div><div style="font-size:1.4rem;font-weight:800;color:#3b82f6">Top 3</div><div style="font-size:0.75rem;color:#64748b">Predictions Given</div></div>', unsafe_allow_html=True)

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if not predict_btn:
        st.markdown("---")
        st.info("👈 Complete your profile in the sidebar and click **Predict My Job Role** to see your results.")
        return

    inp = {
        'branch':branch,'status':status,'cgpa':cgpa,
        'python':python,'cpp':cpp,'java':java,'matlab':matlab,
        'dsa':dsa,'sql':sql,'oop':oop,'os':os_skill,
        'tools':tools_used,
        'proj_count':proj_count,'proj_domain':proj_domain,
        'internship':internship,'prep_domain':prep_domain,'confidence':confidence
    }
    user_skill_map = {'Python':python,'C/C++':cpp,'Java':java,'MATLAB':matlab,
                      'DSA':dsa,'SQL':sql,'OOP':oop,'OS':os_skill}

    feat_vec = build_features(inp, le_dict, feat_cols)
    top3     = get_top3(model, feat_vec, le_target)

    st.markdown("---")
    # ── TOP 3 ROLE CARDS ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Your Top 3 Predicted Job Roles</div>', unsafe_allow_html=True)

    card_styles = [
        ("gold",  "🥇 Best Match"),
        ("silver","🥈 2nd Match"),
        ("bronze","🥉 3rd Match"),
    ]
    cols = st.columns(3)
    for i,(role,conf) in enumerate(top3):
        style, badge = card_styles[i]
        icon = ROLE_ICONS.get(role,'🎯')
        with cols[i]:
            st.markdown(f"""
            <div class="role-card {style}">
                <div style="font-size:2.5rem">{icon}</div>
                <div class="role-title">{role}</div>
                <div class="role-conf">{conf}%</div>
                <div class="role-badge">{badge}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(confidence_bar_chart(top3), use_container_width=True)

    # ── SKILL GAP + RADAR ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📉 Skill Gap Analysis</div>', unsafe_allow_html=True)
    selected_role = st.selectbox("Analyse skill gap for role:", [r for r,_ in top3], key='gap_role')

    col_radar, col_gaps = st.columns([1,1])
    with col_radar:
        st.markdown(f"**Skill Radar — You vs Ideal {selected_role}**")
        st.plotly_chart(radar_chart(user_skill_map, selected_role), use_container_width=True)

    with col_gaps:
        st.markdown(f"**Skill-by-Skill Breakdown**")
        gap_rows = skill_gap_bars(user_skill_map, selected_role)
        for g in gap_rows:
            color = "#16a34a" if g['gap']==0 else ("#f59e0b" if g['gap']<=1 else "#dc2626")
            st.markdown(f"""
            <div class="gap-row">
                <div class="gap-label">{g['skill']} — You: {g['user']}/5 &nbsp;|&nbsp; Ideal: {g['ideal']}/5 &nbsp; <span style="color:{color}">{g['status']}</span></div>
                <div class="gap-bar-bg">
                    <div class="gap-bar-fill" style="width:{g['pct']}%;background:{color}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Tools gap
        ideal_t = IDEAL_TOOLS.get(selected_role, [])
        missing_tools = [t for t in ideal_t if t not in tools_used]
        if missing_tools:
            st.markdown(f"**🛠️ Missing Tools for {selected_role}:**")
            for t in missing_tools:
                st.markdown(f"&nbsp;&nbsp;🔴 {t}")
        else:
            st.markdown("**🛠️ Tools:** ✅ All key tools covered!")

    # ── ROADMAP ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗺️ Your Personalised Preparation Roadmap</div>', unsafe_allow_html=True)

    tab_labels = [f"{ROLE_ICONS.get(r,'🎯')} {r}" for r,_ in top3]
    tabs = st.tabs(tab_labels)
    for i,(role,conf) in enumerate(top3):
        with tabs[i]:
            st.markdown(f"#### {ROLE_ICONS.get(role,'🎯')} {role} &nbsp;—&nbsp; {conf}% match")
            steps = ROADMAPS.get(role, [])
            for step in steps:
                st.markdown(f"""
                <div class="roadmap-step">
                    <div class="step-phase">{step['phase']}</div>
                    <div class="step-title">{step['title']}</div>
                    <div class="step-desc">{step['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── COURSES ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📚 Recommended Courses & Resources</div>', unsafe_allow_html=True)

    tab_labels2 = [f"{ROLE_ICONS.get(r,'🎯')} {r}" for r,_ in top3]
    tabs2 = st.tabs(tab_labels2)
    for i,(role,conf) in enumerate(top3):
        with tabs2[i]:
            st.markdown(f"#### Top courses to become a **{role}**")
            for c in COURSES.get(role,[]):
                st.markdown(f"""
                <div class="course-card">
                    <div>
                        <span class="course-platform">{c['platform']}</span><br>
                        <span class="course-title"><a href="{c['url']}" target="_blank">{c['title']}</a></span><br>
                        <span class="course-desc">{c['desc']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#94a3b8;font-size:0.8rem;padding:10px">
        🎓 Placement Readiness & Job Role Predictor &nbsp;|&nbsp; Applied Machine Learning Project &nbsp;|&nbsp;
        Random Forest · 90.76% Accuracy · 9 Job Roles · 2,380 Training Samples
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
