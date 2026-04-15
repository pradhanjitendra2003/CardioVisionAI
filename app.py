import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import pandas as pd

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="CardioVision AI Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Advanced Professional Styling (CSS) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), 
                    url("https://images.unsplash.com/photo-1576091160550-2173dba999ef?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Header Styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
        font-weight: 800;
        text-align: center;
        padding-bottom: 5px;
    }

    /* TABS ALIGNMENT - Analysis (Left) & History (Right) */
    div[data-testid="stTabs"] [role="tablist"] {
        display: flex;
        justify-content: space-between;
        width: 100%;
    }

    /* History & Reports tab ko Right side push karna */
    div[data-testid="stTabs"] [role="tablist"] button:nth-child(2) {
        margin-left: auto !important;
    }

    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }

    button[aria-selected="true"] {
        color: #ef4444 !important;
        border-bottom-color: #ef4444 !important;
    }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] { background-color: #0f172a !important; }
    section[data-testid="stSidebar"] * { color: #f8fafc !important; }

    /* --- UPLOADER CLEANUP (Only Red Button) --- */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
    }

    [data-testid="stFileUploader"] > section {
        background-color: transparent !important;
        border: none !important;
        width: auto !important;
    }

    [data-testid="stFileUploader"] section button {
        background-color: #ef4444 !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 15px 30px !important;
        font-weight: bold !important;
        font-size: 18px !important;
        min-width: 250px !important;
        height: 60px !important;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4) !important;
        border: none !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"], 
    [data-testid="stFileUploader"] small {
        display: none !important;
    }

    .result-display {
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .footer-container {
        text-align: center;
        color: #1e293b !important;
        font-weight: 500;
        font-size: 0.9rem;
        padding: 30px;
        margin-top: 50px;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Session State & Model Loading ---
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_medical_model():
    try:
        return tf.keras.models.load_model('cardiomegaly_detection_model.h5')
    except:
        return None

model = load_medical_model()

# --- 4. Sidebar ---
with st.sidebar:
    st.markdown("## 🛠️ Diagnostics Panel")
    st.markdown("---")
    st.info("🧬 **System Status:** Online\n\n💡 **Instructions:**\n1. Upload a clear frontal Chest X-ray.\n2. Click 'Run System Analysis'.")
    st.markdown("### Clinical Metrics")
    st.metric(label="Total Scans Analyzed", value=len(st.session_state.history))
    if st.button("🗑️ Reset Diagnostic Session"):
        st.session_state.history = []
        st.rerun()

# --- 5. Main Dashboard ---
st.markdown("<h1 class='main-header'>🧬CardioVision AI Analytics🧬</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569; font-size: 18px;'>Enhanced Cardiomegaly Detection using Deep Learning</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Analysis Workspace", "📊 History & Reports"])

with tab1:
    if model is None:
        st.error("⚠️ CRITICAL ERROR: Neural network weights missing.")
    else:
        _, col_mid, _ = st.columns([0.6, 2, 0.6]) 
        with col_mid:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center; color: #1e293b; margin-bottom: -20px;'>📤 Select Patient X-ray</h4>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            st.divider()
            col_img, col_res = st.columns([1, 1], gap="large")
            
            image = Image.open(uploaded_file)
            
            with col_img:
                st.markdown("#### 📥 Patient Radiograph")
                st.image(image, use_container_width=True)
            
            with col_res:
                st.markdown("#### 🧠 AI Diagnostic Inference")
                if st.button("🚀 Run System Analysis", type="primary", use_container_width=True):
                    with st.status("Analyzing Radiographic Features...", expanded=True) as status:
                        img = ImageOps.grayscale(image)
                        img = img.resize((128, 128))
                        img_array = np.array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=[0, -1])
                        
                        time.sleep(1.2)
                        st.write("Evaluating Cardiothoracic Ratio...")
                        prediction = model.predict(img_array, verbose=0)
                        confidence = float(prediction[0][0])
                        status.update(label="Inference Verified", state="complete", expanded=False)

                    is_normal = confidence > 0.5
                    final_conf = confidence if is_normal else (1 - confidence)
                    label = "NORMAL" if is_normal else "CARDIOMEGALY DETECTED"
                    bg_color = "#059669" if is_normal else "#dc2626"
                    
                    st.session_state.history.append({
                        "Time": time.strftime("%H:%M:%S"),
                        "Finding": label,
                        "Probability": f"{final_conf*100:.1f}%"
                    })

                    st.markdown(f"""
                        <div class="result-display" style="background-color: {bg_color};">
                            <h2 style='color: white; margin:0;'>DIAGNOSIS: {label}</h2>
                            <p style='margin:0; opacity: 0.9;'>Clinical Confidence: {final_conf*100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    st.metric("Detection Confidence", f"{final_conf*100:.2f}%")

with tab2:
    st.markdown("### 📋 Analysis Report History")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
    else:
        st.info("No diagnostic history available.")

# --- 6. Footer ---
st.markdown("""
    <div class="footer-container">
        <b>CardioVision AI</b><br>
        This tool is powered by a Deep Convolutional Neural Network.<br>
        <span style="color: #dc2626; font-weight: bold;">
            Disclaimer: For educational and screening support only. Final decisions must be confirmed by medical professionals.
        </span>
    </div>
    """, unsafe_allow_html=True)
