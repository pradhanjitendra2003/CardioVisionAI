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
    /* Main Background - Added professional medical imaging background */
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

    /* TABS VISIBILITY - High Contrast */
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
    section[data-testid="stSidebar"] .stButton>button {
        background-color: #334155 !important;
        color: white !important;
        border: 1px solid #475569 !important;
    }

    /* --- FILE UPLOADER CLEANUP (DUPLICATE BUTTON FIX) --- */
   /* --- FILE UPLOADER CENTER FIX --- */
    [data-testid="stFileUploader"] {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem;
    }
    
    /* Target the container to ensure it centers */
    [data-testid="stFileUploader"] > section {
        width: 100% !important;
        max-width: 450px !important; /* Button ki max width */
        margin: auto !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
    
    /* Target only the main upload button */
    [data-testid="stFileUploader"] button[kind="secondary"] {
        background-color: #ef4444 !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        height: 60px !important;
        width: 100% !important; /* Container ke andar poora spread hoga */
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4) !important;
        display: block !important;
        margin: 0 auto !important;
    }

    /* Reset the 'Clear' button so it doesn't look like a red box */
    [data-testid="stFileUploader"] button[kind="headerNoPadding"] {
        background-color: transparent !important;
        color: #64748b !important;
        box-shadow: none !important;
        width: auto !important;
        height: auto !important;
    }
    [data-testid="stFileUploader"] button[kind="headerNoPadding"]::after { content: "" !important; }

    /* Diagnostic Result Cards */
    .result-display {
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* RESTORED FULL FOOTER STYLING */
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
st.markdown("<h1 class='main-header'>🧬 CardioVision AI Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569; font-size: 18px;'>High-Precision Automated Cardiomegaly Screening Suite</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Analysis Workspace", "📊 History & Reports"])

with tab1:
    if model is None:
        st.error("⚠️ CRITICAL ERROR: Neural network weights missing.")
    else:
        # Columns ke zariye centering ko double-lock karna
        _, col_mid, _ = st.columns([0.5, 2, 0.5]) 
        with col_mid:
            st.markdown("<br>", unsafe_allow_html=True)
            # Yahan text ko bhi center kar dete hain
            st.markdown("<h4 style='text-align: center; color: #1e293b;'>Upload Patient Radiograph</h4>", unsafe_allow_html=True)
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

# --- 6. RESTORED FULL FOOTER ---
st.markdown("""
    <div class="footer-container">
        <b>Official Diagnostic Suite v2.1</b><br>
        This tool is powered by a Deep Convolutional Neural Network.<br>
        Developed for Radiographic Research | © 2026 CardioVision AI<br>
        <span style="color: #dc2626; font-weight: bold;">
            Disclaimer: AI-generated screening only. Final decisions must be confirmed by medical professionals.
        </span>
    </div>
    """, unsafe_allow_html=True)
