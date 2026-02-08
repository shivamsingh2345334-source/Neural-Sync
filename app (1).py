import streamlit as st
import pandas as pd

# -----------------------------

# -----------------------------

st.set_page_config(
    page_title="Neural-Sync Dashboard",
    layout="wide"
)

st.title("⚙️ Neural-Sync — Predictive Maintenance Dashboard")
st.markdown("AI-powered Motor Monitoring + Remaining Useful Life (RUL) Estimation")

st.markdown("---")

# -----------------------------

# -----------------------------
st.sidebar.header("📂 Upload Motor Sensor Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (vibration, temperature)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File Uploaded Successfully!")
else:
    st.info("Using default demo dataset...")

   
    df = pd.DataFrame({
        "vibration": [0.2, 0.3, 0.25, 0.4, 0.8, 1.2, 1.5],
        "temperature": [35, 36, 36.5, 37, 40, 45, 50]
    })

# -----------------------------

# -----------------------------
st.subheader("📈 Live Motor Sensor Signals")

st.line_chart(df)

st.markdown("---")

# -----------------------------

# -----------------------------
st.subheader("🧠 AI Health Prediction")


failure_prob = 0.82
rul = (1 - failure_prob) * 48

col1, col2 = st.columns(2)

col1.metric("Failure Probability (Next 48 hrs)", f"{failure_prob*100:.1f}%")
col2.metric("Remaining Useful Life (RUL)", f"{rul:.1f} hours")

# -----------------------------

# -----------------------------
st.subheader("🚨 Motor Status")

if failure_prob > 0.7:
    st.error("🚨 ALERT: Motor likely to FAIL within 48 hours!")
elif failure_prob > 0.4:
    st.warning("⚠️ Warning: Motor health degrading.")
else:
    st.success("✅ Motor is Healthy.")

st.markdown("---")

# -----------------------------

# -----------------------------
st.markdown(
    """
    ✅ Built with Streamlit  
    ⚙️ Project: Neural-Sync Predictive Maintenance AI  
    👨‍💻 Developer: Shivam Singh  
    """
)
