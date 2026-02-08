import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

@st.cache_resource
def load_failure_model():
    return load_model("failure_predictor_lstm.h5")

model = load_failure_model()


# Load Data + Model
df = pd.read_csv("motor_data.csv")
model = load_model("failure_predictor_lstm.h5")


st.set_page_config(page_title="Neural-Sync Dashboard", layout="wide")

st.title("⚙️ Neural-Sync — Predictive Maintenance Dashboard")
st.markdown("Real-time Motor Failure Prediction + Remaining Useful Life (RUL)")

# Show Latest Sensor Data
st.subheader("📈 Live Sensor Signals (Last 200 Timesteps)")
st.line_chart(df[["vibration", "temperature"]].tail(200))

# Prepare Input for Prediction
data = df[["vibration","temperature"]].values
sequence_length = 20

sample_input = data[-sequence_length:].reshape(1, sequence_length, 2)

# Predict Failure Probability
failure_prob = model.predict(sample_input)[0][0]

# Remaining Useful Life
rul = (1 - failure_prob) * 48

# Display Metrics
st.subheader("🧠 AI Predictions")

col1, col2 = st.columns(2)

col1.metric("Failure Probability (Next 48 hrs)", f"{failure_prob*100:.2f}%")
col2.metric("Remaining Useful Life (RUL)", f"{rul:.2f} hrs")

# Alert System
if failure_prob > 0.7:
    st.error("🚨 ALERT: Motor likely to FAIL within 48 hours!")
elif failure_prob > 0.4:
    st.warning("⚠️ Warning: Motor health degrading.")
else:
    st.success("✅ Motor is Healthy.")

st.markdown("---")
st.markdown("Built with ❤️ using LSTM + Autoencoder + Streamlit")
