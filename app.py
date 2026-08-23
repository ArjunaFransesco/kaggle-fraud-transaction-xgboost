import streamlit as st
import pandas as pd
import numpy as np
import joblib, os

st.set_page_config(page_title="Fraud Anomaly Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Real-Time Transaction Fraud & Anomaly Detector")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **Portfolio**: [GitHub](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("💳 Transaction Telemetry")
    amt = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=25000.0, value=350.0, step=10.0)
    v1 = st.slider("PCA Latent Feature V1", -10.0, 10.0, -2.1)
    v2 = st.slider("PCA Latent Feature V2", -10.0, 10.0, 3.4)
    v3 = st.slider("PCA Latent Feature V3", -10.0, 10.0, -3.8)
    v4 = st.slider("PCA Latent Feature V4", -10.0, 10.0, 2.9)
    device = st.slider("Device Trust Score (0-1)", 0.0, 1.0, 0.35)

with col2:
    st.subheader("🚨 Fraud Risk Assessment")
    model_path = os.path.join(os.path.dirname(__file__), "models/fraud_xgboost_model.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        amt_log = np.log1p(amt)
        interaction = (v2 * v4) / (device + 0.01)
        feats = pd.DataFrame([{
            "transaction_amount_usd": amt,
            "amount_log": amt_log,
            "latent_v1": v1,
            "latent_v2": v2,
            "latent_v3": v3,
            "latent_v4": v4,
            "device_trust_score": device,
            "risk_interaction": interaction
        }])
        prob = float(model.predict_proba(feats)[0, 1])
        st.metric("Fraud Probability Score", f"{prob:.1%}")
        
        if prob > 0.65:
            st.error("🚨 HIGH RISK FRAUD: Immediate Transaction Block & Security Challenge Triggered!")
        elif prob > 0.30:
            st.warning("⚠️ SUSPICIOUS ANOMALY: Route to 2FA / Manual Review Queue.")
        else:
            st.success("✅ LEGITIMATE: Transaction cleared with high confidence.")
