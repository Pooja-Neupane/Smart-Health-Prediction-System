# app.py
"""
Streamlit app for Smart Health Prediction System.
Make sure you have run train.py at least once to create model.joblib and features.json.

Run:
    streamlit run app.py
"""

import streamlit as st
import joblib
import json
import numpy as np
from pathlib import Path

MODEL_FILE = "model.joblib"
LABEL_ENCODER_FILE = "label_encoder.joblib"
FEATURES_FILE = "features.json"
PRECAUTIONS_FILE = "precautions.json"

st.set_page_config(page_title="Smart Health Prediction", layout="centered")

st.title("Smart Health Prediction System")
st.write("Select symptoms and get predicted diseases (top suggestions). This is a demo system and not a substitute for professional medical advice.")

# Load model & metadata
if not Path(MODEL_FILE).exists() or not Path(FEATURES_FILE).exists() or not Path(LABEL_ENCODER_FILE).exists():
    st.error("Model or feature files not found. Please run `python train.py` first.")
    st.stop()

model = joblib.load(MODEL_FILE)
le = joblib.load(LABEL_ENCODER_FILE)
with open(FEATURES_FILE, "r") as f:
    features = json.load(f)
with open(PRECAUTIONS_FILE, "r") as f:
    precautions = json.load(f)

# Sidebar: symptom selection
st.sidebar.header("Input Symptoms")
st.sidebar.markdown("Select all symptoms that apply:")

selected_symptoms = st.sidebar.multiselect("Symptoms", options=features)

st.sidebar.markdown("---")
top_k = st.sidebar.slider("Number of top predictions to show", min_value=1, max_value=5, value=3)

if st.sidebar.button("Predict"):
    # Build input vector
    x = np.array([[1 if feat in selected_symptoms else 0 for feat in features]])
    # Predict probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        idx_sorted = np.argsort(proba)[::-1][:top_k]
        preds = [(le.inverse_transform([i])[0], proba[i]) for i in idx_sorted]
    else:
        # fallback
        pred_idx = model.predict(x)[0]
        preds = [(le.inverse_transform([pred_idx])[0], None)]

    st.subheader("Predictions")
    for disease, p in preds:
        if p is None:
            st.write(f"- **{disease}**")
        else:
            st.write(f"- **{disease}** — probability: {p:.2f}")

    st.subheader("Recommendations / Precautions")
    primary = preds[0][0]
    if primary in precautions:
        st.write(precautions[primary])
    else:
        st.write("Rest, stay hydrated, and consult a healthcare professional if symptoms persist or worsen.")

    st.info("Reminder: This model is for educational/demo use. For any serious concerns, consult a licensed medical professional.")

else:
    st.write("Select symptoms in the left sidebar and click **Predict** to see results.")

st.markdown("---")
st.write("Model metadata:")
st.write(f"- Trained to recognize {len(le.classes_)} diseases.")
st.write(f"- Symptom vocabulary size: {len(features)}")
