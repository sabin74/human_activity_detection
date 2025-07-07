# app.py

import streamlit as st
import numpy as np
from utils.predictor import predict_activity

# Streamlit UI config
st.set_page_config(page_title="HAR with TXT Upload", layout="centered")
st.title("📄 Human Activity Recognition (HAR) — TXT File Input")
st.markdown("Upload a `.txt` file with **561 values** (space/comma/tab-separated) for prediction.")

# Upload text file
uploaded_file = st.file_uploader("📁 Upload TXT file with 561 values", type=["txt"])

if uploaded_file is not None:
    try:
        # Read and decode content
        content = uploaded_file.read().decode("utf-8").strip()

        # Detect delimiter
        if "," in content:
            values = content.split(",")
        elif "\t" in content:
            values = content.split("\t")
        else:
            values = content.split()  # fallback = space

        # Convert to float array
        features = np.array(values, dtype=float)

        # Validate length
        if features.shape != (561,):
            st.error(f"❌ File must contain exactly 561 values. Got {features.shape[0]}.")
        else:
            activity = predict_activity(features)
            st.success(f"✅ Predicted Activity: **{activity}**")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.markdown("---")
st.markdown("💡 Example format:\n```\n34 67 23 89 ... (561 values total)\n```")
