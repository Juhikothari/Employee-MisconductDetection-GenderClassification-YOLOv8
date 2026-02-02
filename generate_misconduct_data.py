import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ─── Setup Streamlit ───────────────────────────────────────────────────
st.set_page_config(page_title="Generate Misconduct Data (20 Rows)", layout="wide")
st.title("📊 Generate and Download Misconduct Data (20 Rows)")

# ─── Generate Random Misconduct Data ───────────────────────────────────
def generate_random_misconduct_data(num_records=20):
    np.random.seed(42)  # For reproducibility
    timestamps = [datetime.now() - timedelta(seconds=i*10) for i in range(num_records)]
    timestamps.reverse()  # Chronological order
    behaviors = ["Awake", "Sleeping 😴", "Eating", "Using Phone", "Head Bending", "Crowding"]
    genders = ["Male", "Female", "Unknown"]
    chair_statuses = ["Occupied ✅", "Loitering ❌"]
    
    data = {
        'timestamp': timestamps,
        'ear': np.clip(np.random.normal(0.3, 0.1, num_records), 0.1, 0.5),  # Eye Aspect Ratio
        'mar': np.clip(np.random.normal(0.15, 0.1, num_records), 0.0, 1.0),  # Mouth Aspect Ratio
        'head_bend_ratio': np.clip(np.random.normal(0.1, 0.05, num_records), 0.0, 0.5),
        'behavior': np.random.choice(behaviors, num_records, p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]),
        'gender': np.random.choice(genders, num_records, p=[0.45, 0.45, 0.1]),
        'chair_status': np.random.choice(chair_statuses, num_records, p=[0.7, 0.3]),
        'ground_truth_behavior': np.random.choice(behaviors, num_records, p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Adjust metrics to align with behaviors for realism
    df.loc[df['behavior'] == "Sleeping 😴", 'ear'] = np.clip(np.random.normal(0.15, 0.05, sum(df['behavior'] == "Sleeping 😴")), 0.1, 0.22)
    df.loc[df['behavior'] == "Eating", 'mar'] = np.clip(np.random.normal(0.25, 0.1, sum(df['behavior'] == "Eating")), 0.15, 0.5)
    df.loc[df['behavior'] == "Head Bending", 'head_bend_ratio'] = np.clip(np.random.normal(0.2, 0.05, sum(df['behavior'] == "Head Bending")), 0.15, 0.5)
    
    # Save to CSV
    csv_path = "misconduct_data_20.csv"
    df.to_csv(csv_path, index=False)
    return df, csv_path

# ─── Generate and Display Data ─────────────────────────────────────────
df, csv_path = generate_random_misconduct_data()
st.header("Data Preview (20 Rows)")
st.dataframe(df, use_container_width=True)

# ─── Download CSV Button ───────────────────────────────────────────────
with open(csv_path, "rb") as file:
    st.download_button(
        label="Download Misconduct Data CSV (20 Rows)",
        data=file,
        file_name="misconduct_data_20.csv",
        mime="text/csv"
    )