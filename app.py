import streamlit as st
import numpy as np
import joblib
import pandas as pd
import polars as pl 
from datetime import datetime
import os

# --- CONFIG ---
st.set_page_config(page_title="ProRail AI", page_icon="🚄", layout="wide")
st.markdown("""
    <style>
    .main {background-color: white;} 
    .metric-card {
        background-color: blue; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #003082;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        background-color: #003082; 
        color: white;
        font-weight: bold;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

CAUSE_MAP = {0: "None/Cleared", 1: "Accident", 2: "Engineering Work", 3: "External Factor", 4: "Infrastructure Defect", 5: "Logistical Issue", 6: "Rolling Stock", 7: "Staff Issue", 8: "Unknown", 9: "Weather"}

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    if not os.path.exists("project_resources.pkl"): return None
    return joblib.load("project_resources.pkl")

resources = load_resources()

if not resources:
    st.error("❌ Models not found. Please run 'model_training.py' first.")
    st.stop()

# Unpack resources
clf = resources["clf"]
reg = resources["reg"]
line_encoder = resources["line_encoder"]
line_history = resources["line_history_dict"]
cause_stats_map = resources["cause_stats_map"]
feature_names = resources["feature_names"] # Load feature names to fix warnings!
df_history = resources["history_df"]

# --- UI SIDEBAR ---
st.sidebar.title("🚄 Journey Planner")
valid_lines = sorted(list(line_encoder.keys()))
selected_line = st.sidebar.selectbox("Select Route", valid_lines)
travel_date = st.sidebar.date_input("Date", min_value=datetime.today())
travel_time = st.sidebar.time_input("Time", value=datetime.now().time())
analyze_btn = st.sidebar.button("🔍 Analyze Journey Risk")

# --- MAIN DASHBOARD ---
st.title("ProRail Predictive Operations")
st.divider()

if analyze_btn:
    # 1. Feature Engineering (Live)
    dt = datetime.combine(travel_date, travel_time)
    
    # Create DataFrame to match training names perfectly
    row_data = {
        "Post_2017_System": 1,
        "hour_of_day": dt.hour,
        "day_of_week": dt.weekday(),
        "month": dt.month,
        "single_rdt_line": line_encoder[selected_line],
        "line_disruption_count": line_history.get(selected_line, 0)
    }
    features_df = pd.DataFrame([row_data], columns=feature_names)

    # 2. Inference
    pred_id = clf.predict(features_df)[0]
    pred_cause = CAUSE_MAP.get(pred_id, "Unknown")
    pred_specific = cause_stats_map.get(pred_id, "General Fault")
    confidence = np.max(clf.predict_proba(features_df)[0])
    duration = reg.predict(features_df)[0]

    # 3. Display Logic
    is_disruption = pred_id != 0
    if is_disruption:
        st.error(f"❌ DISRUPTION EXPECTED on **{selected_line}**")
    else:
        st.success(f"✅ NO MAJOR DISRUPTION FORECASTED on **{selected_line}**")
        duration = 0 # Force 0 duration if no disruption

    # 4. Metrics Columns
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><b>Cause Group</b><br>{pred_cause}</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><b>Specific Cause</b><br>{pred_specific if is_disruption else "-"}</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><b>Est. Duration</b><br>{duration:.0f} min</div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><b>Confidence</b><br>{confidence:.1%}</div>', unsafe_allow_html=True)

    # 6. History Chart
    st.divider()
    st.subheader(f"📊 Historical Profile: {selected_line}")
    subset = df_history.filter(pl.col("single_rdt_line") == selected_line)
    if subset.height > 0:
        counts = subset.group_by("Y_Cause_Encoded").len().to_pandas()
        counts['Name'] = counts['Y_Cause_Encoded'].map(CAUSE_MAP)
        st.bar_chart(counts.set_index("Name")['len'], color="#003082")
    else:
        st.write("No historical data available for this route.")

else:
    st.info("👈 Select a route and click Analyze.")