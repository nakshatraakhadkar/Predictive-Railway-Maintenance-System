import joblib
import pandas as pd # using pandas for dataframe creation
import random
from datetime import datetime, timedelta
import os

# --- CONFIG ---
ARTIFACTS_FILE = "project_resources.pkl"
CAUSE_MAP = {
    0: "None/Cleared", 1: "Accident", 2: "Engineering Work", 3: "External Factor",
    4: "Infrastructure Defect", 5: "Logistical Issue", 6: "Rolling Stock",
    7: "Staff Issue", 8: "Unknown", 9: "Weather"
}

# --- 1. LOAD THE "BRAIN" ---
if not os.path.exists(ARTIFACTS_FILE):
    print(f"❌ Error: {ARTIFACTS_FILE} not found. Run model_training.py first.")
    exit()

print("Loading saved AI models...")
resources = joblib.load(ARTIFACTS_FILE)

clf = resources["clf"]
reg = resources["reg"]
line_encoder = resources["line_encoder"]
line_history = resources["line_history_dict"]
cause_stats_map = resources["cause_stats_map"]
feature_names = resources["feature_names"] # <<< NEW: Load names

# --- 2. GENERATE FAKE BOOKINGS ---
print("\n--- Simulating User Bookings for Next Week ---")

valid_lines = list(line_encoder.keys())
synthetic_bookings = []
start_date = datetime.now() + timedelta(days=7) 

for i in range(5):
    random_line = random.choice(valid_lines)
    random_hour = random.randint(6, 22) 
    random_day = random.randint(0, 6)   
    
    booking = {
        "id": f"TKT-{1000+i}",
        "line": random_line,
        "time": start_date + timedelta(days=random_day, hours=random_hour)
    }
    synthetic_bookings.append(booking)

# --- 3. RUN PREDICTIONS ---
print(f"{'TICKET':<10} | {'DATE':<20} | {'ROUTE':<30} | {'PREDICTION':<25} | {'SPECIFIC CAUSE'}")
print("-" * 110)

for b in synthetic_bookings:
    dt = b['time']
    line_name = b['line']
    
    # Create a Dictionary first to match column names perfectly
    # The order here doesn't matter as much because we use a DataFrame, 
    # but we must use the exact same logic as training.
    
    row_data = {
        "Post_2017_System": 1,
        "hour_of_day": dt.hour,
        "day_of_week": dt.weekday(),
        "month": dt.month,
        "single_rdt_line": line_encoder[line_name],
        "line_disruption_count": line_history.get(line_name, 0)
    }
    
    # Create DataFrame with specific column order (enforces feature names)
    features_df = pd.DataFrame([row_data], columns=feature_names)
    
    # Inference
    pred_id = clf.predict(features_df)[0]
    pred_cause = CAUSE_MAP.get(pred_id, "Unknown")
    pred_specific = cause_stats_map.get(pred_id, "-")
    duration = reg.predict(features_df)[0]
    
    # Logic: Only show specific cause/duration if there is a disruption
    if pred_id == 0:
        pred_text = "✅ OK"
        pred_specific = "-"
    else:
        pred_text = f"⚠️ {pred_cause} ({duration:.0f}m)"

    print(f"{b['id']:<10} | {dt.strftime('%Y-%m-%d %H:%M'):<20} | {line_name:<30} | {pred_text:<25} | {pred_specific}")