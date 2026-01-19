import polars as pl
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import os

OUTPUT_FILE = "model_ready_data.parquet"
ARTIFACTS_FILE = "project_resources.pkl"

if not os.path.exists(OUTPUT_FILE):
    print("❌ Error: Run data_preparation.py first.")
    exit()

print("1. Loading Data...")
df = pl.read_parquet(OUTPUT_FILE).drop_nulls(subset=["duration_minutes"])

# --- PREPARE DATA ---
# Define columns to exclude from features
EXCLUDE = ["Y_Cause_Encoded", "duration_minutes", "statistical_cause_en", "rdt_id", "end_time_dt"]
# Dynamically get the feature names
X_cols = [c for c in df.columns if c not in EXCLUDE]

print(f"   Using Features: {X_cols}") # Debug print

X_pd = df.select(X_cols).to_pandas()
Y_class = df.select("Y_Cause_Encoded").to_numpy().ravel()
Y_reg = df.select("duration_minutes").to_numpy().ravel()

# --- BUILD MAPPINGS ---
print("2. Building Dictionaries...")
X_pd['single_rdt_line'] = X_pd['single_rdt_line'].astype('category')
line_categories = X_pd['single_rdt_line'].cat.categories
line_encoder = {name: code for code, name in enumerate(line_categories)}

line_history_map = df.group_by("single_rdt_line").agg(pl.first("line_disruption_count"))
line_history_dict = dict(zip(line_history_map["single_rdt_line"], line_history_map["line_disruption_count"]))

cause_stats_map = df.group_by("Y_Cause_Encoded").agg(
    pl.col("statistical_cause_en").mode().first().cast(pl.Utf8)
).to_pandas().set_index('Y_Cause_Encoded')['statistical_cause_en'].to_dict()

# --- TRAIN MODELS ---
print("3. Training AI Models...")
X_pd['single_rdt_line'] = X_pd['single_rdt_line'].cat.codes
# Keep as Pandas for training to preserve feature names!
clf = lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', verbose=-1, random_state=42)
clf.fit(X_pd, Y_class)

reg = lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
reg.fit(X_pd, Y_reg)

# --- SAVE EVERYTHING ---
print(f"4. Saving 'Brain' to {ARTIFACTS_FILE}...")
artifacts = {
    "clf": clf,
    "reg": reg,
    "line_encoder": line_encoder,
    "line_history_dict": line_history_dict,
    "cause_stats_map": cause_stats_map,
    "feature_names": X_cols, # <<< NEW: Saving the column names
    "history_df": df
}
joblib.dump(artifacts, ARTIFACTS_FILE)
print("✅ Training Complete. Models saved.")