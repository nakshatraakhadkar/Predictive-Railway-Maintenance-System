import polars as pl
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_absolute_error, classification_report
import os

# --- CONFIG ---
OUTPUT_FILE = "model_ready_data.parquet"
ARTIFACTS_FILE = "project_resources.pkl"
RANDOM_SEED = 42

def main():
    # 1. Load Data & Resources
    if not os.path.exists(OUTPUT_FILE) or not os.path.exists(ARTIFACTS_FILE):
        print("❌ Error: Missing files. Run data_preparation.py and model_training.py first.")
        return

    print("Loading Data and Models...")
    df = pl.read_parquet(OUTPUT_FILE).drop_nulls(subset=["duration_minutes"])
    resources = joblib.load(ARTIFACTS_FILE)
    
    clf = resources["clf"]
    reg = resources["reg"]
    feature_names = resources["feature_names"]

    # 2. Re-create the Exact Feature Set
    # We must do the exact same cleaning/splitting as we did in training
    EXCLUDE = ["Y_Cause_Encoded", "duration_minutes", "statistical_cause_en"]
    X_cols = [c for c in df.columns if c not in EXCLUDE]
    
    X_pd = df.select(X_cols).to_pandas()
    
    # Convert text column to numbers using the SAVED encoder to match perfectly
    # (We map manually to ensure we use the same IDs)
    line_encoder = resources["line_encoder"]
    X_pd['single_rdt_line'] = X_pd['single_rdt_line'].map(line_encoder).fillna(-1).astype(int)
    
    # Ensure correct column order
    X_final = X_pd[feature_names]
    
    # Get Targets
    Y_class = df.select("Y_Cause_Encoded").to_numpy().ravel()
    Y_reg = df.select("duration_minutes").to_numpy().ravel()

    # 3. Split Data (Train vs Test)
    # We use the SAME random_state so we get the same "Test Set" as before
    _, X_test, _, Y_test_class = train_test_split(X_final, Y_class, test_size=0.2, random_state=RANDOM_SEED, stratify=Y_class)
    _, X_test_reg, _, Y_test_reg = train_test_split(X_final, Y_reg, test_size=0.2, random_state=RANDOM_SEED)

    # 4. Calculate F1-Score (Classification)
    print("\n--- 1. CLASSIFICATION METRICS (Cause Prediction) ---")
    Y_pred_class = clf.predict(X_test)
    
    # Macro Average: Treats all classes equally (important for rare accidents)
    f1 = f1_score(Y_test_class, Y_pred_class, average='macro')
    weighted_f1 = f1_score(Y_test_class, Y_pred_class, average='weighted')
    
    print(f"✅ Macro F1-Score:    {f1:.2%}")
    print(f"✅ Weighted F1-Score: {weighted_f1:.2%}")

    # 5. Calculate MAE (Regression)
    print("\n--- 2. REGRESSION METRICS (Duration Prediction) ---")
    Y_pred_reg = reg.predict(X_test_reg)
    mae = mean_absolute_error(Y_test_reg, Y_pred_reg)
    
    print(f"✅ Mean Absolute Error (MAE): {mae:.2f} minutes")
    
    print("\n" + "="*30)
    print("      NUMBERS FOR PPT      ")
    print("="*30)
    print(f"F1-Score: {f1:.2f}")
    print(f"MAE:      {mae:.1f} min")
    print("="*30)

if __name__ == "__main__":
    main()