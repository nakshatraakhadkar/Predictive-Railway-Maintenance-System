import joblib
import numpy as np
import pandas as pd
import polars as pl
import os

# --- CONFIGURATION ---
ARTIFACTS_FILE = "project_resources.pkl"

def main():
    # 1. Load the Saved Brain
    if not os.path.exists(ARTIFACTS_FILE):
        print(f"❌ Error: {ARTIFACTS_FILE} not found. Run model_training.py first.")
        return

    print("LOADING RESOURCES...")
    resources = joblib.load(ARTIFACTS_FILE)
    
    # Unpack what we need
    clf = resources["clf"]                 # The AI Model
    df_history = resources["history_df"]   # The Historical Data
    line_encoder = resources["line_encoder"] # The Map (Name -> Number)
    feature_names = resources["feature_names"] # The Column Names model expects

    print(f"✅ Loaded {df_history.height:,} historical records.")

    # 2. Re-create the Features (X)
    # We need to turn the history data into the exact format the model trained on
    # so we can ask it: "How confident were you about these past events?"
    
    print("PREPARING DATA FOR ANALYSIS...")
    
    # Convert Polars history to Pandas
    X_pd = df_history.to_pandas()
    
    # Apply the Line Encoder (Convert String 'Amsterdam' -> Int '105')
    # We use .map() to look up the number. Fill unknowns with -1 (though shouldn't be any).
    X_pd['single_rdt_line'] = X_pd['single_rdt_line'].map(line_encoder).fillna(-1).astype(int)
    
    # Select ONLY the columns the model knows about, in the right order
    X_final = X_pd[feature_names]

    # 3. Calculate Confidence
    print("RUNNING BATCH PREDICTION (This might take a moment)...")
    
    # predict_proba gives probabilities for ALL classes [[0.1, 0.9], [0.8, 0.2]...]
    all_probabilities = clf.predict_proba(X_final)
    
    # np.max gives us just the winning score for each row [0.9, 0.8...]
    confidence_scores = np.max(all_probabilities, axis=1)
    
    # 4. Calculate the Business KPI
    avg_confidence = np.mean(confidence_scores)
    
    # --- TERMINAL OUTPUT ---
    print("\n" + "="*40)
    print("      📊 BUSINESS KPI REPORT      ")
    print("="*40)
    print(f"Total Incidents Analyzed:  {len(confidence_scores):,}")
    print("-" * 40)
    print(f"AVERAGE MODEL CONFIDENCE:  {avg_confidence:.1%}")
    print("-" * 40)
    print("Interpretation:")
    print(f"On average, the AI is {avg_confidence:.1%} sure of its")
    print("diagnosis when analyzing historical disruptions.")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()