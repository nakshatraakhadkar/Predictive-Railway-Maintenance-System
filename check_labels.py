import polars as pl
import os

# Scan the original CSVs again just to get the string-to-number mapping
FILE_PATTERN = "*.csv"
print("Scanning for labels...")

df_lazy = pl.scan_csv(FILE_PATTERN, ignore_errors=True, glob=True)

# Select only cause_group
df_labels = df_lazy.select("cause_group").collect()

# Create the mapping logic exactly like we did in Data Prep
# Polars assigns integer codes based on alphabetical order of the strings
unique_causes = df_labels["cause_group"].unique().sort()

print("\n--- CAUSE GROUP MAPPING ---")
for i, cause in enumerate(unique_causes):
    print(f"Class {i}: {cause}")