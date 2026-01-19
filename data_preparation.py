import polars as pl
import os

# Define columns to keep (Retaining 'statistical_cause_en' for the specific cause lookup)
COLUMNS_TO_KEEP = [
    "rdt_lines", "cause_group", "start_time", "end_time", 
    "duration_minutes", "statistical_cause_en"
]

FILE_PATTERN = "*.csv" 

print("1. Scanning and Cleaning Data...")
df = (
    pl.scan_csv(FILE_PATTERN, ignore_errors=True, glob=True)
    .select(COLUMNS_TO_KEEP)
    .with_columns([
        pl.col("start_time").str.to_datetime(strict=False).alias("start_time_dt"),
    ])
    .drop_nulls(subset=["start_time_dt", "cause_group", "rdt_lines"])
    .with_columns(
        (pl.col("start_time_dt").dt.year() >= 2017).cast(pl.Int8).alias("Post_2017_System")
    )
    .collect()
)

print("2. Feature Engineering...")
# Create time features and 'explode' the lines
df_features = df.with_columns([
    pl.col("start_time_dt").dt.hour().alias("hour_of_day"),
    pl.col("start_time_dt").dt.weekday().alias("day_of_week"),
    pl.col("start_time_dt").dt.month().alias("month"),
    pl.col("rdt_lines").str.split(',').alias("line_list")
]).explode("line_list").rename({"line_list": "single_rdt_line"})

# Create Historical Frequency Feature
line_counts = df_features.group_by("single_rdt_line").len().rename({"len": "line_disruption_count"})
df_final = df_features.join(line_counts, on="single_rdt_line", how="left")

# Create Targets
df_model = df_final.with_columns(
    pl.col("cause_group").cast(pl.Categorical).to_physical().alias("Y_Cause_Encoded")
)

# Remove raw text columns not needed for training (but keep statistical_cause_en for mapping)
df_model = df_model.drop(["rdt_lines", "start_time", "end_time", "start_time_dt", "cause_group"])

OUTPUT_FILE = "model_ready_data.parquet"
df_model.write_parquet(OUTPUT_FILE)
print(f"✅ Success! Data saved to {OUTPUT_FILE}")