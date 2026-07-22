import pandas as pd
import numpy as np
import os

print("[START] Starting Kaggle Dataset Conversion...")

# Locate file in data_science_analysis/Insurance.csv
csv_input_path = os.path.join("data_science_analysis", "Insurance.csv")
csv_output_path = os.path.join("data_science_analysis", "mock_claims_dataset.csv")

if not os.path.exists(csv_input_path):
    print(f"[ERROR] Input file not found at {csv_input_path}. Please make sure Insurance.csv is inside data_science_analysis/")
    exit(1)

# Load Kaggle dataset
print(f"Reading {csv_input_path}...")
df = pd.read_csv(csv_input_path)

# 1. Rename target column 'insuranceclaim' to 'claim_denied'
df = df.rename(columns={"insuranceclaim": "claim_denied"})

# 2. Synthesize missing columns that our FastAPI application expects
print("Synthesizing missing columns (pre_existing_conditions, coverage_tier, blood pressure)...")
np.random.seed(42)  # For reproducible synthesis

if "pre_existing_conditions" not in df.columns:
    df["pre_existing_conditions"] = np.where(
        df["age"] > 50, 
        np.random.randint(1, 4, size=len(df)), 
        np.random.randint(0, 2, size=len(df))
    )

if "coverage_tier" not in df.columns:
    df["coverage_tier"] = np.random.choice([1, 2, 3], size=len(df), p=[0.5, 0.3, 0.2])

if "systolic_bp" not in df.columns:
    df["systolic_bp"] = np.random.randint(110, 160, size=len(df))

if "diastolic_bp" not in df.columns:
    df["diastolic_bp"] = np.random.randint(70, 95, size=len(df))

# 3. Keep only the exact features required by the project's ML models
final_columns = [
    "age", "bmi", "smoker", "pre_existing_conditions", 
    "coverage_tier", "systolic_bp", "diastolic_bp", "claim_denied"
]
df_final = df[final_columns]

# 4. Save directly as mock_claims_dataset.csv
df_final.to_csv(csv_output_path, index=False)
print(f"[SUCCESS] Converted dataset saved to: {csv_output_path}")
