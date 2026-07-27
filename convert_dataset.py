"""
Dataset Converter & High-Precision Synthetic Actuarial Data Generator.
Transforms Kaggle Medical/Insurance datasets into real-world correlated Healthcare Claims Datasets.

Usage:
    python convert_dataset.py
"""

import pandas as pd
import numpy as np
import os

print("=========================================================")
print("  Healthcare Claims Dataset Generator & Converter")
print("=========================================================")

output_dir = "data_science_analysis"
os.makedirs(output_dir, exist_ok=True)

csv_input_path = os.path.join(output_dir, "Insurance.csv")
csv_output_path = os.path.join(output_dir, "mock_claims_dataset.csv")

# Configuration for large realistic dataset generation
NUM_SAMPLES = 10000
np.random.seed(42)

def generate_realistic_actuarial_dataset(num_samples=10000):
    """Generate a realistic 10,000+ patient actuarial underwriting dataset with clinical risk rules."""
    print(f"[INFO] Generating {num_samples} realistic patient claims records...")
    
    # 1. Demographic & Clinical Distributions
    age = np.random.randint(18, 80, size=num_samples)
    
    # BMI following log-normal distribution centered around 27.5
    bmi = np.round(np.random.lognormal(mean=3.3, sigma=0.25, size=num_samples), 1)
    bmi = np.clip(bmi, 16.0, 52.0)
    
    # Smoking status (~20% smokers, higher prevalence in older group)
    smoker_prob = 0.15 + (age > 40) * 0.1
    smoker = (np.random.rand(num_samples) < smoker_prob).astype(int)
    
    # Pre-existing conditions (correlated with age and BMI)
    base_conditions = (age / 25) + (bmi / 15) - 2.5
    noise = np.random.poisson(lam=0.8, size=num_samples)
    pre_existing = np.clip(np.round(base_conditions + noise), 0, 6).astype(int)
    
    # Coverage Tier (1 = Basic, 2 = Standard, 3 = Premium)
    coverage_tier = np.random.choice([1, 2, 3], size=num_samples, p=[0.45, 0.35, 0.20])
    
    # Blood Pressure (Systolic & Diastolic correlated with Age, BMI, and Smoking)
    sys_base = 105 + (age * 0.35) + (bmi * 0.4) + (smoker * 6) + np.random.normal(0, 7, size=num_samples)
    dia_base = 68 + (age * 0.20) + (bmi * 0.25) + (smoker * 4) + np.random.normal(0, 5, size=num_samples)
    
    systolic_bp = np.clip(np.round(sys_base), 95, 195).astype(int)
    diastolic_bp = np.clip(np.round(dia_base), 60, 115).astype(int)
    
    # 2. Actuarial Claim Denial Risk Score Calculation (True Medical Logic)
    # Higher risk score -> Higher likelihood of claim denial under underwriting guidelines
    risk_score = (
        (age > 60) * 1.5 +
        (bmi > 30) * 2.0 + (bmi > 35) * 2.5 +
        (smoker == 1) * 3.0 +
        (pre_existing * 1.8) +
        (systolic_bp > 140) * 2.2 + (systolic_bp > 160) * 3.5 +
        (diastolic_bp > 90) * 1.8 +
        (coverage_tier == 1) * 2.0 - (coverage_tier == 3) * 2.5
    )
    
    # Convert risk score to probability via sigmoid function
    prob_denial = 1 / (1 + np.exp(-(risk_score - 7.5) / 2.2))
    
    # Binary denial label (1 = Denied, 0 = Approved)
    claim_denied = (np.random.rand(num_samples) < prob_denial).astype(int)
    
    df_gen = pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "smoker": smoker,
        "pre_existing_conditions": pre_existing,
        "coverage_tier": coverage_tier,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "claim_denied": claim_denied
    })
    
    return df_gen


if os.path.exists(csv_input_path):
    print(f"[INFO] Found existing input dataset at: {csv_input_path}")
    df_raw = pd.read_csv(csv_input_path)
    print(f"[INFO] Raw shape: {df_raw.shape}")
    
    if "insuranceclaim" in df_raw.columns:
        df_raw = df_raw.rename(columns={"insuranceclaim": "claim_denied"})
    
    # Synthesize missing clinical features realistically
    if "pre_existing_conditions" not in df_raw.columns:
        df_raw["pre_existing_conditions"] = np.clip(np.round((df_raw["age"]/20) + (df_raw["bmi"]/15) - 2), 0, 5).astype(int)
    if "coverage_tier" not in df_raw.columns:
        df_raw["coverage_tier"] = np.random.choice([1, 2, 3], size=len(df_raw), p=[0.45, 0.35, 0.20])
    if "systolic_bp" not in df_raw.columns:
        df_raw["systolic_bp"] = np.clip(np.round(105 + (df_raw["age"]*0.35) + (df_raw["bmi"]*0.4)), 95, 185).astype(int)
    if "diastolic_bp" not in df_raw.columns:
        df_raw["diastolic_bp"] = np.clip(np.round(68 + (df_raw["age"]*0.20) + (df_raw["bmi"]*0.25)), 60, 110).astype(int)
        
    final_cols = ["age", "bmi", "smoker", "pre_existing_conditions", "coverage_tier", "systolic_bp", "diastolic_bp", "claim_denied"]
    df_out = df_raw[final_cols]
else:
    print(f"[INFO] No Insurance.csv file found in {output_dir}/. Generating realistic {NUM_SAMPLES} row dataset...")
    df_out = generate_realistic_actuarial_dataset(NUM_SAMPLES)

# Save output
df_out.to_csv(csv_output_path, index=False)
denied_pct = (df_out["claim_denied"].sum() / len(df_out)) * 100

print(f"\n[SUCCESS] Dataset generated and saved to: {csv_output_path}")
print(f"[METRICS] Total Rows: {len(df_out)} | Approved: {len(df_out) - df_out['claim_denied'].sum()} ({100-denied_pct:.1f}%) | Denied: {df_out['claim_denied'].sum()} ({denied_pct:.1f}%)")
print("=========================================================")
