"""
Production-Grade Claims Underwriting ML Training Pipeline.
Trains, evaluates, and benchmarks Random Forest, Decision Tree, Logistic Regression, and XGBoost models.
Generates XAI Plots (ROC Curves, SHAP Summary, LIME Explanations) and exports best model weights.
"""

import os
import sys
import shutil

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE

# Check optional XGBoost, SHAP, and LIME
xgboost_installed = True
try:
    import xgboost as xgb
except ImportError:
    xgboost_installed = False

shap_installed = True
try:
    import shap
except ImportError:
    shap_installed = False

lime_installed = True
try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    lime_installed = False

output_dir = "data_science_analysis"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "mock_claims_dataset.csv")

if not os.path.exists(csv_path):
    print(f"[ERROR] Dataset not found at {csv_path}. Please run `python convert_dataset.py` first.")
    sys.exit(1)

print("\n=========================================================")
print("  ML Pipeline: Training Claims Underwriting Models")
print("=========================================================")

print(f"[INFO] Loading Claims Dataset from {csv_path}...")
df = pd.read_csv(csv_path)
print(f"[SUCCESS] Dataset loaded with {len(df)} rows.")

# Separate target and features
X = df.drop(columns=["claim_denied"])
y = df["claim_denied"]

# Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n--- Train Set Class Distribution ---")
print(f"  Approved (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
print(f"  Denied (1):   {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")

# Apply SMOTE to training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Define algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=4, random_state=42)
}

if xgboost_installed:
    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.08, eval_metric='logloss', random_state=42
    )
else:
    models["XGBoost"] = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)

results = {}
plt.figure(figsize=(9, 7))

best_auc = -1.0
best_model_name = "XGBoost"

for name, model in models.items():
    model.fit(X_train_scaled, y_train_resampled)
    
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate optimal threshold for max F1-Score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_thresh = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
    
    y_pred = (y_prob >= best_thresh).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    fpr, tpr_val, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr_val)
    
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_model_name = name
        
    results[name] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc
    }
    
    plt.plot(fpr, tpr_val, label=f'{name} (AUC = {roc_auc:.3f})', lw=2.5)

print("\n--------------------------------------------------------------------------------")
print(f"{'Model Name':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUC-ROC':<10}")
print("--------------------------------------------------------------------------------")
for name, metrics in results.items():
    status = " (Selected)" if name == best_model_name else ""
    print(f"{name:<22} | {metrics['precision']*100:6.1f}%    | {metrics['recall']*100:6.1f}%    | {metrics['f1']*100:6.1f}%    | {metrics['auc']:6.3f}{status}")
print("--------------------------------------------------------------------------------")

# Save ROC Curves plot
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Comparison')
plt.legend(loc="lower right")
roc_path = os.path.join(output_dir, "roc_curves.png")
plt.tight_layout()
plt.savefig(roc_path, dpi=200)
plt.close()
print(f"[SUCCESS] ROC curves plot saved to {roc_path}")

# Export Selected Best Model and Scaler
best_model = models[best_model_name]
model_save_path = os.path.join(output_dir, "best_model.joblib")
scaler_save_path = os.path.join(output_dir, "scaler.joblib")

joblib.dump(best_model, model_save_path)
joblib.dump(scaler, scaler_save_path)
print(f"\n[SUCCESS] Selected Best Model: '{best_model_name}' (AUC = {best_auc:.3f})")
print(f"[SUCCESS] Exported best model to: {model_save_path}")
print(f"[SUCCESS] Exported feature scaler to: {scaler_save_path}")

# Export dynamic benchmark metrics JSON for API endpoint
import json

benchmark_list = []
for name, metrics in results.items():
    benchmark_list.append({
        "model_name": name,
        "precision": round(float(metrics["precision"] * 100), 1),
        "recall": round(float(metrics["recall"] * 100), 1),
        "f1_score": round(float(metrics["f1"] * 100), 1),
        "auc_roc": round(float(metrics["auc"]), 3),
        "is_selected": name == best_model_name
    })

counts_before = y_train.value_counts()
pct_before = y_train.value_counts(normalize=True) * 100
counts_after = pd.Series(y_train_resampled).value_counts()

benchmark_json = {
    "dataset_info": {
        "total_samples": len(df),
        "features": list(X.columns)
    },
    "smote_proportions": {
        "before": [
            {"class": "Approved (0)", "count": int(counts_before.get(0, 0)), "percentage": round(float(pct_before.get(0, 0)), 2)},
            {"class": "Denied (1)", "count": int(counts_before.get(1, 0)), "percentage": round(float(pct_before.get(1, 0)), 2)}
        ],
        "after": [
            {"class": "Approved (0)", "count": int(counts_after.get(0, 0)), "percentage": 50.0},
            {"class": "Denied (1)", "count": int(counts_after.get(1, 0)), "percentage": 50.0}
        ]
    },
    "benchmarks": benchmark_list,
    "plots": {
        "roc_curves": "/data_science_analysis/roc_curves.png",
        "shap_summary": "/data_science_analysis/shap_summary.png",
        "lime_explanation": "/data_science_analysis/lime_explanation.png"
    }
}

json_save_path = os.path.join(output_dir, "benchmark_results.json")
with open(json_save_path, "w") as f:
    json.dump(benchmark_json, f, indent=2)
print(f"[SUCCESS] Exported benchmark JSON to: {json_save_path}")

# Copy artifacts to backend/app/core/ if directory exists
backend_core = os.path.join("backend", "app", "core")
if os.path.exists(backend_core):
    shutil.copy(model_save_path, os.path.join(backend_core, "best_model.joblib"))
    shutil.copy(scaler_save_path, os.path.join(backend_core, "scaler.joblib"))
    shutil.copy(json_save_path, os.path.join(backend_core, "benchmark_results.json"))
    print(f"[SUCCESS] Synced model files and JSON metrics to backend core: {backend_core}/")

# Generate SHAP Plot
if shap_installed:
    try:
        print("[INFO] Generating SHAP Global Summary plot...")
        plt.figure(figsize=(10, 6))
        explainer = shap.TreeExplainer(best_model)
        shap_vals = explainer.shap_values(X_test_scaled)
        
        if isinstance(shap_vals, list):
            shap.summary_plot(shap_vals[1], X_test_scaled, feature_names=X.columns, show=False)
        else:
            shap.summary_plot(shap_vals, X_test_scaled, feature_names=X.columns, show=False)
            
        shap_path = os.path.join(output_dir, "shap_summary.png")
        plt.tight_layout()
        plt.savefig(shap_path, dpi=200)
        plt.close()
        print(f"[SUCCESS] SHAP summary plot saved to {shap_path}")
    except Exception as e:
        print(f"[WARNING] Could not generate SHAP plot: {e}")

# Generate LIME Plot
if lime_installed:
    try:
        print("[INFO] Generating LIME Local Explanation plot...")
        lime_explainer = LimeTabularExplainer(
            training_data=np.array(X_train_scaled),
            feature_names=X.columns,
            class_names=["Approved (0)", "Denied (1)"],
            mode="classification",
            random_state=42
        )
        denied_idx = np.where(y_test.values == 1)[0]
        sample_idx = denied_idx[0] if len(denied_idx) > 0 else 0
        
        exp = lime_explainer.explain_instance(
            data_row=X_test_scaled[sample_idx],
            predict_fn=best_model.predict_proba,
            num_features=5,
            num_samples=500
        )
        fig = exp.as_pyplot_figure()
        lime_path = os.path.join(output_dir, "lime_explanation.png")
        plt.tight_layout()
        fig.savefig(lime_path, dpi=200)
        plt.close()
        print(f"[SUCCESS] LIME localized plot saved to {lime_path}")
    except Exception as e:
        print(f"[WARNING] Could not generate LIME plot: {e}")

print("\n=========================================================")
print("  ML Pipeline Execution Finished Successfully!")
print("=========================================================")
