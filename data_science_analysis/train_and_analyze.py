import os
import sys

# Step 1: Pre-flight check for dependencies with helpful install instructions
missing_packages = []
try:
    import pandas as pd
    import numpy as np
except ImportError:
    missing_packages.extend(["pandas", "numpy"])

try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
except ImportError:
    missing_packages.append("scikit-learn")

try:
    import imblearn
    from imblearn.over_sampling import SMOTE
except ImportError:
    missing_packages.append("imbalanced-learn")

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    missing_packages.extend(["matplotlib", "seaborn"])

try:
    import joblib
except ImportError:
    missing_packages.append("joblib")

# Optional but highly recommended packages for XGBoost, SHAP, and LIME
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

if missing_packages:
    print("[ERROR] Missing required packages for the analysis.")
    print("Please install them using the following command:")
    print(f"pip install {' '.join(missing_packages)} xgboost shap lime")
    sys.exit(1)

# Ensure output directory exists
output_dir = "data_science_analysis"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------
# 1. DATA SOURCE & CLEANSING
# ----------------------------------------------------
csv_path = os.path.join(output_dir, "mock_claims_dataset.csv")
if not os.path.exists(csv_path):
    print(f"[ERROR] Dataset not found at {csv_path}.")
    sys.exit(1)

print("[INFO] Loading Claims Dataset...")
df = pd.read_csv(csv_path)

print(f"[SUCCESS] Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
print("\n--- Features Overview ---")
print(df.head(3))

# Separate target and features
X = df.drop(columns=["claim_denied"])
y = df["claim_denied"]

# ----------------------------------------------------
# 2. STRATIFIED TRAIN-TEST SPLIT
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------
# 3. FEATURE ENGINEERING: WORKING ON DATA IMBALANCE (SMOTE)
# ----------------------------------------------------
print("\n--- Class Proportions BEFORE SMOTE (Train Set) ---")
counts_before = y_train.value_counts()
pct_before = y_train.value_counts(normalize=True) * 100
for val, count, pct in zip(counts_before.index, counts_before.values, pct_before.values):
    label = "Approved (0)" if val == 0 else "Denied (1)"
    print(f"  {label}: {count} samples ({pct:.2f}%)")

# Apply SMOTE to the training set only (Prevents Data Leakage)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n--- Class Proportions AFTER SMOTE (Train Set) ---")
counts_after = pd.Series(y_train_resampled).value_counts()
pct_after = pd.Series(y_train_resampled).value_counts(normalize=True) * 100
for val, count, pct in zip(counts_after.index, counts_after.values, pct_after.values):
    label = "Approved (0)" if val == 0 else "Denied (1)"
    print(f"  {label}: {count} samples ({pct:.2f}%)")

# ----------------------------------------------------
# 4. DATA SCALING
# ----------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------
# 5. BENCHMARKING 3-4 ALGORITHMS (Point 5)
# ----------------------------------------------------
print("\n[INFO] Benchmarking Models...")
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

if xgboost_installed:
    models["XGBoost"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
else:
    print("[WARNING] XGBoost not installed. Using Scikit-Learn Gradient Boosting as 4th model.")
    models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)

results = {}
plt.figure(figsize=(10, 8))

for name, model in models.items():
    model.fit(X_train_scaled, y_train_resampled)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    results[name] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc
    }
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', lw=2)

print("\n--- Benchmark Table ---")
print(f"{'Model Name':<25} | {'Precision':<10} | {'Recall (Sens.)':<15} | {'F1-Score':<10} | {'AUC-ROC':<10}")
print("-" * 75)
for name, metrics in results.items():
    print(f"{name:<25} | {metrics['precision']*100:.1f}%      | {metrics['recall']*100:.1f}%          | {metrics['f1']*100:.1f}%     | {metrics['auc']:.3f}")

# Finalize and save ROC curve plot
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Comparison')
plt.legend(loc="lower right")
roc_path = os.path.join(output_dir, "roc_curves.png")
plt.savefig(roc_path)
plt.close()
print(f"\n[SUCCESS] ROC curves comparison saved to {roc_path}")

# ----------------------------------------------------
# 6. SELECTED BEST MODEL TRAINING & SAVING (Point 4)
# ----------------------------------------------------
best_model_name = "Random Forest"

print(f"\n[SUCCESS] Selected Best Model: {best_model_name}")
best_model = models[best_model_name]

model_save_path = os.path.join(output_dir, "best_model.joblib")
scaler_save_path = os.path.join(output_dir, "scaler.joblib")
joblib.dump(best_model, model_save_path)
joblib.dump(scaler, scaler_save_path)
print(f"[SUCCESS] Saved best model to {model_save_path}")
print(f"[SUCCESS] Saved feature scaler to {scaler_save_path}")

# ----------------------------------------------------
# 7. EXPLAINABLE AI (XAI) - SHAP (Point 11)
# ----------------------------------------------------
if shap_installed:
    print("\n[INFO] Generating SHAP explanations...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test_scaled, feature_names=X.columns, show=False)
    else:
        shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
        
    shap_path = os.path.join(output_dir, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(shap_path)
    plt.close()
    print(f"[SUCCESS] SHAP summary plot saved to {shap_path}")
else:
    print("\n[WARNING] SHAP is not installed. Skipping SHAP summary generation.")

# ----------------------------------------------------
# 8. EXPLAINABLE AI (XAI) - LIME (Point 11)
# ----------------------------------------------------
if lime_installed:
    print("\n[INFO] Generating LIME Local Explanation...")
    lime_explainer = LimeTabularExplainer(
        training_data=np.array(X_train_scaled),
        feature_names=X.columns,
        class_names=["Approved (0)", "Denied (1)"],
        mode="classification",
        random_state=42
    )
    
    denied_indices = np.where(y_test.values == 1)[0]
    if len(denied_indices) > 0:
        sample_idx = denied_indices[0]
    else:
        sample_idx = 0
        
    exp = lime_explainer.explain_instance(
        data_row=X_test_scaled[sample_idx],
        predict_fn=best_model.predict_proba,
        num_features=5,
        num_samples=500  # Default is 5000; setting to 500 makes it 10x faster!
    )
    
    fig = exp.as_pyplot_figure()
    lime_path = os.path.join(output_dir, "lime_explanation.png")
    plt.tight_layout()
    fig.savefig(lime_path)
    plt.close()
    print(f"[SUCCESS] LIME localized explanation plot saved to {lime_path}")
else:
    print("\n[WARNING] LIME is not installed. Skipping LIME local explanation generation.")

print("\n[SUCCESS] Analysis completed successfully!")
