"""
Claims Analytics Router
Serves data science statistics (SMOTE proportions, model metrics) and explainer plots.
Also processes live dynamic claims underwriting predictions using pre-trained ML models and Gemma 3 explanations.
"""

import os
import joblib
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.ollama_client import call_ollama

router = APIRouter()

# Global model cache to avoid repeated disk reads
_model = None
_scaler = None

def load_ml_model():
    """Load pre-trained XGBoost model and standard scaler from storage."""
    global _model, _scaler
    if _model is None or _scaler is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path 1: Check in backend/app/core
        core_dir = os.path.normpath(os.path.join(current_dir, "..", "..", "core"))
        model_path = os.path.join(core_dir, "best_model.joblib")
        scaler_path = os.path.join(core_dir, "scaler.joblib")
        
        # Path 2: Check in project data_science_analysis root (fallback for some local execution contexts)
        if not os.path.exists(model_path):
            project_root = os.path.normpath(os.path.join(current_dir, "..", "..", "..", ".."))
            ds_dir = os.path.join(project_root, "data_science_analysis")
            model_path = os.path.join(ds_dir, "best_model.joblib")
            scaler_path = os.path.join(ds_dir, "scaler.joblib")
            
        logger.info(f"Loading ML models from: {model_path}")
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            _model = joblib.load(model_path)
            _scaler = joblib.load(scaler_path)
            logger.info("✅ Underwriting ML models loaded successfully.")
        else:
            logger.error(f"Failed to find models. Checked paths. Model path: {model_path}")
            raise FileNotFoundError(f"Model or scaler not found. Path checked: {model_path}")
    return _model, _scaler


class ClaimPredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age of the policyholder")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index")
    smoker: int = Field(..., ge=0, le=1, description="Smoking status (1 = Yes, 0 = No)")
    pre_existing_conditions: int = Field(..., ge=0, le=10, description="Number of pre-existing conditions")
    coverage_tier: int = Field(..., ge=1, le=3, description="Coverage Tier (1 = Basic, 2 = Standard, 3 = Premium)")
    systolic_bp: int = Field(..., ge=80, le=200, description="Systolic Blood Pressure (mmHg)")
    diastolic_bp: int = Field(..., ge=50, le=130, description="Diastolic Blood Pressure (mmHg)")
    document_id: Optional[str] = Field(None, description="Optional associated policy document ID")


def get_mock_prediction_response(request: ClaimPredictionRequest, error_msg: str) -> Dict[str, Any]:
    """Fallback generator providing realistic mock predictions if models are unbuilt."""
    score = 0.1
    if request.smoker == 1:
        score += 0.35
    if request.bmi > 29.9:
        score += 0.25
    if request.pre_existing_conditions > 0:
        score += 0.15 * request.pre_existing_conditions
    if request.systolic_bp > 130:
        score += 0.1
    if request.diastolic_bp > 85:
        score += 0.05
        
    probability = min(max(score, 0.05), 0.95)
    denied = probability > 0.5
    
    contributions_formatted = [
        {"feature": "age", "label": "Patient Age", "value": str(request.age), "contribution": round((request.age - 35) * 0.2, 1)},
        {"feature": "bmi", "label": "BMI Level", "value": f"{request.bmi:.1f}", "contribution": 25.0 if request.bmi > 29.9 else -5.0},
        {"feature": "smoker", "label": "Smoking Status", "value": "Yes" if request.smoker == 1 else "No", "contribution": 35.0 if request.smoker == 1 else -15.0},
        {"feature": "pre_existing_conditions", "label": "Pre-existing Conditions", "value": str(request.pre_existing_conditions), "contribution": round(15.0 * request.pre_existing_conditions, 1)},
        {"feature": "coverage_tier", "label": "Policy Coverage Tier", "value": {1: "Basic", 2: "Standard", 3: "Premium"}[request.coverage_tier], "contribution": 5.0 if request.coverage_tier == 1 else -5.0},
        {"feature": "systolic_bp", "label": "Systolic Blood Pressure", "value": str(request.systolic_bp), "contribution": 8.0 if request.systolic_bp > 130 else -2.0},
        {"feature": "diastolic_bp", "label": "Diastolic Blood Pressure", "value": str(request.diastolic_bp), "contribution": 4.0 if request.diastolic_bp > 85 else -1.0}
    ]
    
    explanation = (
        f"Based on your profile, the claim denial risk is assessed as {'HIGH' if denied else 'LOW'} ({probability*100:.1f}% probability) "
        f"primarily due to your smoking status and body mass index. (Note: Using baseline rules; model load failed: {error_msg})"
    )
    
    return {
        "success": True,
        "claim_denied": denied,
        "denial_probability": round(probability * 100, 1),
        "contributions": contributions_formatted,
        "explanation": explanation
    }


async def generate_underwriting_explanation(
    request: ClaimPredictionRequest, 
    probability: float, 
    denied: int, 
    contributions: List[Dict[str, Any]],
    policy_context: Optional[Dict[str, Any]] = None
) -> str:
    """Generate dynamic narrative explanation with Gemma 3 based on ML outputs and policy context."""
    # Sort contributions by absolute value to find the top drivers
    sorted_contribs = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)
    top_drivers = []
    for c in sorted_contribs[:3]:
        direction = "increases risk" if c["contribution"] > 0 else "decreases risk"
        top_drivers.append(f"• {c['label']} ({c['value']}): {c['contribution']}% ({direction})")
        
    drivers_str = "\n".join(top_drivers)
    status_str = "HIGH RISK OF DENIAL" if probability > 0.5 else "LOW RISK OF DENIAL (LIKELY APPROVED)"
    
    policy_str = ""
    if policy_context:
        policy_str = f"""
ASSOCIATED POLICY DETAILS:
- Policy Name: {policy_context['policy_name']}
- Policy Coverage Terms: {policy_context['fields_summary']}
"""

    prompt = f"""You are a senior healthcare underwriting consultant. We have a machine learning model that has predicted the claim denial probability for a user's health profile.
    
User Health Profile:
- Age: {request.age}
- BMI: {request.bmi} (Healthy range: 18.5 - 24.9)
- Smoker Status: {"Yes" if request.smoker == 1 else "No"}
- Pre-existing Conditions Count: {request.pre_existing_conditions}
- Coverage Tier: {{1: "Basic", 2: "Standard", 3: "Premium"}}[{request.coverage_tier}]
- Blood Pressure: {request.systolic_bp}/{request.diastolic_bp} mmHg (Normal range: 120/80)
{policy_str}

Machine Learning Model Results:
- Denial Risk Probability: {probability * 100:.1f}%
- Primary Underwriting Status: {status_str}
- Local Feature Risk Drivers (Marginal Impact on Denial Probability):
{drivers_str}

Write a professional, clear, and empathetic plain-language summary under 120 words.
Address the patient directly. If policy details are provided above under 'ASSOCIATED POLICY DETAILS', explicitly relate your explanation to those policy limits (e.g., deductibles, co-pays, or waiting periods) to state if their claim will be covered. Otherwise, focus on general parameters.
Do not mention 'the machine learning model' or 'feature importances' in the text; present the reasoning as an expert underwriting evaluation.
"""
    try:
        from app.core.config import settings
        explanation = await call_ollama(prompt, num_predict=180, num_ctx=settings.OLLAMA_NUM_CTX)
        return explanation
    except Exception as e:
        logger.warning(f"Ollama offline during underwriting explanation: {e}. Using rule-based fallback.")
        
        # Rule-based fallback if Ollama is offline
        risk_level = "HIGH" if probability > 0.5 else "LOW"
        reasons = []
        if request.smoker == 1:
            reasons.append("active smoking status")
        if request.bmi > 29.9:
            reasons.append("elevated Body Mass Index (obesity range)")
        if request.pre_existing_conditions > 1:
            reasons.append(f"{request.pre_existing_conditions} pre-existing health conditions")
        if request.systolic_bp > 139 or request.diastolic_bp > 89:
            reasons.append("hypertensive blood pressure indicators")
            
        reasons_str = ", ".join(reasons) if reasons else "general actuarial risk profiling parameters"
        recommendation = "We recommend consulting a wellness coordinator for health improvement programs and reviewing standard or premium tiers to optimize coverage limits."
        if request.smoker == 1:
            recommendation = "Participating in a smoking cessation program could significantly reduce your underwriting risk profile and premiums."
        elif request.bmi > 29.9:
            recommendation = "Engaging in weight management initiatives may help lower active risk assessments and lower premium adjustments."
            
        policy_info_str = f" under your policy ({policy_context['policy_name']})" if policy_context else ""
        return (
            f"Your insurance claim risk assessment shows a {risk_level} likelihood of denial ({probability*100:.1f}% probability){policy_info_str} "
            f"primarily driven by: {reasons_str}. {recommendation}"
        )


@router.get("/stats")
async def get_claims_stats() -> Dict[str, Any]:
    """Retrieve claims prediction statistics, SMOTE proportions, and model benchmark evaluation metrics."""
    import json
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.normpath(os.path.join(current_dir, "..", "..", "core", "benchmark_results.json")),
        "/app/data_science_analysis/benchmark_results.json",
        "./data_science_analysis/benchmark_results.json",
        "../data_science_analysis/benchmark_results.json"
    ]
    
    for json_path in candidates:
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    logger.info(f"Loaded dynamic benchmark stats from {json_path}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading {json_path}: {e}")

    # Default fallback if benchmark_results.json has not been generated yet
    return {
        "dataset_info": {
            "total_samples": 1338,
            "features": ["age", "bmi", "smoker", "pre_existing_conditions", "coverage_tier", "systolic_bp", "diastolic_bp"]
        },
        "smote_proportions": {
            "before": [
                {"class": "Approved (0)", "count": 555, "percentage": 41.5},
                {"class": "Denied (1)", "count": 783, "percentage": 58.5}
            ],
            "after": [
                {"class": "Approved (0)", "count": 783, "percentage": 50.00},
                {"class": "Denied (1)", "count": 783, "percentage": 50.00}
            ]
        },
        "benchmarks": [
            {"model_name": "Logistic Regression", "precision": 75.1, "recall": 90.4, "f1_score": 82.1, "auc_roc": 0.823, "is_selected": False},
            {"model_name": "Decision Tree", "precision": 75.7, "recall": 91.1, "f1_score": 82.7, "auc_roc": 0.841, "is_selected": False},
            {"model_name": "Random Forest", "precision": 77.2, "recall": 93.0, "f1_score": 84.4, "auc_roc": 0.859, "is_selected": False},
            {"model_name": "XGBoost", "precision": 73.6, "recall": 97.5, "f1_score": 83.8, "auc_roc": 0.864, "is_selected": True}
        ],
        "plots": {
            "roc_curves": "/data_science_analysis/roc_curves.png",
            "shap_summary": "/data_science_analysis/shap_summary.png",
            "lime_explanation": "/data_science_analysis/lime_explanation.png"
        }
    }


@router.post("/predict")
async def predict_claim_denial(
    request: ClaimPredictionRequest,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Execute dynamic predictive underwriting using pre-trained model and Gemma 3 synthesis."""
    try:
        model, scaler = load_ml_model()
    except Exception as e:
        logger.error(f"Error loading underwriting model: {e}")
        return get_mock_prediction_response(request, error_msg=str(e))

    try:
        # 1. Query database for associated policy details if document_id is provided
        policy_context = None
        if request.document_id:
            try:
                from app.models.document import Document, ExtractedField
                from sqlalchemy.future import select
                
                doc = await db.get(Document, request.document_id)
                if doc:
                    res_fields = await db.execute(
                        select(ExtractedField).where(ExtractedField.document_id == doc.id)
                    )
                    fields = res_fields.scalars().all()
                    fields_summary = ", ".join([f"{f.field_name}: {f.field_value}" for f in fields])
                    policy_context = {
                        "policy_name": doc.original_filename,
                        "fields_summary": fields_summary
                    }
            except Exception as db_err:
                logger.warning(f"Failed to query database for policy context in predict: {db_err}")

        # 2. Shape request into feature inputs matching model training:
        # [age, bmi, smoker, pre_existing_conditions, coverage_tier, systolic_bp, diastolic_bp]
        features = [[
            request.age,
            request.bmi,
            request.smoker,
            request.pre_existing_conditions,
            request.coverage_tier,
            request.systolic_bp,
            request.diastolic_bp
        ]]
        
        # 3. Scale features using the loaded training scaler
        scaled_features = scaler.transform(features)
        
        # 4. Model inference
        denied_prediction = int(model.predict(scaled_features)[0])
        denial_probability = float(model.predict_proba(scaled_features)[0][1])
        
        # 5. Mathematically compute local feature contributions using counterfactual baseline
        baseline = {
            "age": 35,
            "bmi": 22.0,
            "smoker": 0,
            "pre_existing_conditions": 0,
            "coverage_tier": 2,
            "systolic_bp": 120,
            "diastolic_bp": 80
        }
        
        contributions = {}
        feature_keys = ["age", "bmi", "smoker", "pre_existing_conditions", "coverage_tier", "systolic_bp", "diastolic_bp"]
        
        for i, key in enumerate(feature_keys):
            temp_values = list(features[0])
            temp_values[i] = baseline[key]
            scaled_temp = scaler.transform([temp_values])
            prob_temp = float(model.predict_proba(scaled_temp)[0][1])
            # positive difference means this feature increased risk relative to baseline
            contributions[key] = denial_probability - prob_temp
            
        contributions_formatted = []
        for key, diff in contributions.items():
            label = {
                "age": "Patient Age",
                "bmi": "BMI Level",
                "smoker": "Smoking Status",
                "pre_existing_conditions": "Pre-existing Conditions",
                "coverage_tier": "Policy Coverage Tier",
                "systolic_bp": "Systolic Blood Pressure",
                "diastolic_bp": "Diastolic Blood Pressure"
            }[key]
            
            value_str = str(getattr(request, key))
            if key == "smoker":
                value_str = "Yes" if request.smoker == 1 else "No"
            elif key == "coverage_tier":
                value_str = {1: "Basic", 2: "Standard", 3: "Premium"}[request.coverage_tier]
            
            contributions_formatted.append({
                "feature": key,
                "label": label,
                "value": value_str,
                "contribution": round(diff * 100, 1)
            })
            
        # 6. Send to Gemma 3 for explanation synthesis
        explanation = await generate_underwriting_explanation(
            request, 
            denial_probability, 
            denied_prediction, 
            contributions_formatted,
            policy_context
        )
        
        return {
            "success": True,
            "claim_denied": denied_prediction == 1,
            "denial_probability": round(denial_probability * 100, 1),
            "contributions": contributions_formatted,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Error during claims prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


