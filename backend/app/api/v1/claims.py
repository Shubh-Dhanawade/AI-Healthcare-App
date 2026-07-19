"""
Claims Analytics Router
Serves data science statistics (SMOTE proportions, model metrics) and explainer plots.
"""

from fastapi import APIRouter
from typing import Dict, List, Any

router = APIRouter()

@router.get("/stats")
async def get_claims_stats() -> Dict[str, Any]:
    """Retrieve claims prediction statistics, SMOTE proportions, and model benchmark evaluation metrics."""
    return {
        "dataset_info": {
            "total_samples": 1000,
            "features": ["age", "bmi", "smoker", "pre_existing_conditions", "coverage_tier", "systolic_bp", "diastolic_bp"]
        },
        "smote_proportions": {
            "before": [
                {"class": "Approved (0)", "count": 650, "percentage": 81.25},
                {"class": "Denied (1)", "count": 150, "percentage": 18.75}
            ],
            "after": [
                {"class": "Approved (0)", "count": 650, "percentage": 50.00},
                {"class": "Denied (1)", "count": 650, "percentage": 50.00}
            ]
        },
        "benchmarks": [
            {
                "model_name": "Logistic Regression",
                "precision": 30.2,
                "recall": 68.4,
                "f1_score": 41.9,
                "auc_roc": 0.671,
                "is_selected": False
            },
            {
                "model_name": "Decision Tree",
                "precision": 16.2,
                "recall": 34.2,
                "f1_score": 22.0,
                "auc_roc": 0.482,
                "is_selected": False
            },
            {
                "model_name": "Random Forest",
                "precision": 22.9,
                "recall": 28.9,
                "f1_score": 25.6,
                "auc_roc": 0.582,
                "is_selected": True
            },
            {
                "model_name": "XGBoost",
                "precision": 23.1,
                "recall": 31.6,
                "f1_score": 26.7,
                "auc_roc": 0.553,
                "is_selected": False
            }
        ],
        "plots": {
            "roc_curves": "/data_science_analysis/roc_curves.png",
            "shap_summary": "/data_science_analysis/shap_summary.png",
            "lime_explanation": "/data_science_analysis/lime_explanation.png"
        }
    }
