"""
AI Service — Ollama Integration with intelligent mock fallback.
When Ollama is unavailable, returns realistic demo data for testing.
"""

import json
import re
import httpx
from loguru import logger
from typing import Optional

from app.core.config import settings


# ─────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────

SUMMARIZATION_PROMPT = """You are a healthcare insurance expert. Analyze this insurance policy and provide a JSON summary.

DOCUMENT TEXT:
{document_text}

Return ONLY valid JSON with these exact keys:
{{
  "summary_text": "A brief 1-paragraph summary of the policy (maximum 100 words)",
  "coverage_summary": "List top 3 key coverages (bullet points, max 50 words total)",
  "exclusions_summary": "List top 3 exclusions (bullet points, max 50 words total)",
  "waiting_period_summary": "List key waiting periods (bullet points, max 50 words total)",
  "premium_summary": "List deductible, co-payment and premium info (bullet points, max 50 words total)"
}}"""


FIELD_EXTRACTION_PROMPT = """Extract key insurance policy fields from this document. Return ONLY valid JSON.
Keep every single value extremely short and concise (under 10 words per value). If a field is not mentioned, use null.

DOCUMENT TEXT:
{document_text}

Return JSON:
{{
  "policy_name": "Name of the insurance policy (e.g. 'Individual - Gold Plan')",
  "insurer_name": "Name of the insurance company (e.g. 'LifeGuard Health Insurance Co.')",
  "policy_number": "Policy number (e.g. 'LG-2025-IND-00742')",
  "sum_insured": "Maximum coverage amount (e.g. '10,00,000')",
  "premium_amount": "Total premium payable (e.g. '14,691')",
  "deductible": "Deductible amount, use null if none",
  "co_payment": "Co-payment percentage or amount, use null if none",
  "waiting_period": "Initial waiting period (e.g. '30 days')",
  "coverage_type": "Type of coverage (e.g. 'Individual')",
  "policy_term": "Duration of the policy (e.g. '1 year')",
  "network_hospitals": "Network hospitals listed (brief comma-separated list)",
  "pre_existing_coverage": "Pre-existing disease coverage waiting period (e.g. '48 months')",
  "maternity_coverage": "Maternity benefit details (e.g. '50,000 after 2 years')",
  "room_rent_limit": "Room rent sub-limit (e.g. '5,000 / Day')",
  "claim_process": "How to file a claim (e.g. 'Call 1800-XXX-XXXX')"
}}"""


RISK_ANALYSIS_PROMPT = """Analyze this insurance policy for risky or unfavorable clauses. Identify up to 3 key risk areas. Return ONLY valid JSON.
Keep descriptions, explanations, and recommendations under 15 words each.

DOCUMENT TEXT:
{document_text}

Return:
{{
  "risks": [
    {{
      "clause_text": "Exact short problematic clause text from document",
      "risk_type": "waiting_period|exclusion|deductible|co_payment|hidden_condition|coverage_limit",
      "severity": "low|medium|high",
      "explanation": "Brief explanation of risk (max 15 words)",
      "recommendation": "Brief recommendation (max 15 words)"
    }}
  ],
  "overall_risk_level": "low|medium|high"
}}"""


# ─────────────────────────────────────────
# Mock Data (used when Ollama is unavailable)
# ─────────────────────────────────────────

MOCK_SUMMARY = {
    "summary_text": (
        "This healthcare insurance policy provides comprehensive coverage for hospitalization, "
        "medical procedures, and outpatient treatments. The policy covers a wide range of medical "
        "conditions and includes benefits for critical illness, daycare procedures, and pre and "
        "post-hospitalization expenses.\n\n"
        "The policy includes a network of 5,000+ hospitals across India where cashless treatment "
        "is available. Room rent is covered up to the sum insured limit, with ICU charges also "
        "covered separately. Ambulance charges up to ₹2,000 per hospitalization are included.\n\n"
        "Annual health check-up benefits are provided after completing one claim-free year. The "
        "policy also includes a no-claim bonus of 10% increase in sum insured for each claim-free "
        "year, up to a maximum of 50% of the original sum insured."
    ),
    "coverage_summary": (
        "• Hospitalization expenses including room rent, ICU, nursing charges\n"
        "• Pre-hospitalization expenses up to 30 days before admission\n"
        "• Post-hospitalization expenses up to 60 days after discharge\n"
        "• Daycare procedures that don't require 24-hour hospitalization\n"
        "• Ambulance charges up to ₹2,000 per hospitalization\n"
        "• Domiciliary treatment when hospitalization is not possible\n"
        "• AYUSH treatments (Ayurveda, Yoga, Unani, Siddha, Homeopathy)"
    ),
    "exclusions_summary": (
        "• Pre-existing diseases during the initial 48-month waiting period\n"
        "• Cosmetic or aesthetic treatments\n"
        "• Dental treatment except due to accident\n"
        "• Spectacles, contact lenses, and hearing aids\n"
        "• Experimental or unproven treatments\n"
        "• Injuries from self-infliction or hazardous activities\n"
        "• War, nuclear, or radiation-related injuries\n"
        "• Infertility and assisted reproduction"
    ),
    "waiting_period_summary": (
        "• Initial waiting period: 30 days for all illnesses (except accidents)\n"
        "• Pre-existing conditions: 48 months (4 years) from policy inception\n"
        "• Specific diseases (hernia, cataracts, joint replacement): 24 months\n"
        "• Maternity benefits: 9 months from policy inception\n"
        "• Waiting periods may be waived with additional premium payment"
    ),
    "premium_summary": (
        "• Annual premium varies by age, sum insured, and city of residence\n"
        "• Co-payment: 20% of claim amount for each hospitalization\n"
        "• Deductible: ₹5,000 per hospitalization (applicable before insurance kicks in)\n"
        "• Premium increases by 10-15% annually based on claim history\n"
        "• GST at 18% applicable on base premium\n"
        "• Premium discounts available for family floater plans"
    ),
}

MOCK_FIELDS = [
    {"field_name": "Policy Name", "field_value": "Star Health Comprehensive Plan", "field_category": "policy_info"},
    {"field_name": "Insurer Name", "field_value": "Star Health and Allied Insurance Co. Ltd.", "field_category": "policy_info"},
    {"field_name": "Policy Number", "field_value": "P/211111/01/2024/000001", "field_category": "policy_info"},
    {"field_name": "Sum Insured", "field_value": "₹5,00,000", "field_category": "coverage"},
    {"field_name": "Premium Amount", "field_value": "₹12,500 per annum", "field_category": "premium"},
    {"field_name": "Deductible", "field_value": "₹5,000 per hospitalization", "field_category": "premium"},
    {"field_name": "Co Payment", "field_value": "20% of eligible claim amount", "field_category": "premium"},
    {"field_name": "Waiting Period", "field_value": "30 days initial, 48 months for pre-existing", "field_category": "restrictions"},
    {"field_name": "Coverage Type", "field_value": "Individual / Family Floater", "field_category": "coverage"},
    {"field_name": "Policy Term", "field_value": "1 year (renewable)", "field_category": "policy_info"},
    {"field_name": "Network Hospitals", "field_value": "5,000+ hospitals across India", "field_category": "coverage"},
    {"field_name": "Pre Existing Coverage", "field_value": "Covered after 48 months waiting period", "field_category": "coverage"},
    {"field_name": "Maternity Coverage", "field_value": "Up to ₹25,000, 9-month waiting period", "field_category": "coverage"},
    {"field_name": "Room Rent Limit", "field_value": "1% of Sum Insured per day", "field_category": "restrictions"},
    {"field_name": "Claim Process", "field_value": "Cashless at network hospitals; Reimbursement within 30 days", "field_category": "process"},
]

MOCK_RISKS = [
    {
        "clause_text": "All pre-existing diseases shall not be covered during the first 48 months of the policy.",
        "risk_type": "waiting_period",
        "severity": "high",
        "explanation": "A 48-month (4-year) waiting period for pre-existing conditions is extremely long. If you have diabetes, hypertension, or any chronic condition, you'll pay premiums for 4 years before receiving any benefit for those conditions.",
        "recommendation": "Negotiate for a shorter waiting period (12-24 months) or look for insurers offering immediate coverage for pre-existing conditions with a medical underwriting.",
    },
    {
        "clause_text": "Co-payment of 20% shall be applicable for each and every claim under this policy.",
        "risk_type": "co_payment",
        "severity": "high",
        "explanation": "A 20% co-payment on every claim means you always pay 20% out of pocket. On a ₹5 lakh claim, you'd owe ₹1 lakh yourself. This can be financially devastating for large medical bills.",
        "recommendation": "Request a zero co-payment plan or opt for a co-payment waiver rider. Compare plans without co-payment clauses as the premium difference is often worth it.",
    },
    {
        "clause_text": "Room rent shall be limited to 1% of the Sum Insured per day. If room rent exceeds this limit, proportionate deduction shall apply.",
        "risk_type": "coverage_limit",
        "severity": "medium",
        "explanation": "The 1% room rent cap means only ₹5,000/day for a ₹5L policy. Most private hospitals charge ₹8,000-15,000/day for standard rooms. If you exceed this, ALL medical charges (doctor fees, procedures) are proportionately reduced — not just the room cost.",
        "recommendation": "Opt for a plan with no room rent sub-limit or a higher cap. Alternatively, choose a shared accommodation to stay within limits during hospitalization.",
    },
    {
        "clause_text": "Deductible of ₹5,000 applicable per hospitalization event.",
        "risk_type": "deductible",
        "severity": "medium",
        "explanation": "You pay ₹5,000 from your own pocket for every hospitalization before insurance kicks in. For frequent hospitalizations, this adds up significantly.",
        "recommendation": "Factor the deductible into your emergency fund planning. Consider plans with zero deductible if you expect frequent hospital visits.",
    },
    {
        "clause_text": "Cosmetic or aesthetic treatments, dental procedures (except accidental), and vision correction are excluded.",
        "risk_type": "exclusion",
        "severity": "low",
        "explanation": "Standard exclusions that are common across most policies. Dental and vision costs can be significant over time.",
        "recommendation": "Consider supplemental dental and vision insurance separately. Budget for these out-of-pocket expenses annually.",
    },
]


# ─────────────────────────────────────────
# Ollama Client
# ─────────────────────────────────────────

async def call_ollama(prompt: str, model: Optional[str] = None) -> str:
    """Call Ollama API. Raises httpx.ConnectError if unavailable."""
    model = model or settings.OLLAMA_MODEL
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 2048,
        },
    }

    logger.info(f"Calling Ollama model={model}")
    async with httpx.AsyncClient(timeout=240.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from model response."""
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        text = json_match.group(1)

    text = text.strip()
    start_idx = min(
        (text.find("{") if text.find("{") != -1 else len(text)),
        (text.find("[") if text.find("[") != -1 else len(text)),
    )
    if start_idx == len(text):
        return {}

    text = text[start_idx:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for i in range(len(text), 0, -1):
            try:
                return json.loads(text[:i])
            except json.JSONDecodeError:
                continue
        return {}


# ─────────────────────────────────────────
# AI Service Functions
# ─────────────────────────────────────────

async def generate_summary(document_text: str) -> dict:
    """Generate AI summary. Falls back to demo data if Ollama unavailable."""
    truncated = document_text[:6000] if len(document_text) > 6000 else document_text

    try:
        response = await call_ollama(SUMMARIZATION_PROMPT.format(document_text=truncated))
        result = extract_json_from_response(response)
        if result.get("summary_text"):
            logger.info("✅ Ollama summarization successful")
            
            def clean_field(val):
                if val is None:
                    return None
                if isinstance(val, list):
                    return "\n".join(str(item) for item in val)
                return str(val)

            return {
                "summary_text": clean_field(result.get("summary_text", MOCK_SUMMARY["summary_text"])),
                "coverage_summary": clean_field(result.get("coverage_summary")),
                "exclusions_summary": clean_field(result.get("exclusions_summary")),
                "waiting_period_summary": clean_field(result.get("waiting_period_summary")),
                "premium_summary": clean_field(result.get("premium_summary")),
            }
    except Exception as e:
        logger.warning(f"Ollama unavailable ({e}), using demo summary data")

    # Return realistic mock data
    return dict(MOCK_SUMMARY)


async def extract_policy_fields(document_text: str) -> list[dict]:
    """Extract key fields. Falls back to demo data if Ollama unavailable."""
    truncated = document_text[:6000] if len(document_text) > 6000 else document_text

    try:
        response = await call_ollama(FIELD_EXTRACTION_PROMPT.format(document_text=truncated))
        result = extract_json_from_response(response)
        if result:
            logger.info("✅ Ollama field extraction successful")
            field_category_map = {
                "policy_name": "policy_info", "insurer_name": "policy_info",
                "policy_number": "policy_info", "sum_insured": "coverage",
                "premium_amount": "premium", "deductible": "premium",
                "co_payment": "premium", "waiting_period": "restrictions",
                "coverage_type": "coverage", "policy_term": "policy_info",
                "network_hospitals": "coverage", "pre_existing_coverage": "coverage",
                "maternity_coverage": "coverage", "room_rent_limit": "restrictions",
                "claim_process": "process",
            }
            fields = []
            for key, value in result.items():
                if value and value not in ("null", None):
                    fields.append({
                        "field_name": key.replace("_", " ").title(),
                        "field_value": str(value),
                        "field_category": field_category_map.get(key, "general"),
                    })
            return fields
    except Exception as e:
        logger.warning(f"Ollama unavailable ({e}), using demo field data")

    return list(MOCK_FIELDS)


async def analyze_risks(document_text: str) -> dict:
    """Detect risky clauses. Falls back to demo data if Ollama unavailable."""
    truncated = document_text[:6000] if len(document_text) > 6000 else document_text

    try:
        response = await call_ollama(RISK_ANALYSIS_PROMPT.format(document_text=truncated))
        result = extract_json_from_response(response)
        if result.get("risks"):
            logger.info("✅ Ollama risk analysis successful")
            return {
                "risks": [
                    {
                        "clause_text": r.get("clause_text", ""),
                        "risk_type": r.get("risk_type", "general"),
                        "severity": r.get("severity", "medium").lower(),
                        "explanation": r.get("explanation"),
                        "recommendation": r.get("recommendation"),
                    }
                    for r in result["risks"]
                ],
                "overall_risk_level": result.get("overall_risk_level", "medium"),
            }
    except Exception as e:
        logger.warning(f"Ollama unavailable ({e}), using demo risk data")

    return {"risks": list(MOCK_RISKS), "overall_risk_level": "high"}
