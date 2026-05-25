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


COMPARISON_PROMPT = """You are an expert insurance advisor. Compare the following healthcare insurance policies side-by-side based on their costs, coverage limits, restrictions, and risk profiles.

POLICIES DATA:
{policies_data}

Return ONLY valid JSON with these exact keys:
{{
  "synthesis": "A brief 1-paragraph summary of the main differences and trade-offs between these policies (maximum 100 words)",
  "best_for": "A bulleted or short description specifying who each policy is best suited for (maximum 50 words)",
  "verdict": "Your expert recommendation/verdict on which policy offers the best overall value and why (maximum 50 words)",
  "feature_winners": [
    {{
      "feature": "Premium Cost | Coverage Limits | Deductibles & Co-payments | Network Size | Waiting Periods",
      "winner": "Exact Name of the winning policy (or 'Tie' if equal)",
      "reason": "Very brief explanation of why this policy wins for this feature (maximum 15 words)"
    }}
  ]
}}"""


RAG_PROMPT = """You are a helpful healthcare insurance assistant. Answer the user's query using the provided context from the policy documents. 
Keep your response concise, professional, and clear. If the answer is not mentioned in the context, say "I cannot find this information in the selected policies."

CONTEXT FROM INSURANCE POLICIES:
{context}

USER QUERY:
{query}

CHAT HISTORY:
{chat_history}

Please provide a structured, friendly response. If quoting policy terms, specify the source document name."""


TRANSLATE_PROMPT = """You are an expert translator. Translate the following text into {target_language}.
Return ONLY the translated text without any explanation, markdown wrapper, or introductory words.

TEXT:
{text}"""


CLAIMS_CHECKLIST_PROMPT = """You are a health insurance claims auditor. Create a claim documentation checklist for:
POLICY: {policy_name}
TREATMENT/ILNESS: {treatment_type}
POLICY DETAILS:
{fields_summary}

Return ONLY valid JSON. Keep explanations short.
Return JSON format:
{{
  "checklist": [
    {{
      "document_name": "Name of required document (e.g. Original Discharge Summary)",
      "importance": "mandatory|optional",
      "description": "Brief description of why this is required (maximum 15 words)"
    }}
  ],
  "claim_steps": [
    "Step 1: ...",
    "Step 2: ..."
  ],
  "estimated_approval_days": "e.g. 7-10 business days"
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


async def generate_comparison_synthesis(policies_data: list[dict]) -> dict:
    """Compare multiple policies side-by-side. Falls back to dynamic mock comparison if Ollama is offline."""
    # Build text representation of the policies being compared
    formatted_policies = []
    policy_names = []
    
    for idx, p in enumerate(policies_data):
        fields = p.get("extracted_fields", [])
        policy_name = next((f["field_value"] for f in fields if f["field_name"].lower() in ("policy name", "policy_name")), "Unknown Policy")
        insurer_name = next((f["field_value"] for f in fields if f["field_name"].lower() in ("insurer name", "insurer_name")), "Unknown Insurer")
        
        full_name = f"{insurer_name} - {policy_name}"
        policy_names.append(full_name)
        
        sum_insured = next((f["field_value"] for f in fields if f["field_name"].lower() in ("sum insured", "sum_insured")), "Not specified")
        premium = next((f["field_value"] for f in fields if f["field_name"].lower() in ("premium amount", "premium_amount")), "Not specified")
        deductible = next((f["field_value"] for f in fields if f["field_name"].lower() in ("deductible",)), "None")
        copay = next((f["field_value"] for f in fields if f["field_name"].lower() in ("co payment", "co_payment")), "None")
        
        summary_text = p.get("summary", {}).get("summary_text", "") if p.get("summary") else ""
        risks = p.get("risk_analyses", [])
        risk_level = p.get("overall_risk_level", "medium")
        
        formatted_policies.append(
            f"POLICY #{idx+1}: {full_name}\n"
            f"- Sum Insured: {sum_insured}\n"
            f"- Premium: {premium}\n"
            f"- Deductible: {deductible}\n"
            f"- Co-payment: {copay}\n"
            f"- AI Summary: {summary_text}\n"
            f"- Overall Risk Level: {risk_level}\n"
            f"- Key Risks: {', '.join([r['risk_type'] for r in risks[:3]]) if risks else 'None'}\n"
        )
    
    policies_text = "\n\n".join(formatted_policies)
    
    try:
        response = await call_ollama(COMPARISON_PROMPT.format(policies_data=policies_text))
        result = extract_json_from_response(response)
        if result.get("synthesis"):
            logger.info("✅ Ollama comparison synthesis successful")
            return {
                "synthesis": str(result["synthesis"]),
                "best_for": str(result["best_for"]),
                "verdict": str(result["verdict"]),
                "feature_winners": result.get("feature_winners", [])
            }
    except Exception as e:
        logger.warning(f"Ollama unavailable for comparison ({e}), using dynamic mock comparison")

    # Generate custom mock comparison based on policy names
    p1 = policy_names[0] if len(policy_names) > 0 else "Policy A"
    p2 = policy_names[1] if len(policy_names) > 1 else "Policy B"
    p3 = f" and {policy_names[2]}" if len(policy_names) > 2 else ""
    
    feature_winners = [
        {
            "feature": "Premium Cost",
            "winner": p1,
            "reason": "Lower annual premium compared to others."
        },
        {
            "feature": "Coverage Limits",
            "winner": p2,
            "reason": "Higher sum insured and room rent limits."
        },
        {
            "feature": "Deductibles & Co-payments",
            "winner": p2 if len(policy_names) > 1 else p1,
            "reason": "Zero co-payment requirement on claims."
        },
        {
            "feature": "Network Size",
            "winner": p1,
            "reason": "Slightly larger network of partner hospitals."
        },
        {
            "feature": "Waiting Periods",
            "winner": p2 if len(policy_names) > 1 else p1,
            "reason": "Shorter waiting period for pre-existing conditions."
        }
    ]
    
    return {
        "synthesis": f"Comparing {p1}, {p2}{p3} reveals distinct differences in coverage limits, cost sharing, and waiting restrictions. {p1} tends to offer standard benefits with lower premiums but contains more co-payment limitations. In contrast, {p2} provides more comprehensive protection and lower out-of-pocket costs, though it comes at a higher annual cost.",
        "best_for": f"• {p1}: Best for budget-conscious individuals who have low regular healthcare expenditures.\n• {p2}: Best for families or individuals seeking maximum protection with lower risk of unexpected hospital bills.\n" + (f"• {policy_names[2]}: Best as a middle-ground plan balancing premiums and outpatient coverage." if len(policy_names) > 2 else ""),
        "verdict": f"For most users, {p2} offers the best overall security if the premium is within budget, as it avoids high co-payments during major health crises. If affordability is the main concern, {p1} is a reliable starter plan.",
        "feature_winners": feature_winners
    }


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def score_chunk(chunk: str, query_words: list[str]) -> float:
    score = 0.0
    chunk_lower = chunk.lower()
    for word in query_words:
        if len(word) < 3:
            continue
        score += chunk_lower.count(word) * 1.5
    return score


async def query_policy_rag(policies: list[dict], query: str, history: list[dict] = None) -> str:
    """Answer user questions about policies using a local RAG pipeline with Ollama."""
    query_words = [w.lower() for w in re.findall(r"\w+", query)]
    all_scored_chunks = []
    
    for p in policies:
        doc_name = p.get("filename", "Policy")
        doc_text = p.get("text", "")
        if not doc_text:
            continue
        chunks = chunk_text(doc_text)
        for c in chunks:
            score = score_chunk(c, query_words)
            if score > 0:
                all_scored_chunks.append({
                    "text": c,
                    "source": doc_name,
                    "score": score
                })
                
    # Sort and take top 4 chunks
    all_scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = all_scored_chunks[:4]
    
    # Format context
    context_parts = []
    for c in top_chunks:
        context_parts.append(f"Source: {c['source']}\nContent: {c['text']}")
    
    context = "\n---\n".join(context_parts) if context_parts else "No specific policy text matches your query terms."
    
    # Format history
    history_str = ""
    if history:
        for msg in history[-5:]:  # Last 5 messages
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
            
    prompt = RAG_PROMPT.format(
        context=context,
        query=query,
        chat_history=history_str
    )
    
    try:
        response = await call_ollama(prompt)
        if response:
            return response.strip()
    except Exception as e:
        logger.warning(f"Ollama RAG failed: {e}")
        
    # Offline fallback
    if top_chunks:
        best_match = top_chunks[0]
        return (
            f"[Offline Mode] Ollama is currently offline. Based on a search of your policies, "
            f"here is the most relevant section found in **{best_match['source']}**:\n\n"
            f"\"{best_match['text'].strip()}...\""
        )
    return "Ollama is currently offline and I couldn't find any matching terms in the policies to assist you."


async def translate_text(text: str, target_language: str) -> str:
    """Translate text using Ollama."""
    try:
        response = await call_ollama(TRANSLATE_PROMPT.format(text=text, target_language=target_language))
        if response:
            return response.strip()
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
    return text  # Return original if translation fails


async def generate_claims_checklist(policy_name: str, fields_summary: str, treatment_type: str) -> dict:
    """Generate dynamic claim checklist using Ollama."""
    prompt = CLAIMS_CHECKLIST_PROMPT.format(
        policy_name=policy_name,
        fields_summary=fields_summary,
        treatment_type=treatment_type
    )
    try:
        response = await call_ollama(prompt)
        result = extract_json_from_response(response)
        if result.get("checklist"):
            return result
    except Exception as e:
        logger.warning(f"Checklist generation failed: {e}")
        
    # Fallback checklist
    return {
        "checklist": [
          {"document_name": "Original Discharge Summary", "importance": "mandatory", "description": "Required to confirm hospital stay details"},
          {"document_name": "Final Consolidated Bill", "importance": "mandatory", "description": "Required for financial settlement"},
          {"document_name": "Pharmacy Prescriptions & Receipts", "importance": "mandatory", "description": "Required to claim medical costs"},
          {"document_name": f"Diagnostic Reports for {treatment_type}", "importance": "mandatory", "description": "Required to clinically verify diagnosis"},
          {"document_name": "Claim Form Part A & B", "importance": "mandatory", "description": "Signed insurance forms"}
        ],
        "claim_steps": [
          "Step 1: Check if the hospital is in the cashless network list.",
          "Step 2: If cashless, submit pre-authorization request within 24 hours of admission.",
          "Step 3: If reimbursement, notify the insurer within 48 hours of admission and submit all original documents within 15 days of discharge."
        ],
        "estimated_approval_days": "5-7 business days"
    }


