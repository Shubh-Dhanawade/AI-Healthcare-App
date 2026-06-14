"""
AI Service — Ollama Integration with intelligent mock fallback.
When Ollama is unavailable, returns realistic demo data for testing.
"""

import json
import re
import httpx
import asyncio
from loguru import logger
from typing import Optional

from app.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession


# ─────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────

SUMMARIZATION_PROMPT = """You are a healthcare insurance expert. Analyze this insurance policy and provide a JSON summary that explains advanced details in simple, user-friendly plain language.
If you encounter complex terminology (like deductibles, copays, sublimits, or waiting periods), explain briefly what they mean in practice for the user's out-of-pocket costs.

DOCUMENT TEXT:
{document_text}

Return ONLY valid JSON with these exact keys:
{{
  "summary_text": "A comprehensive 1-paragraph summary explaining the policy's purpose, target audience, and overall value in plain language (maximum 150 words)",
  "coverage_summary": "List top 3 key coverages. Include details of what is covered and explain any relevant sub-limits simply (bullet points, max 80 words total)",
  "exclusions_summary": "List top 3 critical exclusions. Explain what is NOT covered so a non-expert can understand the scope (bullet points, max 80 words total)",
  "waiting_period_summary": "List key waiting periods (initial, pre-existing diseases, specific treatments) and explain how they delay benefits (bullet points, max 80 words total)",
  "premium_summary": "List the premium, deductible, and co-payment details, explaining how these cost-sharing terms affect the user's wallet (bullet points, max 80 words total)"
}}"""


FIELD_EXTRACTION_PROMPT = """Extract key insurance policy fields from this document. Return ONLY valid JSON.
If a field has specific conditions, limits, or sub-limits, include them in the value so the user gets complete context (keep under 15 words per value). If a field is not mentioned, use null.

DOCUMENT TEXT:
{document_text}

Return JSON:
{{
  "policy_name": "Name of the insurance policy (e.g. 'Individual - Gold Plan')",
  "insurer_name": "Name of the insurance company (e.g. 'LifeGuard Health Insurance Co.')",
  "policy_number": "Policy number (e.g. 'LG-2025-IND-00742')",
  "sum_insured": "Maximum coverage amount (e.g. '10,00,000')",
  "premium_amount": "Total premium payable, including taxes if known (e.g. '14,691/year')",
  "deductible": "Deductible amount (what you pay before insurance starts), use null if none",
  "co_payment": "Co-payment percentage or amount (your share of every claim), use null if none",
  "waiting_period": "Initial waiting period before illnesses are covered (e.g. '30 days except accidents')",
  "coverage_type": "Type of coverage (e.g. 'Individual' or 'Family Floater')",
  "policy_term": "Duration of the policy (e.g. '1 year')",
  "network_hospitals": "Network hospitals listed (brief comma-separated list or count)",
  "pre_existing_coverage": "Pre-existing disease coverage waiting period (e.g. '48 months')",
  "maternity_coverage": "Maternity benefit details and waiting periods, use null if none",
  "room_rent_limit": "Room rent sub-limit (e.g. '1% of Sum Insured per day; proportionate deduction applies')",
  "claim_process": "Brief instructions on how to file cashless or reimbursement claims"
}}"""


RISK_ANALYSIS_PROMPT = """Analyze this insurance policy for risky, hidden, or unfavorable clauses. Identify up to 3 key risk areas. Return ONLY valid JSON.
For each risk, provide a clear explanation of how it affects the user's out-of-pocket costs and give a highly actionable recommendation.

DOCUMENT TEXT:
{document_text}

Return:
{{
  "risks": [
    {{
      "clause_text": "Exact short problematic clause text from document",
      "risk_type": "waiting_period|exclusion|deductible|co_payment|hidden_condition|coverage_limit",
      "severity": "low|medium|high",
      "explanation": "Clear explanation of why this clause is risky, explaining the exact real-world financial impact (max 40 words)",
      "recommendation": "Actionable advice on how to mitigate this risk, negotiate terms, or plan financially (max 40 words)"
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


TRANSLATE_PROMPT = """You are an expert translator. Translate the English text into {target_language}.

Rules:
1. Translate to {target_language} language.
2. Use the correct script for {target_language}. For example, if target language is Marathi, use Devanagari script (like 'पॉलिसीचे नियम व अटी'). If Hindi, use Devanagari script (like 'पॉलिसी के नियम और शर्तें').
3. Do NOT mix or use letters from other scripts like Gujarati or Bengali.
4. Output ONLY the translation. Do NOT include any explanations, introductory text, or formatting.

Examples for Marathi (मराठी):
English: "This policy covers hospital room rent up to 1% of sum insured."
Marathi: "या पॉलिसीमध्ये विमा रक्कमेच्या १% पर्यंत रुग्णालयाच्या खोलीच्या भाड्याचा समावेश आहे."

English: "Pre-existing diseases are covered after a waiting period of 3 years."
Marathi: "३ वर्षांच्या प्रतीक्षा कालावधीनंतर आधीपासून असलेले आजार कव्हर केले जातात."

Examples for Hindi (हिंदी):
English: "This policy covers hospital room rent up to 1% of sum insured."
Hindi: "यह पॉलिसी बीमा राशि के १% तक अस्पताल के कमरे के किराए को कवर करती है।"

TEXT TO TRANSLATE:
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
        "This is the Star Health Comprehensive Plan, a premium family-floater policy designed for comprehensive "
        "medical coverage. It covers standard hospitalization, daycare treatments, and advanced procedures up to "
        "a ₹5,00,000 annual limit. To make it affordable, the policy incorporates cost-sharing features like a "
        "₹5,00,000 sum insured, a ₹5,000 initial deductible (the amount you pay before insurance pays anything), "
        "and a 20% co-payment (your share of every hospital bill). Room rent is capped at 1% of the sum insured "
        "per day (₹5,000/day), meaning staying in a room that costs more will trigger a 'proportionate deduction' "
        "where the insurer reduces payouts for doctor fees and surgeries as well."
    ),
    "coverage_summary": (
        "• Hospitalization & ICU: Complete coverage up to the ₹5,00,000 sum insured.\n"
        "• Daycare Procedures: All daycare treatments (requiring less than 24h stay) are fully covered.\n"
        "• Road Ambulance: Covered up to ₹2,000 per hospitalization event to handle emergencies.\n"
        "• No-Claim Bonus: 10% increase in sum insured for every claim-free year (up to 50% maximum) as an incentive."
    ),
    "exclusions_summary": (
        "• Cosmetic & Obesity Treatments: Surgeries for aesthetic enhancement or weight loss are excluded.\n"
        "• Outpatient Dental & Vision: Routine dental work and spectacles are not covered unless due to an accident.\n"
        "• Hazardous Sports: Treatment for injuries from extreme adventure sports or self-harm is completely excluded."
    ),
    "waiting_period_summary": (
        "• Initial 30-Day Delay: No illness coverage for the first 30 days, except for emergency accident claims.\n"
        "• Specific Treatments (24 Months): Common procedures (cataracts, hernia, joint replacements) require 2 years.\n"
        "• Pre-existing Conditions (48 Months): Long-term conditions (diabetes, high blood pressure) are covered only after 4 years."
    ),
    "premium_summary": (
        "• Base Premium: ₹12,500 annually for a typical family plan (plus 18% GST).\n"
        "• Deductible: You pay the first ₹5,000 deductible per hospital admission before insurance coverage applies.\n"
        "• Co-Payment (20%): You must pay 20% of every approved claim value, while the insurer pays the remaining 80%."
    ),
}

MOCK_FIELDS = [
    {"field_name": "Policy Name", "field_value": "Star Health Comprehensive Plan", "field_category": "policy_info"},
    {"field_name": "Insurer Name", "field_value": "Star Health and Allied Insurance Co. Ltd.", "field_category": "policy_info"},
    {"field_name": "Policy Number", "field_value": "P/211111/01/2024/000001", "field_category": "policy_info"},
    {"field_name": "Sum Insured", "field_value": "₹5,00,000 (Maximum annual coverage limit)", "field_category": "coverage"},
    {"field_name": "Premium Amount", "field_value": "₹12,500 per annum (excludes 18% GST)", "field_category": "premium"},
    {"field_name": "Deductible", "field_value": "₹5,00,000 (The amount you pay first for each hospitalization)", "field_category": "premium"},
    {"field_name": "Co Payment", "field_value": "20% (Your share of the total bill for every claim)", "field_category": "premium"},
    {"field_name": "Waiting Period", "field_value": "30 days initial; 48 months for pre-existing diseases", "field_category": "restrictions"},
    {"field_name": "Coverage Type", "field_value": "Family Floater (Covers all listed family members)", "field_category": "coverage"},
    {"field_name": "Policy Term", "field_value": "1 year (Requires annual renewal to stay active)", "field_category": "policy_info"},
    {"field_name": "Network Hospitals", "field_value": "5,000+ partner hospitals providing cashless claims", "field_category": "coverage"},
    {"field_name": "Pre Existing Coverage", "field_value": "Covered only after a 48-month continuous waiting period", "field_category": "coverage"},
    {"field_name": "Maternity Coverage", "field_value": "Up to ₹25,000 limit after a 24-month waiting period", "field_category": "coverage"},
    {"field_name": "Room Rent Limit", "field_value": "1% of Sum Insured per day (₹5,000/day); proportionate deductions apply", "field_category": "restrictions"},
    {"field_name": "Claim Process", "field_value": "Cashless at network hospitals; reimbursement documents must be sent within 30 days", "field_category": "process"},
]

MOCK_RISKS = [
    {
        "clause_text": "All pre-existing diseases shall not be covered during the first 48 months of the policy.",
        "risk_type": "waiting_period",
        "severity": "high",
        "explanation": "A 48-month (4-year) waiting period means you pay premiums for 4 full years before receiving any coverage for chronic illnesses like diabetes or hypertension. This is an exceptionally long delay.",
        "recommendation": "If you have pre-existing illnesses, consider policies with shorter waiting periods (12 to 24 months) or look into paying a slightly higher premium for a waiver rider.",
    },
    {
        "clause_text": "Co-payment of 20% shall be applicable for each and every claim under this policy.",
        "risk_type": "co_payment",
        "severity": "high",
        "explanation": "A 20% co-pay requires you to pay 20% of every hospital bill out of pocket. For a large claim of ₹5 Lakhs, you must pay ₹1 Lakh yourself. This represents a heavy unexpected financial burden.",
        "recommendation": "Check if you can buy a 'Co-payment Waiver Rider' to remove this clause. Compare other plans without co-payments, as the higher upfront premium is often cheaper than one hospital bill.",
    },
    {
        "clause_text": "Room rent shall be limited to 1% of the Sum Insured per day. If room rent exceeds this limit, proportionate deduction shall apply.",
        "risk_type": "coverage_limit",
        "severity": "medium",
        "explanation": "At a ₹5 Lakh sum insured, your room cap is ₹5,000/day. If you stay in a standard room costing ₹8,000/day, you don't just pay the ₹3,000 difference—all your surgeon fees, ICU, and medicine charges will be reduced by 37.5% proportionately.",
        "recommendation": "Select a plan with no room rent sub-limit, or ensure you strictly stay in a room that costs less than ₹5,000/day during hospitalization to avoid massive out-of-pocket charges.",
    },
    {
        "clause_text": "Deductible of ₹5,000 applicable per hospitalization event.",
        "risk_type": "deductible",
        "severity": "medium",
        "explanation": "A ₹5,000 deductible means the insurance company will deduct ₹5,000 from your approved claim amount for every single hospital stay. You are fully responsible for this initial sum.",
        "recommendation": "Maintain a dedicated health emergency fund of at least ₹15,000 to cover these per-incident deductibles without affecting your main savings.",
    },
    {
        "clause_text": "Cosmetic or aesthetic treatments, dental procedures (except accidental), and vision correction are excluded.",
        "risk_type": "exclusion",
        "severity": "low",
        "explanation": "Standard exclusions common to most health insurance. Cosmetic treatments, routine dental work, and vision aids are not covered.",
        "recommendation": "These are industry-standard exclusions, so no action is needed. Just be sure to budget for dental and eye care separately outside of insurance.",
    },
]


# ─────────────────────────────────────────
# Ollama Client
# ─────────────────────────────────────────

async def call_ollama(prompt: str, model: Optional[str] = None, num_predict: int = 1024) -> str:
    """Call Ollama API. Raises httpx.ConnectError if unavailable."""
    model = model or settings.OLLAMA_MODEL
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": num_predict,
            "num_ctx": 4096,
        },
    }

    logger.info(f"Calling Ollama model={model} (num_predict={num_predict})")
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


async def query_policy_rag(policies: list[dict], query: str, db: AsyncSession = None, history: list[dict] = None) -> str:
    """Answer user questions about policies using a local RAG pipeline with Ollama."""
    all_scored_chunks = []
    top_chunks = []
    use_vector_search = False
    
    if db:
        from app.models.document import DocumentChunk
        from app.services.rag_service import generate_embeddings, cosine_similarity
        from sqlalchemy import select
        
        policy_ids = [p["id"] for p in policies if p.get("id")]
        
        if policy_ids:
            res = await db.execute(select(DocumentChunk).where(DocumentChunk.document_id.in_(policy_ids)))
            db_chunks = res.scalars().all()
            
            if db_chunks:
                use_vector_search = True
                query_emb = await generate_embeddings(query)
                
                for chunk in db_chunks:
                    try:
                        p_dict = next((p for p in policies if p.get("id") == chunk.document_id), {})
                        doc_name = p_dict.get("filename", "Policy")
                        
                        chunk_emb = json.loads(chunk.embedding)
                        sim = cosine_similarity(query_emb, chunk_emb)
                        
                        all_scored_chunks.append({
                            "text": chunk.text_content,
                            "source": doc_name,
                            "score": sim
                        })
                    except Exception as e:
                        logger.error(f"Failed to process chunk vector in query_policy_rag: {e}")
                        
                all_scored_chunks.sort(key=lambda x: x["score"], reverse=True)
                top_chunks = all_scored_chunks[:4]
                
    if not use_vector_search:
        logger.warning("No database vector chunks available for cross-document RAG. Falling back to keyword search.")
        query_words = [w.lower() for w in re.findall(r"\w+", query)]
        
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
        response = await call_ollama(prompt, num_predict=512)
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


async def call_ollama_stream(prompt: str, model: Optional[str] = None, num_predict: int = 512):
    """Generate streaming tokens from Ollama."""
    model = model or settings.OLLAMA_MODEL
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    
    if "localhost" in url:
        url = url.replace("localhost", "127.0.0.1")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_predict": num_predict,
            "num_ctx": 4096,
        },
    }

    logger.info(f"Calling Ollama stream model={model} (num_predict={num_predict})")
    try:
        async with httpx.AsyncClient(timeout=240.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_lines():
                    if chunk:
                        try:
                            data = json.loads(chunk)
                            yield data.get("response", "")
                            if data.get("done", False):
                                break
                        except Exception as e:
                            logger.error(f"Error parsing streaming chunk: {e}")
    except Exception as e:
        logger.error(f"Failed to communicate with Ollama stream: {e}")
        raise


async def query_policy_rag_stream(policies: list[dict], query: str, db: AsyncSession = None, history: list[dict] = None, user_name: str = "krushna"):
    """Answer user questions about policies using a local RAG pipeline with streaming Ollama."""
    all_scored_chunks = []
    top_chunks = []
    use_vector_search = False
    
    if db:
        from app.models.document import DocumentChunk
        from app.services.rag_service import generate_embeddings, cosine_similarity
        from sqlalchemy import select
        
        policy_ids = [p["id"] for p in policies if p.get("id")]
        
        if policy_ids:
            res = await db.execute(select(DocumentChunk).where(DocumentChunk.document_id.in_(policy_ids)))
            db_chunks = res.scalars().all()
            
            if db_chunks:
                use_vector_search = True
                query_emb = await generate_embeddings(query)
                
                for chunk in db_chunks:
                    try:
                        p_dict = next((p for p in policies if p.get("id") == chunk.document_id), {})
                        doc_name = p_dict.get("filename", "Policy")
                        
                        chunk_emb = json.loads(chunk.embedding)
                        sim = cosine_similarity(query_emb, chunk_emb)
                        
                        all_scored_chunks.append({
                            "text": chunk.text_content,
                            "source": doc_name,
                            "score": sim
                        })
                    except Exception as e:
                        logger.error(f"Failed to process chunk vector in query_policy_rag_stream: {e}")
                        
                all_scored_chunks.sort(key=lambda x: x["score"], reverse=True)
                top_chunks = all_scored_chunks[:4]
                
    if not use_vector_search:
        logger.warning("No database vector chunks available for cross-document RAG. Falling back to keyword search.")
        query_words = [w.lower() for w in re.findall(r"\w+", query)]
        
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
            
    # Format prompt using the updated system RAG prompt
    prompt = f"""You are a friendly and professional healthcare insurance assistant. 
The current user you are talking to is: {user_name}.

Guidelines:
1. Primary Source: Use the provided context from the insurance policies to answer questions about coverages, limits, exclusions, and claims. Ground your answers in these documents.
2. Personal Info: If the user asks about themselves (e.g., their name), address them as {user_name}.
3. General Advice / Greetings: If the user asks general questions (e.g., greetings, general health advice, general insurance definitions), answer them politely using your general knowledge, but clarify that this is general advice and not specified in their uploaded policy documents.
4. Missing Policy Facts: If the user asks a specific question about their policy coverages that is NOT in the context, politely state: "I cannot find this information in the selected policies."

CONTEXT FROM INSURANCE POLICIES:
{context}

USER QUERY:
{query}

CHAT HISTORY:
{history_str}

Please provide a structured, friendly response. If quoting policy terms, specify the source document name."""

    try:
        async for token in call_ollama_stream(prompt, num_predict=512):
            yield token
    except Exception as e:
        logger.warning(f"Ollama RAG stream failed: {e}")
        fallback_msg = "Ollama is currently offline. Based on the matches:\n"
        if top_chunks:
            best = top_chunks[0]
            fallback_msg += f"Most relevant source: {best['source']}\n\"{best['text'][:300]}...\""
        else:
            fallback_msg += "No matching information found."
        for word in fallback_msg.split(" "):
            yield word + " "
            await asyncio.sleep(0.05)


