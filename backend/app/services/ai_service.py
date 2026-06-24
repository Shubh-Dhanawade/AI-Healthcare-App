"""
AI Service — Ollama Integration with intelligent mock fallback.
When Ollama is unavailable, returns realistic demo data for testing.
"""

import json
import re
import httpx
import asyncio
import hashlib
from loguru import logger
from typing import Optional

from app.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession

# ─────────────────────────────────────────
# In-memory result cache (keyed by doc text hash + task)
# Clears on server restart — keeps memory safe, avoids repeated LLM calls
# ─────────────────────────────────────────
_ai_cache: dict[str, dict] = {}
_rag_cache: dict[str, str] = {}  # Cache for RAG query responses

def _cache_key(task: str, text: str) -> str:
    """Create a stable cache key from task name + first 4000 chars of text."""
    digest = hashlib.md5(text[:4000].encode()).hexdigest()
    return f"{task}:{digest}"

def _rag_cache_key(query: str, policy_ids: list) -> str:
    """Cache key for RAG responses — keyed by query + sorted policy IDs."""
    id_str = ",".join(sorted(str(i) for i in policy_ids))
    digest = hashlib.md5(f"{query}:{id_str}".encode()).hexdigest()
    return f"rag:{digest}"


# ─────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────

# ── Concise prompts — fewer input tokens = faster model processing ──

SUMMARIZATION_PROMPT = """You are a healthcare insurance expert. Analyze the following insurance document and provide a BRIEF summary.
Return ONLY a valid JSON object matching the schema below. Do not output any preamble, explanation, or conversational text.
Be concise and direct.

Return format JSON:
{{
  "summary_text": "Executive summary of the policy (maximum 80 words)",
  "coverage_summary": "Summary of major coverages and benefits (maximum 50 words)",
  "exclusions_summary": "Summary of key exclusions and what is not covered (maximum 50 words)",
  "waiting_period_summary": "Summary of waiting periods for pre-existing or standard diseases (maximum 50 words)",
  "premium_summary": "Summary of premium, deductibles, and co-payment details (maximum 50 words)"
}}

DOCUMENT:
{document_text}"""


FIELD_EXTRACTION_PROMPT = """Extract key fields from this health insurance document. Return ONLY valid JSON, no preamble. Use null if field is not found.

Return format JSON:
{{
  "policy_name": "policy plan name",
  "insurer_name": "insurance provider",
  "policy_number": "policy ID",
  "sum_insured": "coverage amount",
  "premium_amount": "premium cost",
  "deductible": "deductible",
  "co_payment": "co-payment",
  "waiting_period": "waiting period",
  "coverage_type": "plan type",
  "policy_term": "duration",
  "network_hospitals": "hospitals count",
  "pre_existing_coverage": "pre-existing terms",
  "maternity_coverage": "maternity benefits",
  "room_rent_limit": "room rent limit",
  "claim_process": "claim filing steps"
}}

DOCUMENT:
{document_text}"""


RISK_ANALYSIS_PROMPT = """Identify up to 3 risky clauses in this health insurance policy. Return ONLY valid JSON, no preamble.

Return format JSON:
{{
  "risks": [
    {{
      "clause_text": "exact clause from document",
      "risk_type": "waiting_period|exclusion|deductible|co_payment|coverage_limit",
      "severity": "low|medium|high",
      "explanation": "why it's risky (max 30 words)",
      "recommendation": "recommended action (max 30 words)"
    }}
  ],
  "overall_risk_level": "low|medium|high"
}}

DOCUMENT:
{document_text}"""


COMPARISON_PROMPT = """Compare policies. JSON:
{policies_data}
{{"synthesis":"<80w","best_for":"<40w","verdict":"<40w","feature_winners":[{{"feature":"","winner":"","reason":"<12w"}}]}}"""


# NOTE: RAG_PROMPT is now only used in rag_service.py (single-doc pipeline).
# The chat RAG (query_policy_rag / query_policy_rag_stream) builds prompts
# as plain strings to avoid Python .format() issues with policy text containing { }


TRANSLATE_PROMPT = """Translate to {target_language}:
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

async def warmup_model() -> None:
    """Send a tiny keep-alive prompt to Ollama so model weights stay resident in VRAM."""
    try:
        model = settings.OLLAMA_MODEL
        url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        if "localhost" in url:
            url = url.replace("localhost", "127.0.0.1")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "options": {"num_predict": 1, "num_ctx": 128, "temperature": 0},
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(url, json=payload)
        logger.info("Ollama model warmup complete — weights resident in memory")
    except Exception as e:
        logger.warning(f"Model warmup skipped: {e}")


async def call_ollama(
    prompt: str,
    model: Optional[str] = None,
    num_predict: int = 700,
    num_ctx: int = 1024,
) -> str:
    """Call Ollama API with aggressive speed-optimised options."""
    model = model or settings.OLLAMA_MODEL
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    if "localhost" in url:
        url = url.replace("localhost", "127.0.0.1")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0,          # Greedy decoding — zero sampling overhead
            "num_predict": num_predict, # Strict token limit per task
            "num_ctx": num_ctx,        # REDUCED context for speed — 1024 base instead of 2048
            "num_batch": 1024,         # Larger batches = faster prefill 
            "use_mmap": True,          # Memory-map for fast reloads
            "use_mlock": False,        # Disable model locking in RAM to avoid slow OS virtual memory allocation
            "repeat_penalty": 1.0,     # No sampling overhead
            "top_k": 1,                # Greedy only — fastest
            "top_p": 1.0,              # Disable nucleus sampling
        },
    }

    logger.info(f"Ollama: {model} predict={num_predict} ctx={num_ctx}")
    async with httpx.AsyncClient(timeout=180.0) as client:  # Increased timeout from 60s to 180s to prevent early aborts
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")


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

def _clean_field(val):
    if val is None:
        return None
    if isinstance(val, list):
        return "\n".join(str(item) for item in val)
    return str(val)


async def generate_summary(document_text: str) -> dict:
    """Generate AI summary. Cached per document hash, falls back to demo data."""
    ck = _cache_key("summary", document_text)
    if ck in _ai_cache:
        logger.info("Cache hit: summary")
        return _ai_cache[ck]

    # Trim to 2000 chars — enough context, MUCH faster prefill
    truncated = document_text[:2000] if len(document_text) > 2000 else document_text
    try:
        # 400 tokens max (reduced from 600) 
        response = await call_ollama(
            SUMMARIZATION_PROMPT.format(document_text=truncated),
            num_predict=600,
            num_ctx=2048,
        )
        result = extract_json_from_response(response)
        if result.get("summary_text"):
            logger.info("Ollama summarization successful")
            out = {
                "summary_text": _clean_field(result.get("summary_text", MOCK_SUMMARY["summary_text"])),
                "coverage_summary": _clean_field(result.get("coverage_summary")),
                "exclusions_summary": _clean_field(result.get("exclusions_summary")),
                "waiting_period_summary": _clean_field(result.get("waiting_period_summary")),
                "premium_summary": _clean_field(result.get("premium_summary")),
            }
            _ai_cache[ck] = out
            return out
    except Exception as e:
        logger.warning(f"Ollama unavailable ({e}), using demo summary data")
    return dict(MOCK_SUMMARY)


async def extract_policy_fields(document_text: str) -> list[dict]:
    """Extract key fields. Cached per document hash, falls back to demo data."""
    ck = _cache_key("fields", document_text)
    if ck in _ai_cache:
        logger.info("Cache hit: fields")
        return _ai_cache[ck]

    truncated = document_text[:2000] if len(document_text) > 2000 else document_text
    try:
        # Fields extraction is pure JSON — 200 tokens max (reduced from 400)
        response = await call_ollama(
            FIELD_EXTRACTION_PROMPT.format(document_text=truncated),
            num_predict=600,
            num_ctx=2048,
        )
        result = extract_json_from_response(response)
        if result:
            logger.info("Ollama field extraction successful")
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
            fields = [
                {
                    "field_name": key.replace("_", " ").title(),
                    "field_value": str(value),
                    "field_category": field_category_map.get(key, "general"),
                }
                for key, value in result.items()
                if value and value not in ("null", None)
            ]
            _ai_cache[ck] = fields
            return fields
    except Exception as e:
        logger.warning(f"Ollama unavailable ({e}), using demo field data")
    return list(MOCK_FIELDS)


async def analyze_risks(document_text: str) -> dict:
    """Detect risky clauses. Cached per document hash, falls back to demo data."""
    ck = _cache_key("risks", document_text)
    if ck in _ai_cache:
        logger.info("Cache hit: risks")
        return _ai_cache[ck]

    truncated = document_text[:2000] if len(document_text) > 2000 else document_text
    try:
        # Risk JSON is compact — 250 tokens max (reduced from 350)
        response = await call_ollama(
            RISK_ANALYSIS_PROMPT.format(document_text=truncated),
            num_predict=600,
            num_ctx=2048,
        )
        result = extract_json_from_response(response)
        if result.get("risks"):
            logger.info("Ollama risk analysis successful")
            out = {
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
            _ai_cache[ck] = out
            return out
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
        response = await call_ollama(
            COMPARISON_PROMPT.format(policies_data=policies_text),
            num_predict=300,  # REDUCED from 700
            num_ctx=1024  # REDUCED from 2048
        )
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


# ─────────────────────────────────────────
# Comparison Intent Detection
# ─────────────────────────────────────────

_COMPARE_KEYWORDS = {
    "compare", "comparison", "vs", "versus", "better", "best",
    "difference", "differ", "which", "recommend", "recommendation",
    "all policies", "both", "most benefit", "more benefit", "benefits in all",
    "which policy", "should i choose", "more coverage",
}

def _is_comparison_query(query: str) -> bool:
    """Return True if query is asking to compare across policies."""
    q = query.lower()
    return any(kw in q for kw in _COMPARE_KEYWORDS)


def _build_context_and_prompt(
    top_chunks: list[dict],
    policies: list[dict],
    query: str,
    history_str: str,
    user_name: str,
    is_comparison: bool,
) -> str:
    """
    Build a clean prompt string for Ollama.
    """
    if is_comparison:
        # Group chunks by source policy
        by_source: dict[str, list[str]] = {}
        for c in top_chunks:
            src = c["source"]
            by_source.setdefault(src, []).append(c["text"][:400])

        context_lines = []
        for src, texts in by_source.items():
            context_lines.append(f"=== {src} ===")
            for t in texts:
                context_lines.append(t)
            context_lines.append("")
        context_block = "\n".join(context_lines) if context_lines else "No policy text found."

        policy_names = [p.get("filename", "Policy") for p in policies]
        names_str = ", ".join(policy_names)

        prompt = (
            f"You are HealthAI, an expert healthcare insurance advisor helping {user_name}.\n"
            f"You are comparing the following insurance policies: {names_str}.\n"
            "\n"
            "Instructions:\n"
            "1. Compare the policies directly based on the provided POLICY CONTEXT and PREVIOUS CONVERSATION.\n"
            "2. Clearly specify which details belong to which policy by name.\n"
            "3. Use a markdown comparison table or bullet lists grouped by policy name for readability.\n"
            "4. Highlight differences in key terms (deductibles, co-pays, waiting periods, room rent caps).\n"
            "5. Maintain a professional tone and end with a concise recommendation.\n"
            "6. Do NOT include any 'ASSISTANT:', 'USER:', or 'context:' labels in your response.\n"
            "\n"
            f"POLICY CONTEXT:\n{context_block}\n"
            f"PREVIOUS CONVERSATION:\n{history_str}\n"
            f"User Query: {query}\n"
            "\n"
            "Comparison Response:"
        )
    else:
        context_lines = []
        for c in top_chunks:
            context_lines.append(f"[{c['source']}]\n{c['text'][:450]}")
        context_block = "\n---\n".join(context_lines) if context_lines else "No relevant policy text found."

        prompt = (
            f"You are HealthAI, a knowledgeable and friendly healthcare insurance assistant helping {user_name}.\n"
            "\n"
            "Instructions:\n"
            "1. Answer the user's query clearly and concisely using the provided POLICY CONTEXT and PREVIOUS CONVERSATION.\n"
            "2. If the user is asking for clarification, explanation of terms, or a follow-up question on previous responses, use the PREVIOUS CONVERSATION and general insurance knowledge to answer directly and politely.\n"
            "3. When referencing specific policy facts, always mention the source document name (e.g., 'In Star_Health.pdf...').\n"
            "4. If the query asks for policy details that are not in the context, and cannot be inferred from history, state: 'I could not find this specific information in the selected policies.'\n"
            "5. Do NOT output 'ASSISTANT:', 'USER:', or 'context:' labels in your response.\n"
            "6. Never output curly braces in your answer.\n"
            "\n"
            f"POLICY CONTEXT:\n{context_block}\n"
            f"PREVIOUS CONVERSATION:\n{history_str}\n"
            f"User Query: {query}\n"
            "\n"
            "Response:"
        )
    return prompt


async def query_policy_rag(
    policies: list[dict],
    query: str,
    db: AsyncSession = None,
    history: list[dict] = None,
    user_name: str = "there",
    user_id: str = None,
) -> str:
    """Answer user questions about policies using a local RAG pipeline with Ollama."""
    from app.services.chat_service import run_chat_query
    return await run_chat_query(
        policies=policies,
        query=query,
        db=db,
        history=history,
        user_name=user_name,
        user_id=user_id
    )


async def translate_text(text: str, target_language: str) -> str:
    """Translate text using Ollama — optimized for speed."""
    try:
        # Limit translation to 300 tokens — usually enough for policy snippets
        response = await call_ollama(
            TRANSLATE_PROMPT.format(text=text[:1000], target_language=target_language),  # REDUCED input
            num_predict=300,
            num_ctx=512  # MINIMAL context for translation
        )
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


async def call_ollama_stream(prompt: str, model: Optional[str] = None, num_predict: int = 200):
    """Generate streaming tokens from Ollama — optimized for speed."""
    model = model or settings.OLLAMA_MODEL
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    
    if "localhost" in url:
        url = url.replace("localhost", "127.0.0.1")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": True,
        "options": {
            "temperature": 0,  # CHANGED from 0.1 to 0 for greedy decoding
            "num_predict": num_predict,  # REDUCED from 512
            "num_ctx": 1024,  # REDUCED from 4096
            "num_batch": 1024,
            "top_k": 1,  # GREEDY only
            "top_p": 1.0,
        },
    }

    logger.info(f"Ollama stream: {model} predict={num_predict}")
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_lines():
                    if chunk:
                        try:
                            data = json.loads(chunk)
                            yield data.get("message", {}).get("content", "")
                            if data.get("done", False):
                                break
                        except Exception as e:
                            logger.error(f"Error parsing streaming chunk: {e}")
    except Exception as e:
        logger.error(f"Failed to communicate with Ollama stream: {e}")
        raise


# ─────────────────────────────────────────
# Greeting / Conversational Intent Detection
# ─────────────────────────────────────────

_GREETING_PATTERNS = {
    "hi", "hii", "hiii", "hey", "hello", "heya", "hola", "howdy",
    "good morning", "good afternoon", "good evening", "good night",
    "yo", "greetings", "how are you", "what's up", "sup",
}

_GUIDANCE_PATTERNS = {
    "help", "what can i do", "what can you do", "features", "how to use",
    "how does this work", "what is this app", "what is this chatbot",
    "capabilities", "instructions", "guidelines", "menu", "commands"
}

_THANKS_PATTERNS = {
    "thanks", "thank you", "ty", "thx", "appreciate it", "great", "perfect",
    "ok", "okay", "bye", "goodbye", "see ya", "talk to you later"
}

def _is_greeting(query: str) -> bool:
    """Fallback check for greeting or general chitchat."""
    q = query.strip().lower().rstrip("!?.")
    if q in _GREETING_PATTERNS or q in _GUIDANCE_PATTERNS or q in _THANKS_PATTERNS:
        return True
    words = q.split()
    if len(words) <= 2 and (q in _GREETING_PATTERNS or q in _THANKS_PATTERNS):
        return True
    return False

def _get_chitchat_response(query: str, user_name: str) -> Optional[str]:
    """Identify if a query is a general chitchat or guidance request and return a standard response."""
    q = query.strip().lower().rstrip("!?.")
    
    # Check words and phrase matches
    is_greeting = q in _GREETING_PATTERNS or any(pat in q for pat in ["hello", "how are you", "what's up", "greetings"]) and len(q.split()) <= 3
    
    # Substring lists for guidance (about app features or assistant role)
    guidance_substrings = [
        "what can i do", "what can you do", "how to use", "how does this work", 
        "what is this app", "what is this application", "what is this chatbot",
        "who are you", "what do you do", "what is your purpose", "features of this",
        "capabilities", "instructions", "guidelines", "about you"
    ]
    is_guidance = (
        q in _GUIDANCE_PATTERNS or 
        any(pat in q for pat in guidance_substrings) or 
        (any(pat in q for pat in ["help", "info"]) and len(q.split()) <= 2)
    )
    
    is_thanks = q in _THANKS_PATTERNS or any(pat in q for pat in ["thank you", "thanks", "goodbye", "bye"]) and len(q.split()) <= 3
    
    if is_greeting:
        return (
            f"Hi {user_name}! 👋 I'm **HealthAI**, your healthcare insurance assistant. "
            "How can I help you today? You can ask me questions about your uploaded insurance policies — "
            "such as coverage limits, premiums, exclusions, co-pays, and waiting periods!"
        )
        
    if is_guidance:
        return (
            f"Hi {user_name}! I can help you analyze and understand your healthcare insurance policies. "
            "Here are some things you can ask me to do:\n\n"
            "* **Analyze Coverage**: Ask questions like *\"Is maternity covered?\"* or *\"What is my room rent limit?\"*\n"
            "* **Check Exclusions & Waiting Periods**: Ask *\"Are pre-existing conditions covered?\"* or *\"What is excluded?\"*\n"
            "* **Review Financial Terms**: Ask *\"What is the deductible?\"* or *\"What is the co-payment percentage?\"*\n"
            "* **Compare Policies**: Select multiple policies and ask *\"Compare these policies\"* or *\"Which policy is better for maternity?\"*\n"
            "* **Generate Claims Checklist**: Ask *\"What documents do I need for a heart surgery claim?\"*\n\n"
            "To get started, please make sure you've uploaded policy documents and selected them in the dropdown above!"
        )
        
    if is_thanks:
        if any(w in q for w in ["bye", "goodbye", "see ya"]):
            return f"Goodbye! Have a great day ahead. Let me know if you need help with your insurance policies in the future! 😊"
        return f"You're very welcome! If you have any other questions about your policies, feel free to ask. I'm here to help! 🏥"
        
    return None

def _needs_query_rewriting(query: str, history: list[dict]) -> bool:
    """Return True if the query is a follow-up that likely needs context from history."""
    if not history:
        return False
    
    q = query.strip().lower()
    
    # Pronoun / referential indicators
    referential_terms = {
        "it", "they", "them", "this", "that", "these", "those", "its", "their",
        "first", "second", "third", "former", "latter", "previous", "above",
        "other", "another", "same", "both", "difference", "compare", "which"
    }
    
    words = re.findall(r'\w+', q)
    if any(w in referential_terms for w in words):
        return True
        
    # Short question words or clarifications
    clarification_starts = (
        "why", "how", "explain", "elaborate", "tell me more", "what about", 
        "is there", "does it", "can you", "any other", "what is the"
    )
    if q.startswith(clarification_starts) and len(words) < 6:
        return True
        
    if len(words) < 4:
        return True
        
    return False

async def rewrite_query_with_history(query: str, history: list[dict]) -> str:
    """Rewrite a conversational follow-up query into a standalone search query using history."""
    if not history:
        return query
    
    # We only format the last 2-3 turns of history to keep it brief and fast
    history_context = ""
    for msg in history[-3:]:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")[:150]
        history_context += f"{role}: {content}\n"
    
    prompt = (
        "You are a search query optimizer. Given the following conversation history and a follow-up query, "
        "rewrite the follow-up query to be a standalone, self-contained search query. "
        "The standalone query should contain all specific names, terms, or policies referred to (e.g., replace 'it' or 'the first policy' with the actual names from the history).\n"
        "Do not answer the query. Output ONLY the rewritten search query text, without quotes, explanations, or preamble.\n\n"
        f"CONVERSATION HISTORY:\n{history_context}\n"
        f"FOLLOW-UP QUERY: {query}\n\n"
        "Standalone Query:"
    )
    try:
        rewritten = await call_ollama(prompt, num_predict=60, num_ctx=512)
        rewritten_clean = rewritten.strip().strip('"\'')
        if rewritten_clean:
            logger.info(f"Rewrote query: '{query}' -> '{rewritten_clean}'")
            return rewritten_clean
    except Exception as e:
        logger.warning(f"Failed to rewrite query: {e}")
    return query


async def query_policy_rag_stream(
    policies: list[dict],
    query: str,
    db: AsyncSession = None,
    history: list[dict] = None,
    user_name: str = "there",
    user_id: str = None,
):
    """Answer user questions about policies using a local RAG pipeline with streaming Ollama."""
    from app.services.chat_service import run_chat_query_stream
    async for token in run_chat_query_stream(
        policies=policies,
        query=query,
        db=db,
        history=history,
        user_name=user_name,
        user_id=user_id
    ):
        yield token



