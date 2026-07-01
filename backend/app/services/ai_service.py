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

SUMMARIZATION_PROMPT = """You are a healthcare insurance expert. Analyze the following insurance document and provide a comprehensive yet clear explanatory summary.
Return ONLY a valid JSON object matching the schema below. Do not output any preamble, explanation, or conversational text.

Return format JSON:
{{
  "summary_text": "A comprehensive, detailed plain-language executive summary of the policy (around 200-250 words). Outline the primary insured parties, main coverage scopes, financial obligations (premium, co-pays, deductibles), crucial waiting periods, room rent limits, and critical exclusions. Provide a thorough explanation so the user understands the key aspects of the document.",
  "coverage_summary": "Thorough summary of major coverages, sub-limits, and benefits (maximum 100 words)",
  "exclusions_summary": "Thorough summary of key exclusions and what is not covered (maximum 100 words)",
  "waiting_period_summary": "Thorough summary of waiting periods for pre-existing or standard diseases (maximum 100 words)",
  "premium_summary": "Thorough summary of premium, deductibles, and co-payment details (maximum 100 words)"
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
# Text-based fallback extractor (used when Ollama is unavailable)
# Extracts real content from the uploaded document instead of returning hardcoded demo data.
# ─────────────────────────────────────────

def _extract_sentences_with_keywords(text: str, keywords: list[str], max_results: int = 3) -> list[str]:
    """Find sentences in document text that contain any of the given keywords."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    hits = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        if any(kw.lower() in s_clean.lower() for kw in keywords):
            hits.append(s_clean[:200])
        if len(hits) >= max_results:
            break
    return hits


def _regex_find(pattern: str, text: str, group: int = 1, default: str = "Not found in document") -> str:
    """Run a regex on document text and return the first match or a default."""
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        try:
            return m.group(group).strip()
        except IndexError:
            pass
    return default


def _build_fallback_summary(document_text: str) -> dict:
    """Build a document-specific summary by extracting real text snippets from the PDF."""
    text = document_text[:15000]  # Use more text for better extraction
    words = text.split()
    word_count = len(words)

    # --- Identify insurer / policy name ---
    insurer = _regex_find(
        r'(?:insurer|insurance company|underwritten by|issued by)[:\s]+([A-Za-z &.]+(?:Ltd|Limited|Co|Inc)?)',
        text, default=""
    )
    policy_name = _regex_find(
        r'(?:policy name|plan name|product name)[:\s]+([A-Za-z0-9 \-&]+)',
        text, default=""
    )

    # --- Introductory description (first 3 non-trivial sentences) ---
    intro_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
                       if len(s.strip()) > 60][:3]
    intro = " ".join(intro_sentences) if intro_sentences else text[:500]

    header_parts = []
    if insurer:
        header_parts.append(f"Insurer: {insurer}.")
    if policy_name:
        header_parts.append(f"Plan: {policy_name}.")
    header = " ".join(header_parts)
    summary_text = f"{header} {intro}".strip()[:1200]
    if not summary_text:
        summary_text = f"Document extracted ({word_count} words). " + text[:800]

    # --- Coverage ---
    coverage_hits = _extract_sentences_with_keywords(
        text,
        ["hospitaliz", "cover", "benefit", "daycare", "ICU", "ambulance", "sum insured", "reimburse"],
        max_results=4,
    )
    coverage_summary = "\n".join(f"• {s}" for s in coverage_hits) if coverage_hits else \
        "• Coverage details not clearly identified — please review the full document."

    # --- Exclusions ---
    excl_hits = _extract_sentences_with_keywords(
        text,
        ["exclud", "not cover", "not payable", "exception", "not admissible", "shall not"],
        max_results=4,
    )
    exclusions_summary = "\n".join(f"• {s}" for s in excl_hits) if excl_hits else \
        "• Exclusions not clearly identified — please review the full document."

    # --- Waiting period ---
    wait_hits = _extract_sentences_with_keywords(
        text,
        ["waiting period", "initial waiting", "pre-existing", "months waiting", "days waiting"],
        max_results=3,
    )
    waiting_period_summary = "\n".join(f"• {s}" for s in wait_hits) if wait_hits else \
        "• Waiting period details not found — please review the policy schedule."

    # --- Premium / cost ---
    premium_hits = _extract_sentences_with_keywords(
        text,
        ["premium", "deductible", "co-pay", "co pay", "copay", "sum insured", "GST"],
        max_results=4,
    )
    premium_summary = "\n".join(f"• {s}" for s in premium_hits) if premium_hits else \
        "• Premium and cost details not clearly identified — please review the policy schedule."

    return {
        "summary_text": summary_text,
        "coverage_summary": coverage_summary,
        "exclusions_summary": exclusions_summary,
        "waiting_period_summary": waiting_period_summary,
        "premium_summary": premium_summary,
    }


def _build_fallback_fields(document_text: str) -> list[dict]:
    """Extract structured fields directly from document text using regex patterns."""
    text = document_text[:15000]
    fields = []

    def add(name: str, value: str, category: str):
        if value and value != "Not found in document":
            fields.append({"field_name": name, "field_value": value, "field_category": category})

    add("Policy Name", _regex_find(
        r'(?:policy name|plan name|product)[:\s]+([A-Za-z0-9 \-&]+)',
        text), "policy_info")
    add("Insurer Name", _regex_find(
        r'(?:insurer|insurance company|underwritten by)[:\s]+([A-Za-z &.]+(?:Ltd|Limited|Co)?)',
        text), "policy_info")
    add("Policy Number", _regex_find(
        r'(?:policy no|policy number|certificate no)[.:\s]+([A-Z0-9/\-]+)',
        text), "policy_info")
    add("Sum Insured", _regex_find(
        r'(?:sum insured|sum assured|coverage amount)[:\s₹Rs.]+([\d,]+(?:\s*(?:Lakh|Lakhs|lakh))?)',
        text), "coverage")
    add("Premium Amount", _regex_find(
        r'(?:premium)[:\s₹Rs.]+([\d,]+(?:\s*per\s*(?:annum|year|month))?)',
        text), "premium")
    add("Deductible", _regex_find(
        r'(?:deductible|excess)[:\s₹Rs.]+([\d,]+)',
        text), "premium")
    add("Co Payment", _regex_find(
        r'(?:co-?pay(?:ment)?)[:\s]+([\d]+%?[^.\n]{0,60})',
        text), "premium")
    add("Waiting Period", _regex_find(
        r'(?:waiting period)[:\s]+([^.\n]{0,120})',
        text), "restrictions")
    add("Coverage Type", _regex_find(
        r'(?:coverage type|plan type|floater|individual)[:\s]+([A-Za-z ]+)',
        text), "coverage")
    add("Policy Term", _regex_find(
        r'(?:policy term|policy period|duration)[:\s]+([^.\n]{0,60})',
        text), "policy_info")
    add("Network Hospitals", _regex_find(
        r'([\d,]+\+?\s*(?:network|cashless|empanelled)\s*hospitals?)',
        text), "coverage")
    add("Room Rent Limit", _regex_find(
        r'(?:room rent)[^.\n]{0,40}([\d%,]+[^.\n]{0,80})',
        text), "restrictions")
    add("Claim Process", _regex_find(
        r'(?:claim process|how to claim|claims?)[:\s]+([^.\n]{0,150})',
        text), "process")

    if not fields:
        # Last resort: return document-specific note
        snippet = document_text[:200].replace("\n", " ")
        fields.append({
            "field_name": "Document Content",
            "field_value": f"Could not extract structured fields. Document preview: {snippet}",
            "field_category": "general",
        })
    return fields


def _build_fallback_risks(document_text: str) -> dict:
    """Detect risk clauses directly from document text."""
    text = document_text[:15000]
    risks = []

    risk_patterns = [
        {
            "keywords": ["pre-existing", "pre existing"],
            "risk_type": "waiting_period",
            "severity": "high",
            "explanation": "Pre-existing disease clauses typically impose multi-year waiting periods before coverage starts.",
            "recommendation": "Check the exact waiting period length and compare with other policies offering shorter waits.",
        },
        {
            "keywords": ["co-pay", "co pay", "copay", "co-payment"],
            "risk_type": "co_payment",
            "severity": "high",
            "explanation": "Co-payment clauses require you to pay a percentage of every claim out of pocket.",
            "recommendation": "Check if a co-payment waiver rider is available or compare zero co-pay alternatives.",
        },
        {
            "keywords": ["room rent", "room-rent"],
            "risk_type": "coverage_limit",
            "severity": "medium",
            "explanation": "Room rent limits can trigger proportionate deductions on all associated hospital charges.",
            "recommendation": "Choose a room within the policy's limit to avoid proportionate deductions on your entire bill.",
        },
        {
            "keywords": ["deductible", "excess"],
            "risk_type": "deductible",
            "severity": "medium",
            "explanation": "A deductible is the amount deducted from every approved claim before the insurer pays.",
            "recommendation": "Maintain an emergency fund to cover per-hospitalization deductibles.",
        },
        {
            "keywords": ["exclud", "not cover", "not payable"],
            "risk_type": "exclusion",
            "severity": "low",
            "explanation": "Exclusions define situations where the policy will not pay out, limiting your coverage.",
            "recommendation": "Read all exclusions carefully and ensure they don't apply to your medical history.",
        },
    ]

    for pattern in risk_patterns:
        hits = _extract_sentences_with_keywords(text, pattern["keywords"], max_results=1)
        if hits:
            risks.append({
                "clause_text": hits[0],
                "risk_type": pattern["risk_type"],
                "severity": pattern["severity"],
                "explanation": pattern["explanation"],
                "recommendation": pattern["recommendation"],
            })

    overall = "high" if any(r["severity"] == "high" for r in risks) else \
              "medium" if any(r["severity"] == "medium" for r in risks) else "low"

    if not risks:
        # No specific risk clauses found — return a generic note based on document content
        snippet = document_text[:300].replace("\n", " ")
        risks.append({
            "clause_text": snippet,
            "risk_type": "general",
            "severity": "low",
            "explanation": "No specific risk clauses automatically detected. Manual review recommended.",
            "recommendation": "Read the full policy document carefully before purchasing.",
        })
        overall = "low"

    return {"risks": risks, "overall_risk_level": overall}


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
    num_predict: int = 512,
    num_ctx: int = 1536,
) -> str:
    """Call Ollama API using shared connection pool with GPU-accelerated settings."""
    from app.services.ollama_client import call_ollama as _pooled_call
    return await _pooled_call(prompt, model=model, num_predict=num_predict, num_ctx=num_ctx)


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


async def generate_summary(document_text: str, force_regenerate: bool = False) -> dict:
    """Generate AI summary. Cached per document hash, falls back to demo data."""
    ck = _cache_key("summary", document_text)
    if not force_regenerate and ck in _ai_cache:
        logger.info("Cache hit: summary")
        return _ai_cache[ck]

    # Use up to 5000 chars — enough for key policy content, keeps input tokens low for speed
    truncated = document_text[:5000] if len(document_text) > 5000 else document_text
    try:
        response = await call_ollama(
            SUMMARIZATION_PROMPT.format(document_text=truncated),
            num_predict=500,
            num_ctx=2048,
        )
        result = extract_json_from_response(response)
        if result.get("summary_text"):
            logger.info("Ollama summarization successful")
            out = {
                "summary_text": _clean_field(result.get("summary_text", "")),
                "coverage_summary": _clean_field(result.get("coverage_summary")),
                "exclusions_summary": _clean_field(result.get("exclusions_summary")),
                "waiting_period_summary": _clean_field(result.get("waiting_period_summary")),
                "premium_summary": _clean_field(result.get("premium_summary")),
            }
            _ai_cache[ck] = out
            return out
    except Exception as e:
        logger.warning(f"Ollama unavailable ({e}), extracting summary from document text")
    # Ollama offline: extract real content from the uploaded document instead of returning hardcoded demo data
    fallback = _build_fallback_summary(document_text)
    _ai_cache[ck] = fallback
    return fallback


async def extract_policy_fields(document_text: str, force_regenerate: bool = False) -> list[dict]:
    """Extract key fields. Cached per document hash, falls back to demo data."""
    ck = _cache_key("fields", document_text)
    if not force_regenerate and ck in _ai_cache:
        logger.info("Cache hit: fields")
        return _ai_cache[ck]

    truncated = document_text[:5000] if len(document_text) > 5000 else document_text
    try:
        response = await call_ollama(
            FIELD_EXTRACTION_PROMPT.format(document_text=truncated),
            num_predict=400,
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
        logger.warning(f"Ollama unavailable ({e}), extracting fields from document text")
    # Ollama offline: extract real fields from the uploaded document
    fallback = _build_fallback_fields(document_text)
    _ai_cache[ck] = fallback
    return fallback


async def analyze_risks(document_text: str, force_regenerate: bool = False) -> dict:
    """Detect risky clauses. Cached per document hash, falls back to demo data."""
    ck = _cache_key("risks", document_text)
    if not force_regenerate and ck in _ai_cache:
        logger.info("Cache hit: risks")
        return _ai_cache[ck]

    truncated = document_text[:5000] if len(document_text) > 5000 else document_text
    try:
        response = await call_ollama(
            RISK_ANALYSIS_PROMPT.format(document_text=truncated),
            num_predict=400,
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
        logger.warning(f"Ollama unavailable ({e}), extracting risks from document text")
    # Ollama offline: detect risk clauses from the actual uploaded document
    fallback = _build_fallback_risks(document_text)
    _ai_cache[ck] = fallback
    return fallback


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



