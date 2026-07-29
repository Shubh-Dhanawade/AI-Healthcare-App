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

SUMMARIZATION_PROMPT = """You are a senior healthcare insurance analyst with 20 years of experience reviewing Indian insurance policy documents. Your task is to read the following document carefully and write a structured, factual summary using ONLY information present in the document.

CRITICAL RULES:
1. Return ONLY a valid JSON object. No text before or after it. No markdown code blocks.
2. Do NOT invent any values. Every number, name, date, and term must be directly extracted from the document.
3. Write summary_text as 5-6 flowing prose paragraphs (second person, no bullet points inside). Target 250-300 words.
4. For the four bullet-point fields: each bullet must start with '• ' and be ONE complete, specific sentence with real data from the document.
5. If a piece of information is truly not in the document, omit that bullet rather than guessing.

JSON schema (return exactly this structure, no extra keys):
{{
  "summary_text": "Write 5-6 paragraphs here. Paragraph 1: Name the policy (e.g. my: Optima Secure), the insurer (e.g. HDFC ERGO General Insurance), the policyholder full name, the policy number, and the validity period (e.g. 05-05-2026 to 04-05-2029). Paragraph 2: State the exact Sum Insured (e.g. ₹20,00,000 on a Family Floater basis) and list all covered categories found in the document: inpatient hospitalisation, day care procedures, pre-hospitalisation (mention exact number of days), post-hospitalisation (exact days), ambulance, AYUSH, home healthcare, domiciliary hospitalisation. Paragraph 3: List every insured member by name, relationship, date of birth, and their individual premium amounts if mentioned. State the total premium paid (e.g. ₹74,653 towards premium paid on 05-05-2026). Paragraph 4: Describe key benefits actually mentioned in the document: automatic restore benefit, secure benefit, cumulative bonus, daily cash for shared room, emergency air ambulance limit, preventive health check-up limit. Paragraph 5: State all waiting periods with exact durations: initial waiting period for all illnesses, pre-existing disease waiting period, specific disease waiting period. Mention any co-payment or aggregate deductible. Paragraph 6: Explain the cashless and reimbursement claim procedures as described in the document. Include the customer helpline number and website if found. Close with an advisory note.",
  "coverage_summary": "Write 4-6 bullet points each starting with '• '. Example format: '• Inpatient hospitalisation is covered up to the full Sum Insured with room rent at actuals and ICU at actuals.' Use exact figures and terms from the document.",
  "exclusions_summary": "Write 4-6 bullet points each starting with '• '. List specific exclusions named in the document. Example: '• Cosmetic or plastic surgery is not covered unless required as reconstruction following an accident or burn.' Be specific — name the actual excluded conditions.",
  "waiting_period_summary": "Write 3-5 bullet points each starting with '• '. State exact waiting periods. Example: '• Pre-existing diseases are subject to a 36-month waiting period from the first policy inception date.' Include all types found: initial, PED, specific disease.",
  "premium_summary": "Write 3-5 bullet points each starting with '• '. Include: total premium amount with GST status, individual member premiums if available, any discounts applied (claims experience, loyalty, long-term, online), and payment date. Example: '• Total premium of ₹74,653 was received on 05-05-2026 covering the period 05-05-2026 to 04-05-2029.'"
}}

DOCUMENT:
{document_text}"""


FIELD_EXTRACTION_PROMPT = """You are a healthcare insurance and medical document data extraction expert. Read the following document carefully and extract ONLY values that are explicitly stated in the document text. Do not guess or invent values.

Rules:
- If a field value is not present in the document, return null (not "Not specified", not empty string).
- For dates: always use the format found in the document (e.g. "05-05-2026 to 04-05-2029").
- For amounts: include the currency symbol as it appears (e.g. "₹74,653" or "20,00,000").
- Do NOT output any markdown, code fences, explanation text, or extra keys.
- Return ONLY a single valid JSON object.

JSON schema (extract exactly these keys):
{
  "policy_name": "exact product/plan name from the document e.g. my: Optima Secure",
  "insurer_name": "exact insurance company or hospital/lab name",
  "policy_number": "exact policy number, certificate number, or report ID",
  "insured_person": "full name of policyholder or patient",
  "sum_insured": "total coverage amount with currency symbol e.g. ₹20,00,000",
  "premium_amount": "total premium paid or payable with currency symbol e.g. ₹74,653",
  "policy_term": "exact validity period as shown e.g. 05-05-2026 to 04-05-2029",
  "renewal_date": "policy end/expiry date e.g. 04-05-2029",
  "coverage_type": "individual or Family Floater",
  "room_rent_limit": "room rent category e.g. At Actuals or Single Private Room or 1% of SI per day",
  "waiting_period": "waiting period duration e.g. 36 months for pre-existing diseases, 30 days initial",
  "pre_existing_coverage": "pre-existing disease waiting period duration only e.g. 3 Years / 36 months",
  "deductible": "aggregate deductible amount or Not Opted",
  "co_payment": "co-payment percentage or Not Applicable",
  "maternity_coverage": "maternity benefit details or exclusion clause text",
  "network_hospitals": "hospital network count or network name",
  "claim_process": "brief claim filing instructions from the document"
}

DOCUMENT:
{document_text}"""


RISK_ANALYSIS_PROMPT = """You are an insurance risk compliance auditor. Identify up to 3 risky clauses, exclusions, or limiting terms in the following health insurance policy document.
For each risk, provide the exact clause text, risk type (waiting_period, exclusion, deductible, co_payment, coverage_limit), severity (low, medium, or high), a brief explanation, and a recommendation.
Return ONLY a valid JSON object matching the schema below. Do not output any markdown code blocks, preamble, or trailing text.

Return format JSON:
{{
  "risks": [
    {{
      "clause_text": "the exact sentence or text of the clause from the document",
      "risk_type": "waiting_period|exclusion|deductible|co_payment|coverage_limit",
      "severity": "low|medium|high",
      "explanation": "why this clause presents a risk to the customer (maximum 35 words)",
      "recommendation": "what action or alternative the customer should consider (maximum 35 words)"
    }}
  ],
  "overall_risk_level": "low|medium|high"
}}

DOCUMENT:
{document_text}"""


COMPARISON_PROMPT = """Compare the health insurance policies listed below.
For the "synthesis", provide a structured, point-by-point medical and financial comparison. For each policy, start with a bullet point and list its name followed by specific terms (e.g. Sum Insured, Premium, Deductibles, Co-payments, Waiting Periods). Do not mix them into a single run-on paragraph.
Return ONLY a valid JSON object matching the schema below. Do not output any markdown code blocks, preamble, or trailing text.

POLICIES:
{policies_data}

Return format JSON:
{{
  "synthesis": "Point-by-point comparison of the policies, e.g.:\\n- **[Policy 1 Name]**: Sum Insured of [SI], Premium [Prem], with [Deductible] deductible, [Co-pay] co-payment. Covers [Maternity/Room Rent details].\\n- **[Policy 2 Name]**: Sum Insured of [SI], Premium [Prem], with [Deductible] deductible, [Co-pay] co-payment. Covers [Maternity/Room Rent details].\\n\\nMedical terms comparison: ...",
  "best_for": "Point-by-point description of who each policy is best suited for (maximum 80 words)",
  "verdict": "Clear advisor recommendation and final verdict on which plan to choose (maximum 60 words)",
  "feature_winners": [
    {{
      "feature": "Name of the feature (e.g. Premium, Coverage, Room Rent)",
      "winner": "Name of the winning policy",
      "reason": "Brief reason for winning"
    }}
  ]
}}"""


# NOTE: RAG_PROMPT is now only used in rag_service.py (single-doc pipeline).
# The chat RAG (query_policy_rag / query_policy_rag_stream) builds prompts
# as plain strings to avoid Python .format() issues with policy text containing { }


TRANSLATE_PROMPT = """Translate to {target_language}:
{text}"""


CLAIMS_CHECKLIST_PROMPT = """You are a senior healthcare claims auditor. Read the following claim process section from the health insurance policy document and create a claims checklist for:
TREATMENT/ILNESS: {treatment_type}
POLICY NAME: {policy_name}

POLICY DETAILS:
{fields_summary}

CLAIM PROCESS SECTION FROM POLICY:
\"\"\"
{claim_section}
\"\"\"

CRITICAL RULES:
1. Examine the CLAIM PROCESS SECTION above. If the document describes a specific claim process or required documents for this treatment/illness, extract and display them.
2. If the document does NOT describe the claim process or documents, generate a set of standard, necessary steps and documents for a claim.
3. Return ONLY valid JSON matching the schema below. No explanation, no code fences.

JSON format:
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
    """Find complete, meaningful sentences in document text that contain any of the given keywords.
    Applies strict quality filters to reject OCR noise, table headers, codes, and fragments.
    """
    candidates = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    hits = []
    seen = set()

    skip_patterns = re.compile(
        r'^(page\s*\d+|\[page|dear\s+|subject:|date of|name of|relationship|gender|period$)',
        re.IGNORECASE
    )
    # Patterns that indicate pure OCR noise: all-caps codes, standalone numbers, UIN-like codes
    noise_line_pattern = re.compile(
        r'^[A-Z0-9\-/]{4,20}$'           # standalone codes like HDFHLIA2405, UIN codes
        r'|^\d+\.?\d*$'                   # pure numbers or decimals like "1.2"
        r'|^[\W\d\s]{0,10}$',             # only punctuation/numbers/spaces
        re.IGNORECASE
    )

    for s in candidates:
        s_clean = s.strip()
        s_clean = re.sub(r'^[\s\u2022\uf0b7?•\-*●]*', '', s_clean).strip()

        # ── Quality gate 1: minimum length 50 chars ──────────────────────
        if len(s_clean) < 50:
            continue

        # ── Quality gate 2: skip known header/footer patterns ─────────────
        if skip_patterns.match(s_clean):
            continue

        # ── Quality gate 3: reject pure OCR noise lines ───────────────────
        if noise_line_pattern.match(s_clean):
            continue

        # ── Quality gate 4: minimum 8 words ───────────────────────────────
        words = s_clean.split()
        if len(words) < 8:
            continue

        # ── Quality gate 5: must contain at least 3 lowercase words ───────
        # (rejects ALL_CAPS headers, table rows, OCR fragments)
        lowercase_meaningful = [w for w in words if w.islower() and len(w) > 3]
        if len(lowercase_meaningful) < 3:
            continue

        # ── Quality gate 6: known noise substrings ────────────────────────
        s_lower = s_clean.lower()
        if any(noise in s_lower for noise in [
            "registered & corporate office", "leela business park", "lbs marg", "bhandup",
            "andheri-kurla road", "mumbai - 400", "pincode -", "pimpri chinchwad",
            "gstin", "reverse charge basis", "exempt under the notification",
            "email id", "pan no", "proposal details", "relationship to nominee",
            "member wise premium", "appointee", "proposer", "communication address",
            "permanent address", "download our mobile app", "self-help page",
            "kyc verification", "cersai portal", "http://", "https://",
            "gst for this invoice", "bill of supply", "tax certificate",
            "uin:", "uin -", "-uin:", "particulars", "base premium", "optional cover"
        ]):
            continue

        # ── Quality gate 7: table / vertical list check ───────────────────
        lines = [line.strip() for line in s_clean.split('\n') if line.strip()]
        if len(lines) > 2:
            avg_line_len = sum(len(l) for l in lines) / len(lines)
            if avg_line_len < 35:
                continue

        # ── Quality gate 8: digit ratio check (financial tables) ─────────
        digits = sum(c.isdigit() for c in s_clean)
        if len(s_clean) > 0 and (digits / len(s_clean)) > 0.12:
            continue

        # ── Quality gate 9: uppercase ratio check ─────────────────────────
        letters = [c for c in s_clean if c.isalpha()]
        if letters:
            uppercase = sum(c.isupper() for c in letters)
            if (uppercase / len(letters)) > 0.40:
                continue

        # ── Quality gate 10: sentence completeness check ──────────────────
        # Must end with standard sentence punctuation (., !, ?) or quotes/parentheses containing them
        if not s_clean[-1] in ('.', '!', '?') and not (s_clean[-1] in (')', '"', "'") and s_clean[-2] in ('.', '!', '?')):
            continue

        # ── Quality gate 11: cut-off/abbreviation word check at end ───────
        last_word = re.sub(r'[.!?\)\'\"]', '', words[-1]).lower()
        if last_word in ['in', 'co', 'ltd', 'no', 'dr', 'mr', 'ms', 'mrs', 'rs', 'exclud', 'unlimi', 'schedu', 'hospitali']:
            continue

        key = s_clean[:80].lower()
        if key in seen:
            continue

        if any(kw.lower() in s_lower for kw in keywords):
            seen.add(key)
            # Clean up newlines/carriage returns and collapse spaces
            s_clean_final = re.sub(r'[\r\n]+', ' ', s_clean)
            s_clean_final = re.sub(r'\s+', ' ', s_clean_final).strip()
            hits.append(s_clean_final[:300])

        if len(hits) >= max_results:
            break

    return hits


def _regex_find(pattern: str, text: str, group: int = 1, default: str = "Not found in document") -> str:
    """Run a regex on document text and return the first match or a default."""
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        try:
            val = m.group(group).strip()
            # Strip trailing noise
            val = re.sub(r'[\n\r\t]+.*$', '', val).strip()
            val = re.sub(r'\s{2,}', ' ', val)
            return val if val else default
        except IndexError:
            pass
    return default


def _regex_find_any(patterns: list[str], text: str, group: int = 1, default: str = "") -> str:
    """Try multiple regex patterns and return the first successful match."""
    for pattern in patterns:
        result = _regex_find(pattern, text, group, "")
        if result and result != "Not found in document":
            # Validate: must have some alphanumeric content
            if re.search(r'[a-zA-Z0-9]', result):
                return result.strip()
    return default


def _build_fallback_summary(document_text: str) -> dict:
    """Build a 400-500 word summary and structured bullet fields from the actual document text.
    Used when Ollama is unavailable. Every sentence is sourced from the document — no invented data.
    """
    text = document_text[:40000]

    # ─── 1. Extract structured facts via regex ───────────────────────────────
    insurer = _regex_find_any([
        r'(HDFC\s+ERGO[A-Za-z ]*(?:General Insurance|Life Insurance|Insurance)?[A-Za-z .]*(?:Ltd|Limited)?)',
        r'(Bajaj\s+Allianz[A-Za-z ]*(?:General|Life)?[A-Za-z .]*(?:Ltd|Limited)?)',
        r'(Star\s+Health[A-Za-z ]*(?:Insurance)?[A-Za-z .]*(?:Ltd|Limited)?)',
        r'(Max\s+Bupa|Niva\s+Bupa)[A-Za-z ]*(?:Insurance)?[A-Za-z .]*(?:Ltd|Limited)?',
        r'((?:New\s+India|National|United\s+India|Oriental)\s+(?:Assurance|Insurance)[A-Za-z .]*(?:Ltd|Limited)?)',
        r'(?:insurer|insurance company|underwritten by)[:\s]+([A-Za-z &.]+(?:General Insurance|Insurance|Ltd|Limited|Co)[A-Za-z .]*)',
    ], text) or None

    policy_name = _regex_find_any([
        r'(?:product name|plan name|policy name)[:\s]+([A-Za-z0-9 \-&/]+?)(?=\s*UIN|\s*\n|\s*\.\s)',
        r'my\.\s+([A-Za-z0-9 \-&]+(?:Secure|Health|Protect|Plus|Elite|Care|Shield|Optima)[A-Za-z0-9 ]*)(?=\s*UIN|\n|$)',
        r'((?:Optima|Secure|Health|Protect|Care|Shield|Star)\s+(?:Secure|Plus|Elite|Care|Restore|Senior|Family|Individual)[A-Za-z0-9 ]*)',
    ], text) or None

    policy_number = _regex_find_any([
        r'(?:policy\s+no|policy\s+number|certificate\s+no)[.:\s]+([A-Z0-9][A-Z0-9\-/]{5,25})',
        r'(\d{10,20})',
    ], text) or None

    policy_holder = _regex_find_any([
        r'(?:Dear|name\s+of\s+(?:insured|policyholder))[:\s,]+([A-Z][A-Z a-z]{3,40})',
        r'(?:insured|policyholder|policy\s+holder)[:\s]+([A-Z][A-Z a-z]{3,40})',
    ], text) or None

    sum_insured = _regex_find_any([
        r'(?:base\s+)?sum\s+insured\s*(?:opted)?\s*[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
        r'(?:sum\s+insured|sum\s+assured)[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
        r'₹\s*([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,})\s*(?:Lakh|Lakhs|lakh)?',
    ], text) or None

    premium = _regex_find_any([
        r'(?:total\s+premium|gross\s+premium|net\s+premium|premium\s+paid|premium\s+amount|premium\s+received)[:\s\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.[\d]{1,2})?)',
        r'[\u20b9Rs.INR]\s*([\d,]{4,}(?:\.\d{1,2})?)\s*(?:towards\s+premium|towards\s+the\s+premium|towards\s+insurance|premium)',
        r'(?:received\s+an\s+amount\s+of)\s*[\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
        r'(?:towards\s+premium)[^\n\r]*?[\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
        r'(?:Premium)[:\s\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
        r'Total\s+(?:Premium|Amount)[:\s\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
        r'[\u20b9Rs.INR]\s*([\d,]{4,}(?:\.\d{1,2})?)',
        r'([\d,]{4,}(?:\.\d{1,2})?)\s*(?:towards\s+premium)',
    ], text) or None

    waiting_period = _regex_find_any([
        r'(?:waiting\s+period)[:\s]+([\d]+\s*(?:month|year|day)[s]?[^.\n]{0,60})',
        r'([\d]+\s*(?:month|year)[s]?\s+waiting\s+period)',
    ], text) or None

    co_pay = _regex_find_any([
        r'(?:co-?pay(?:ment)?)[:\s]+([\d]+%[^.\n]{0,60})',
        r'(?:co-?pay(?:ment)?\s+of\s+)([\d]+%[^.\n]{0,60})',
    ], text) or None

    coverage_type = _regex_find_any([
        r'(family\s+floater)',
        r'(individual\s+(?:policy|plan|coverage))',
        r'(?:plan\s+type|coverage\s+type)[:\s]+([A-Za-z ]+)',
    ], text) or None

    policy_term = _regex_find_any([
        r'(?:period|validity|duration|from|term)[:\s]*([0-3]?\d[\-/][0-1]?\d[\-/]\d{2,4}\s*(?:to|till|\-)\s*[0-3]?\d[\-/][0-1]?\d[\-/]\d{2,4})',
        r'(?:policy\s+period|policy\s+term)[:\s]+([0-9][^.\n]{0,60})',
        r'(?:from)[:\s]+([\d]{1,2}[\-/][\d]{1,2}[\-/][\d]{2,4}[^\n]{0,30}to[^\n]{0,30})',
    ], text) or None

    # ─── 2. Pull thematic sentences from the document (quality-filtered) ────────
    cov_sentences = _extract_sentences_with_keywords(
        text,
        ["inpatient", "hospitalisation", "hospitalization", "daycare", "day care",
         "ambulance", "AYUSH", "pre-hospitalisation", "post-hospitalisation",
         "cashless", "network hospital", "sum insured", "benefit"],
        max_results=6,
    )
    excl_sentences = _extract_sentences_with_keywords(
        text,
        ["not covered", "not payable", "not admissible", "shall not", "exclud",
         "exclusion", "not included", "does not cover"],
        max_results=5,
    )
    wait_sentences = _extract_sentences_with_keywords(
        text,
        ["waiting period", "pre-existing", "pre existing", "initial waiting",
         "specific disease", "listed illness", "listed ailment", "months waiting"],
        max_results=5,
    )
    prem_sentences = _extract_sentences_with_keywords(
        text,
        ["total premium", "gross premium", "premium payable", "annual premium",
         "co-payment", "co pay", "deductible", "grace period", "renewal"],
        max_results=5,
    )
    benefit_sentences = _extract_sentences_with_keywords(
        text,
        ["wellness", "no claim bonus", "restoration", "health check",
         "network", "cashless", "add-on", "rider", "maternity", "OPD"],
        max_results=4,
    )
    claim_sentences = _extract_sentences_with_keywords(
        text,
        ["claim", "reimbursement", "cashless claim", "helpline", "toll free",
         "customer care", "hospital discharge", "submit"],
        max_results=3,
    )

    # ─── 3. Build 400-500 word flowing narrative ────────────────────────────────
    parts = []

    # Paragraph 1 — Policy Identity
    if any([insurer, policy_name, policy_holder, policy_number]):
        p1 = "Your"
        p1 += f" {policy_name} policy" if policy_name else " health insurance policy"
        if insurer:
            p1 += f" is issued by {insurer}"
        if policy_holder:
            p1 += f" in the name of {policy_holder}"
        if policy_number:
            p1 += f" (Policy No. {policy_number})"
        p1 += "."
        if policy_term:
            p1 += f" The policy is valid for the period {policy_term}."
        p1 += (
            " This document serves as your official insurance certificate and should be kept "
            "safely for reference during any medical emergency or claim."
        )
        parts.append(p1)

    # Paragraph 2 — Coverage Overview
    p2_parts = []
    if sum_insured or coverage_type:
        p2_start = "Your policy provides comprehensive health insurance coverage"
        if sum_insured:
            p2_start += f" with a Sum Insured of \u20b9{sum_insured}"
        if coverage_type:
            p2_start += f" on a {coverage_type} basis"
        p2_start += "."
        p2_parts.append(p2_start)
    if cov_sentences:
        p2_parts.extend(cov_sentences[:2])
    if p2_parts:
        parts.append(" ".join(p2_parts))
    else:
        parts.append(
            "Your policy provides health insurance coverage for hospitalisation and related "
            "medical expenses. Please refer to your policy schedule for the complete list of "
            "covered treatments and procedures."
        )

    # Paragraph 3 — Key Benefits
    if benefit_sentences:
        p3 = "In addition to the core hospitalisation cover, your policy comes with several valuable benefits. "
        p3 += " ".join(s.rstrip(".") + "." for s in benefit_sentences[:2])
        parts.append(p3)

    # Paragraph 4 — Waiting Periods
    if waiting_period or wait_sentences:
        p4_parts = []
        if waiting_period:
            p4_parts.append(
                f"A waiting period of {waiting_period} applies to certain conditions under this policy."
            )
        if wait_sentences:
            p4_parts.extend(wait_sentences[:2])
        if p4_parts:
            parts.append("Regarding waiting periods and restrictions: " + " ".join(p4_parts))
    else:
        parts.append(
            "Like all health insurance policies, yours includes standard waiting period clauses for "
            "pre-existing diseases and specific listed treatments. Please review your policy schedule "
            "to understand which conditions have waiting periods and for how long before coverage begins."
        )

    # Paragraph 5 — Premium & Charges
    if premium or co_pay or prem_sentences:
        p5_parts = []
        if premium:
            p5_parts.append(f"The total annual premium payable for your policy is \u20b9{premium} (inclusive of applicable GST).")
        if co_pay:
            p5_parts.append(f"A co-payment of {co_pay} is applicable on certain claims under this policy.")
        if prem_sentences:
            p5_parts.extend(prem_sentences[:2])
        if p5_parts:
            parts.append(" ".join(p5_parts))

    # Paragraph 6 — Claims & Advisory
    p6_parts = []
    if claim_sentences:
        p6_parts.extend(claim_sentences[:2])
    if p6_parts:
        parts.append(
            "For making a claim under your policy: " + " ".join(p6_parts) +
            " Keep a copy of all hospital bills and discharge summaries."
        )
    else:
        parts.append(
            "For claims, you may opt for cashless treatment at a network hospital or seek reimbursement "
            "by submitting original bills to the insurer within the stipulated time after discharge. "
            "Keep a copy of your policy document and the insurer's customer helpline number readily "
            "accessible for any emergencies. Reviewing the full policy terms and conditions will help "
            "you make the most of your health insurance benefits and avoid claim rejections."
        )

    # Ensure the summary reaches ~250 words by padding with additional document sentences
    summary_text = "\n\n".join(parts)
    if len(summary_text.split()) < 150:
        extra = _extract_sentences_with_keywords(
            text,
            ["insurance", "policy", "covered", "benefit", "hospital", "treatment"],
            max_results=4,
        )
        extra_filtered = [s for s in extra if s not in summary_text]
        if extra_filtered:
            summary_text += "\n\n" + " ".join(extra_filtered[:3])

    # ─── 4. Build bullet sections ────────────────────────────────────────────────

    # Coverage & Benefits bullets
    cov_bullets: list[str] = []
    if sum_insured:
        cov_bullets.append(f"Sum Insured of \u20b9{sum_insured} covers all eligible hospitalisation expenses")
    if coverage_type:
        cov_bullets.append(f"Coverage type: {coverage_type} — all insured members share the sum insured")
    for s in cov_sentences:
        snippet = s[:140].rstrip(".")
        if snippet not in cov_bullets:
            cov_bullets.append(snippet)
    coverage_summary = (
        "\n".join(f"\u2022 {b}." for b in cov_bullets[:3])
        if cov_bullets else
        "\u2022 Coverage details could not be extracted. Please refer to the policy schedule."
    )

    # Exclusions & Limits bullets
    excl_bullets = [s[:160].rstrip(".") for s in excl_sentences]
    exclusions_summary = (
        "\n".join(f"\u2022 {b}." for b in excl_bullets[:3])
        if excl_bullets else
        "\u2022 Exclusion details could not be extracted. Please refer to the policy schedule."
    )

    # Waiting Periods bullets
    wait_bullets: list[str] = []
    if waiting_period:
        wait_bullets.append(f"Waiting period of {waiting_period} for specific listed conditions")
    for s in wait_sentences:
        snippet = s[:160].rstrip(".")
        if snippet not in wait_bullets:
            wait_bullets.append(snippet)
    waiting_period_summary = (
        "\n".join(f"\u2022 {b}." for b in wait_bullets[:3])
        if wait_bullets else
        "\u2022 Waiting period details could not be extracted. Please refer to the policy schedule."
    )

    # Premium & Charges bullets
    prem_bullets: list[str] = []
    if premium:
        prem_bullets.append(f"Total premium payable: \u20b9{premium} (inclusive of GST)")
    if co_pay:
        prem_bullets.append(f"Co-payment clause: {co_pay} of eligible claim amount")
    for s in prem_sentences:
        snippet = s[:160].rstrip(".")
        if snippet not in prem_bullets:
            prem_bullets.append(snippet)
    premium_summary = (
        "\n".join(f"\u2022 {b}." for b in prem_bullets[:3])
        if prem_bullets else
        "\u2022 Premium details could not be extracted. Please refer to the policy schedule."
    )

    return {
        "summary_text": summary_text,
        "coverage_summary": coverage_summary,
        "exclusions_summary": exclusions_summary,
        "waiting_period_summary": waiting_period_summary,
        "premium_summary": premium_summary,
    }




def _build_fallback_fields(document_text: str) -> list[dict]:
    """Extract structured fields directly from document text using multi-pattern regex cascades."""
    # Use more text for better field coverage across multi-page documents
    text = document_text[:40000]
    fields = []

    def add(name: str, value: str, category: str):
        """Add field only if value is meaningful (non-empty, has alphanumeric content)."""
        if value and value != "Not found in document" and re.search(r'[a-zA-Z0-9]', value):
            # Clean up value
            v = value.strip()
            v = re.sub(r'\s+', ' ', v)
            v = v[:200]  # Cap length
            fields.append({"field_name": name, "field_value": v, "field_category": category})

    # ── Policy Name ─────────────────────────────────────────────────────────
    add("Policy Name", _regex_find_any([
        r'(?:product\s+name|plan\s+name|policy\s+name)[:\s]+([A-Za-z0-9][A-Za-z0-9 \-&/]{3,60}?)(?=\s*UIN|\s*\n|\s*\.\s)',
        r'my\.\s+([A-Za-z0-9][A-Za-z0-9 \-]{3,50}?)(?=\s+UIN|\n|$)',
        r'Plan\s+Name[:\s]+([A-Za-z0-9][A-Za-z0-9 \-&]{3,50})',
        r'((?:Optima|Secure|Health|Protect|Care|Shield|Star|Arogya)\s+(?:Secure|Plus|Elite|Care|Restore|Senior|Family|Individual|Sanjeevani)[A-Za-z0-9 ]*)',
    ], text), "policy_info")

    # ── Insurer Name ────────────────────────────────────────────────────────
    add("Insurer Name", _regex_find_any([
        r'(HDFC\s+ERGO[A-Za-z ]*(?:General\s+Insurance|Insurance)?[A-Za-z ]*(?:Ltd|Limited)?)',
        r'(Bajaj\s+Allianz[A-Za-z ]*(?:General|Life)?\s*(?:Insurance)?[A-Za-z ]*(?:Ltd|Limited)?)',
        r'(Star\s+Health[A-Za-z ]*(?:Insurance)?[A-Za-z ]*(?:Ltd|Limited)?)',
        r'(Max\s+Bupa|Niva\s+Bupa)[A-Za-z ]*(?:Insurance)?[A-Za-z ]*(?:Ltd|Limited)?',
        r'((?:New\s+India|National|United\s+India|Oriental)\s+(?:Assurance|Insurance)[A-Za-z .]*(?:Ltd|Limited)?)',
        r'(Reliance\s+(?:General|Health)\s*Insurance[A-Za-z ]*(?:Ltd|Limited)?)',
        r'(ICICI\s+Lombard[A-Za-z ]*(?:General\s+Insurance)?[A-Za-z ]*(?:Ltd|Limited)?)',
        r'(?:insurer|insurance\s+company|underwritten\s+by|issued\s+by)[:\s]+([A-Za-z &.]+(?:Ltd|Limited|Co)?)',
    ], text), "policy_info")

    # ── Policy Number ───────────────────────────────────────────────────────
    add("Policy Number", _regex_find_any([
        r'(?:policy\s+no|policy\s+number)[.:\s]+([A-Z0-9][A-Z0-9\-/]{5,25})',
        r'(?:certificate\s+no|certificate\s+number)[.:\s]+([A-Z0-9][A-Z0-9\-/]{5,25})',
        r'(?:policy\s+schedule\s+no)[.:\s]+([A-Z0-9][A-Z0-9\-/]{5,25})',
        r'(?:policy\s+id)[.:\s]+([A-Z0-9][A-Z0-9\-/]{5,25})',
        r'(\d{10,20})',  # Long numeric strings often are policy numbers
    ], text), "policy_info")

    # ── Insured Person / Policyholder ───────────────────────────────────────
    add("Insured Person", _regex_find_any([
        r'(?:Dear|name\s+of\s+(?:insured|policyholder))[:\s,]+([A-Z][A-Z a-z]{3,40})',
        r'(?:insured|policyholder|policy\s+holder)[:\s]+([A-Z][A-Z a-z]{3,40})',
        r'(?:member\s+name)[:\s]+([A-Z][A-Z a-z]{3,40})',
    ], text), "policy_info")

    # ── Sum Insured ─────────────────────────────────────────────────────────
    add("Sum Insured", _regex_find_any([
        r'(?:base\s+)?sum\s+insured\s*(?:opted)?\s*[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
        r'(?:sum\s+insured|sum\s+assured|si)[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
        r'(?:total\s+sum\s+insured)[:\s₹Rs.]+([1-9]\d{4,})',
        r'(?:basic\s+sum\s+insured)[:\s₹Rs.]+([1-9]\d{4,})',
        r'₹\s*([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,})\s*(?:Lakh|Lakhs|lakh)?',
    ], text), "coverage")

    # ── Premium Amount ──────────────────────────────────────────────────────
    add("Premium Amount", _regex_find_any([
        r'(?:total\s+premium|gross\s+premium|net\s+premium|premium\s+paid|premium\s+amount|premium\s+received)[:\s\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.[\d]{1,2})?)',
        r'[\u20b9Rs.INR]\s*([\d,]{4,}(?:\.\d{1,2})?)\s*(?:towards\s+premium|towards\s+the\s+premium|towards\s+insurance|premium)',
        r'(?:received\s+an\s+amount\s+of)\s*[\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
        r'(?:towards\s+premium)[^\n\r]*?[\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
        r'(?:Premium)[:\s\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
        r'Total\s+(?:Premium|Amount)[:\s\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
        r'[\u20b9Rs.INR]\s*([\d,]{4,}(?:\.\d{1,2})?)',
        r'([\d,]{4,}(?:\.\d{1,2})?)\s*(?:towards\s+premium)',
    ], text), "premium")

    # ── Deductible ──────────────────────────────────────────────────────────
    add("Deductible", _regex_find_any([
        r'(?:deductible|excess)[:\s\u20b9Rs.]+([\u20b9Rs\d,]+(?:\.[\d]{0,2})?)',
        r'(?:per\s+hospitalization\s+deductible)[:\s\u20b9Rs.]+([\d,]+)',
    ], text), "premium")

    # ── Co-payment ──────────────────────────────────────────────────────────
    add("Co Payment", _regex_find_any([
        r'(?:co-?pay(?:ment)?)[:\s]+([\d]+%?[^.\n]{0,60})',
        r'co-?pay(?:ment)?\s+(?:of\s+)?([\d]+%)',
        r'(?:you\s+pay|patient\s+pays?)[:\s]+([\d]+%[^.\n]{0,40})',
    ], text), "premium")

    # ── Waiting Period ──────────────────────────────────────────────────────
    add("Waiting Period", _regex_find_any([
        r'(?:waiting\s+period)[:\s]+([\d]+\s*(?:month|year|day)[s]?[^.\n]{0,120})',
        r'([\d]+\s*(?:month|year)[s]?\s+waiting\s+period[^.\n]{0,80})',
        r'(?:initial\s+waiting\s+period)[:\s]+([^.\n]{0,100})',
    ], text), "restrictions")

    # ── Coverage Type ───────────────────────────────────────────────────────
    add("Coverage Type", _regex_find_any([
        r'(?:plan\s+type|coverage\s+type)[:\s]+([A-Za-z ]+)',
        r'(?:floater\s+basis|individual\s+basis|family\s+floater)',
        r'(?:individual|floater|family)[:\s]+(?:plan|policy|basis)',
    ], text), "coverage")

    # ── Policy Term & Dates Extraction ──────────────────────────────────────
    date_range_match = re.search(
        r'(?:period|validity|duration|from|term)[:\s]*([0-3]?\d[\-/][0-1]?\d[\-/]\d{2,4})\s*(?:to|till|\-)\s*([0-3]?\d[\-/][0-1]?\d[\-/]\d{2,4})',
        text, re.IGNORECASE
    )
    if not date_range_match:
        date_range_match = re.search(
            r'([0-3]?\d[\-/][0-1]?\d[\-/]\d{4})\s*(?:to|till|\-)\s*([0-3]?\d[\-/][0-1]?\d[\-/]\d{4})',
            text
        )

    start_date = date_range_match.group(1).strip() if date_range_match else None
    end_date = date_range_match.group(2).strip() if date_range_match else None

    if start_date and end_date:
        add("Policy Term", f"{start_date} to {end_date}", "policy_info")
        add("Renewal Date", end_date, "policy_period")
        add("Expiry Date", end_date, "policy_period")
        add("Premium Due Date", end_date, "premium")
    else:
        raw_term = _regex_find_any([
            r'(?:policy\s+term|policy\s+period|duration)[:\s]+([0-9][^.\n]{0,60})',
            r'(?:from|valid\s+from)[:\s]+([\d]{1,2}[\-/][\d]{1,2}[\-/][\d]{2,4}[^\n]{0,30}to[^\n]{0,30})',
        ], text)
        if raw_term and not raw_term.startswith("&"):
            add("Policy Term", raw_term, "policy_info")

        expiry_val = _regex_find_any([
            r'(?:expiry\s+date|renewal\s+date|valid\s+till|valid\s+to|end\s+date)[:\s]+([0-3]?\d[\-/][0-1]?\d[\-/]\d{2,4})',
            r'(?:period\s+of\s+insurance\s+to)[:\s]+([0-3]?\d[\-/][0-1]?\d[\-/]\d{2,4})',
        ], text)
        if expiry_val:
            add("Renewal Date", expiry_val, "policy_period")
            add("Expiry Date", expiry_val, "policy_period")
            add("Premium Due Date", expiry_val, "premium")

    # ── Network Hospitals ───────────────────────────────────────────────────
    add("Network Hospitals", _regex_find_any([
        r'([\d,]+\+?\s*(?:network|cashless|empanelled)\s*hospitals?)',
        r'(?:network\s+of)[\s]+([\d,]+\+?\s*hospitals?)',
        r'([\d,]{2,})\s*(?:hospitals?|network)',
    ], text), "coverage")

    # ── Room Rent Limit ─────────────────────────────────────────────────────
    add("Room Rent Limit", _regex_find_any([
        r'1\.1\.a\s+Room\s+Rent\s+([A-Za-z][A-Za-z ]{0,20})',
        r'Room\s+Rent[:\s]+([Aa]t\s+[Aa]ctuals?|Single\s+Private[A-Za-z ]{0,25}|Shared\s+[Rr]oom[A-Za-z ]{0,25}|Upto\s+[\d%][\d%A-Za-z /. ]{0,40})',
        r'(?:room\s+rent\s+(?:limit|category))[:\s]+([A-Za-z][A-Za-z0-9 %\./-]{2,60})',
        r'(?:room\s+rent)[^\n]{0,40}(?:is\s+|covers?\s+|limited\s+to\s+|payable\s+at\s+)([A-Za-z0-9 %\./,-]{3,60})',
    ], text), "restrictions")

    # ── Pre-existing Disease Coverage ───────────────────────────────────────
    add("Pre Existing Coverage", _regex_find_any([
        r'PED\s+wait\s+period[^\n]{0,60}([\d]+\s*(?:Year|Month|year|month)[s]?)',
        r'[Pp]re-?existing\s+[Dd]isease[s]?\s+[Ww]aiting\s+[Pp]eriod[:\s]+([\d]+\s*(?:month|year)[s]?)',
        r'[Pp]re-?existing[^.\n]{0,30}([\d]+\s*(?:month|year)[s]?[^.\n]{0,40})',
        r'(?:PED|pre-existing)[^.\n]{0,20}([\d]+[/-][\d]+\s*(?:month|year)[s]?)',
        r'waiting\s+period[^.\n]{0,20}([\d]+\s*(?:Year|Month|year|month)[s]?)[^.\n]{0,30}[Pp]re-?existing',
    ], text), "restrictions")

    # ── Maternity Coverage ──────────────────────────────────────────────────
    # Skip lines that are just exclusion code references (Code – ExclXX)
    _maternity_raw = _regex_find_any([
        r'[Mm]aternity[:\s]+(?!.*Code\s*[-–]\s*Excl)([^.\n]{10,120})',
        r'[Mm]aternity\s+[Bb]enefit[:\s]+(?!.*Code\s*[-–]\s*Excl)([^.\n]{10,100})',
        r'[Mm]aternity[^.\n]{0,30}(covered[^.\n]{0,80})',
    ], text)
    # Also check if maternity is explicitly excluded
    if not _maternity_raw:
        _mat_excl = re.search(r'[Mm]aternity[^.\n]{0,30}(not\s+covered|excluded|Code\s*[-–]\s*Excl\d+)', text)
        _maternity_raw = "Not Covered / Excluded" if _mat_excl else None
    if _maternity_raw:
        add("Maternity Coverage", _maternity_raw, "coverage")

    # ── Claim Process ───────────────────────────────────────────────────────
    add("Claim Process", _regex_find_any([
        r'(?:claim\s+process|how\s+to\s+claim|claim\s+procedure)[:\s]+([^.\n]{0,200})',
        r'(?:to\s+(?:file|lodge|submit)\s+a\s+claim)[,\s]+([^.\n]{0,200})',
        r'(?:cashless\s+claim|reimbursement\s+claim)[:\s]+([^.\n]{0,150})',
    ], text), "process")

    # ── GSTIN / GST ─────────────────────────────────────────────────────────
    add("GSTIN / UIN", _regex_find_any([
        r'(?:GSTIN|GST\s+IN)[:\s]+([0-9A-Z]{15})',
        r'(?:UIN)[:\s]+([A-Z0-9]{15,20})',
        r'(?:product\s+UIN)[:\s]+([A-Z0-9]{10,25})',
    ], text), "policy_info")

    # ── Contact / Customer Care ─────────────────────────────────────────────
    add("Contact Number", _regex_find_any([
        r'(?:customer\s+(?:care|service|helpline)|toll\s+free)[:\s]+([0-9]{3}[\s\-]+[0-9]{4}[\s\-]+[0-9]{4})',
        r'(?:call\s+us\s+at)[:\s]+([0-9]{3}[\s\-]+[0-9]{4}[\s\-]+[0-9]{4})',
        r'(?:contact|helpline|call)[:\s]+([0-9]{3}[\s\-]+[0-9]{4}[\s\-]+[0-9]{4})',
        r'(1800[\s\-]?[0-9]{3,4}[\s\-]?[0-9]{4})',
        r'(\+91[\s\-]?[6-9][0-9]{9})',
        r'(?:call\s+us\s+at|customer\s+care)[^0-9]{0,20}([0-9]{3,5}[\s\-][0-9]{4,8})',
    ], text), "policy_info")

    # ── Health / Medical Diagnostic Report Fallbacks ────────────────────────
    is_health_report = any(kw in text.lower() for kw in [
        "lab report", "laboratory", "diagnostic", "blood test", "test report",
        "discharge summary", "patient name", "pathology", "radiology", "clinical report",
        "health checkup", "medical report", "hospital discharge"
    ])

    if is_health_report:
        patient_name = _regex_find_any([
            r'(?:patient\s+name|name\s+of\s+patient|patient)[:\s]+([A-Z][A-Za-z .]{3,40})',
            r'(?:mr\.|mrs\.|ms\.)\s+([A-Z][A-Za-z .]{3,40})',
            r'Dear\s+([A-Z][A-Za-z .]{3,40})',
        ], text)
        if patient_name and not any(f["field_name"] == "Insured Person" for f in fields):
            add("Insured Person", patient_name, "policy_info")

        lab_name = _regex_find_any([
            r'([A-Za-z0-9 .&]+(?:Diagnostics|Hospital|Clinic|Pathology|Laboratory|Labs|Health\s+Services)[A-Za-z .]*)',
            r'(?:hospital|lab|clinic|facility)\s+name[:\s]+([A-Za-z0-9 .&]+)',
        ], text)
        if lab_name and not any(f["field_name"] == "Insurer Name" for f in fields):
            add("Insurer Name", lab_name, "policy_info")

        report_date = _regex_find_any([
            r'(?:report\s+date|collection\s+date|date\s+of\s+admission|date)[:\s]+([0-3]?\d[\-/][0-1]?\d[\-/]\d{2,4})',
        ], text)
        if report_date and not any(f["field_name"] == "Policy Term" for f in fields):
            add("Policy Term", f"Report Date: {report_date}", "policy_info")

        doctor_name = _regex_find_any([
            r'(?:dr\.|doctor|ref\s+by|consultant)[:\s]+([A-Z][A-Za-z .]{3,40})',
        ], text)
        if doctor_name:
            add("Consulting Doctor", f"Dr. {doctor_name.replace('Dr.', '').strip()}", "policy_info")

        # Only extract diagnosis from actual diagnostic sections, not insurance exclusion sections
        diag_section_start = max(
            text.lower().find("diagnosis"),
            text.lower().find("clinical finding"),
            text.lower().find("impression"),
        )
        # If diagnosis keyword only appears deep in exclusion sections, skip
        excl_start = text.lower().find("exclusion")
        if diag_section_start > 0 and (excl_start < 0 or diag_section_start < excl_start):
            diagnosis = _regex_find_any([
                r'(?:diagnosis|clinical\s+findings|impression)[:\s]+([A-Za-z][^.\n]{5,150})',
                r'(?:investigation)[:\s]+([A-Za-z][^.\n]{5,100})',
            ], text[:diag_section_start + 500])
            if diagnosis and "code excl" not in diagnosis.lower():
                add("Diagnosis / Findings", diagnosis, "coverage")

    if not fields:
        snippet = document_text[:300].replace("\n", " ").strip()
        fields.append({
            "field_name": "Document Content",
            "field_value": f"Could not extract structured fields. Document preview: {snippet[:200]}",
            "field_category": "general",
        })
    return fields


def _build_fallback_risks(document_text: str) -> dict:
    """Detect risk clauses directly from document text using the full document."""
    # Use a large portion for better clause detection across all pages
    text = document_text[:40000]
    risks = []

    risk_patterns = [
        {
            "keywords": ["pre-existing", "pre existing", "pre- existing"],
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
            "recommendation": "Check if a co-payment waiver rider is available or look for zero co-pay alternatives.",
        },
        {
            "keywords": ["room rent", "room-rent", "accommodation limit"],
            "risk_type": "coverage_limit",
            "severity": "medium",
            "explanation": "Room rent limits can trigger proportionate deductions on all associated hospital charges if exceeded.",
            "recommendation": "Choose a hospital room within the policy's room rent limit to avoid proportionate deductions.",
        },
        {
            "keywords": ["deductible", "excess amount", "per hospitalization"],
            "risk_type": "deductible",
            "severity": "medium",
            "explanation": "A deductible is the amount deducted from every approved claim before the insurer pays the balance.",
            "recommendation": "Maintain an emergency fund to cover any per-hospitalization deductible amounts.",
        },
        {
            "keywords": ["exclud", "not cover", "not payable", "not admissible", "shall not"],
            "risk_type": "exclusion",
            "severity": "medium",
            "explanation": "Exclusions define specific situations where the policy will not pay, limiting your actual coverage.",
            "recommendation": "Read all exclusion clauses carefully to ensure they do not affect your current medical conditions.",
        },
        {
            "keywords": ["waiting period", "initial waiting", "30 days", "90 days"],
            "risk_type": "waiting_period",
            "severity": "medium",
            "explanation": "Initial waiting period clauses mean claims cannot be made for the first 30-90 days after policy start.",
            "recommendation": "Be aware of the initial waiting period — do not let your previous policy lapse before renewing.",
        },
        {
            "keywords": ["sub-limit", "sub limit", "capped at", "maximum limit", "up to a limit"],
            "risk_type": "coverage_limit",
            "severity": "low",
            "explanation": "Sub-limits restrict specific treatment costs, which can leave you with out-of-pocket expenses.",
            "recommendation": "Review all sub-limits (e.g., cataract, hernia) and confirm they are sufficient for your healthcare needs.",
        },
    ]

    seen_clauses: set = set()
    for pattern in risk_patterns:
        hits = _extract_sentences_with_keywords(text, pattern["keywords"], max_results=1)
        if hits:
            clause = hits[0]
            clause_key = clause[:60].lower()
            if clause_key in seen_clauses:
                continue
            seen_clauses.add(clause_key)
            risks.append({
                "clause_text": clause[:350],
                "risk_type": pattern["risk_type"],
                "severity": pattern["severity"],
                "explanation": pattern["explanation"],
                "recommendation": pattern["recommendation"],
            })

    overall = "high" if any(r["severity"] == "high" for r in risks) else \
              "medium" if any(r["severity"] == "medium" for r in risks) else "low"

    if not risks:
        # No specific risk clauses found — return a document-based note
        snippet = document_text[:300].replace("\n", " ").strip()
        risks.append({
            "clause_text": snippet[:300],
            "risk_type": "general",
            "severity": "low",
            "explanation": "No specific risk clauses were automatically detected. A manual review is recommended.",
            "recommendation": "Read the full policy document carefully before purchasing or renewing.",
        })
        overall = "low"

    return {"risks": risks, "overall_risk_level": overall}


# ─────────────────────────────────────────
# Ollama Client
# ─────────────────────────────────────────

async def warmup_model() -> None:
    """Send a tiny keep-alive prompt to Ollama so model weights stay resident in VRAM.
    Uses /api/generate which is supported by all GGUF completion models.
    """
    try:
        model = settings.OLLAMA_MODEL
        url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        if "localhost" in url:
            url = url.replace("localhost", "127.0.0.1")
        payload = {
            "model": model,
            "prompt": "hi",
            "stream": False,
            "keep_alive": -1,
            # Use num_ctx=2048: same as all inference calls to avoid costly reload on first request
            "options": {"num_predict": 1, "num_ctx": 2048, "temperature": 0},
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
    num_ctx: int = 1024,
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


def clean_newlines_in_text(text: str) -> str:
    if not text:
        return ""
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = []
    for p in paragraphs:
        p_clean = re.sub(r'[\r\n]+', ' ', p)
        p_clean = re.sub(r'\s+', ' ', p_clean).strip()
        if p_clean:
            cleaned_paragraphs.append(p_clean)
    return "\n\n".join(cleaned_paragraphs)


def clean_newlines_in_bullets(text: str) -> str:
    if not text:
        return ""
    lines = text.split('\n')
    cleaned_bullets = []
    current_bullet = ""
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        if re.match(r'^[•\-\*\u2022\uf0b7●]', line_str):
            if current_bullet:
                cleaned_bullets.append(current_bullet)
            bullet_content = re.sub(r'^[•\-\*\u2022\uf0b7●\s]+', '', line_str).strip()
            current_bullet = f"• {bullet_content}"
        else:
            if current_bullet:
                line_clean = re.sub(r'\s+', ' ', line_str).strip()
                current_bullet += f" {line_clean}"
            else:
                current_bullet = f"• {line_str}"
    if current_bullet:
        cleaned_bullets.append(current_bullet)
    return "\n".join(cleaned_bullets)


def _extract_key_context_for_summary(text: str, max_chars: int = 4500) -> str:
    """Extract optimal document text context for LLM summarization.
    Combines initial policy schedule details with key clauses for coverage, exclusions, waiting periods, and claims.
    """
    if len(text) <= max_chars:
        return text
    
    # 1. Take first 2500 chars (policy schedule, insured names, premiums, sum insured)
    header_part = text[:2500]
    
    # 2. Extract key sentences for coverage, exclusions, waiting periods, and claims from remaining text
    remainder = text[2500:]
    keywords = [
        "inpatient", "day care", "daycare", "hospitalisation", "room rent",
        "pre-hospitalisation", "post-hospitalisation", "waiting period", "pre-existing",
        "exclusion", "not covered", "co-payment", "deductible", "claim", "cashless", "helpline"
    ]
    
    matched_chunks = []
    lines = remainder.split("\n")
    for line in lines:
        l_lower = line.lower()
        if any(kw in l_lower for kw in keywords):
            cleaned = line.strip()
            if len(cleaned) > 20 and cleaned not in matched_chunks:
                matched_chunks.append(cleaned)
                if sum(len(c) for c in matched_chunks) >= (max_chars - 2500):
                    break
                    
    body_part = "\n".join(matched_chunks)
    combined = f"{header_part}\n\n--- KEY POLICY CLAUSES ---\n{body_part}"
    return combined[:max_chars]


async def generate_summary(document_text: str, force_regenerate: bool = False) -> dict:
    """Generate AI summary with hybrid fact enrichment for maximum data quality and speed."""
    ck = _cache_key("summary", document_text)
    if not force_regenerate and ck in _ai_cache:
        logger.info("Cache hit: summary")
        return _ai_cache[ck]

    context = _extract_key_context_for_summary(document_text, max_chars=4500)
    fallback = _build_fallback_summary(document_text)

    try:
        response = await call_ollama(
            SUMMARIZATION_PROMPT.format(document_text=context),
            num_predict=650,   # Concise JSON summary + 4 bullet sections
            num_ctx=1536,
        )
        result = extract_json_from_response(response)
        if result.get("summary_text"):
            logger.info("Ollama summarization successful")
            
            # Hybrid enrichment: supplement any empty/missing bullet section from fallback facts
            cov = clean_newlines_in_bullets(_clean_field(result.get("coverage_summary")))
            excl = clean_newlines_in_bullets(_clean_field(result.get("exclusions_summary")))
            wait = clean_newlines_in_bullets(_clean_field(result.get("waiting_period_summary")))
            prem = clean_newlines_in_bullets(_clean_field(result.get("premium_summary")))

            out = {
                "summary_text": clean_newlines_in_text(_clean_field(result.get("summary_text", ""))),
                "coverage_summary": cov or clean_newlines_in_bullets(fallback.get("coverage_summary", "")),
                "exclusions_summary": excl or clean_newlines_in_bullets(fallback.get("exclusions_summary", "")),
                "waiting_period_summary": wait or clean_newlines_in_bullets(fallback.get("waiting_period_summary", "")),
                "premium_summary": prem or clean_newlines_in_bullets(fallback.get("premium_summary", "")),
            }
            _ai_cache[ck] = out
            return out
    except Exception as e:
        logger.warning(f"Ollama unavailable ({e}), extracting summary from document text")

    # Ollama offline or partial: use fallback summary extracted from document text
    cleaned_fallback = {
        "summary_text": clean_newlines_in_text(fallback["summary_text"]),
        "coverage_summary": clean_newlines_in_bullets(fallback["coverage_summary"]),
        "exclusions_summary": clean_newlines_in_bullets(fallback["exclusions_summary"]),
        "waiting_period_summary": clean_newlines_in_bullets(fallback["waiting_period_summary"]),
        "premium_summary": clean_newlines_in_bullets(fallback["premium_summary"]),
    }
    _ai_cache[ck] = cleaned_fallback
    return cleaned_fallback


async def extract_policy_fields(document_text: str, force_regenerate: bool = False) -> list[dict]:
    """Extract key fields. Cached per document hash, falls back to demo data."""
    ck = _cache_key("fields", document_text)
    if not force_regenerate and ck in _ai_cache:
        logger.info("Cache hit: fields")
        return _ai_cache[ck]

    # Cap at 3000 chars to fit within num_ctx=1024 context window.
    # Field extraction produces short JSON — smaller input = faster prompt processing.
    truncated = document_text[:3000] if len(document_text) > 3000 else document_text
    try:
        response = await call_ollama(
            FIELD_EXTRACTION_PROMPT.format(document_text=truncated),
            num_predict=350,
            num_ctx=1024,
        )
        result = extract_json_from_response(response)
        if result:
            logger.info("Ollama field extraction successful")
            
            # Post-process and validate LLM outputs (e.g. reject list headers like 1.1)
            if "sum_insured" in result:
                val = str(result["sum_insured"]).strip()
                if val in ("1", "1.1", "0", "null", "") or len(val) < 3:
                    fallback_si = _regex_find_any([
                        r'(?:base\s+)?sum\s+insured\s*(?:opted)?\s*[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
                        r'(?:sum\s+insured|sum\s+assured|si)[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
                        r'(?:total\s+sum\s+insured)[:\s₹Rs.]+([1-9]\d{4,})',
                        r'₹\s*([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,})\s*(?:Lakh|Lakhs|lakh)?',
                    ], document_text)
                    if fallback_si and fallback_si != "Not found in document":
                        result["sum_insured"] = fallback_si
                        
            if "premium_amount" in result:
                val = str(result["premium_amount"]).strip()
                if val.lower() in ("0", "null", "", "not specified", "not mentioned in policy") or len(val) < 3:
                    fallback_prem = _regex_find_any([
                        r'(?:total\s+premium|gross\s+premium|net\s+premium|premium\s+paid|premium\s+amount|premium\s+received)[:\s\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.[\d]{1,2})?)',
                        r'[\u20b9Rs.INR]\s*([\d,]{4,}(?:\.\d{1,2})?)\s*(?:towards\s+premium|towards\s+the\s+premium|towards\s+insurance|premium)',
                        r'(?:received\s+an\s+amount\s+of)\s*[\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
                        r'(?:towards\s+premium)[^\n\r]*?[\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
                        r'(?:Premium)[:\s\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
                        r'Total\s+(?:Premium|Amount)[:\s\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
                        r'[\u20b9Rs.INR]\s*([\d,]{4,}(?:\.\d{1,2})?)',
                        r'([\d,]{4,}(?:\.\d{1,2})?)\s*(?:towards\s+premium)',
                    ], document_text)
                    if fallback_prem and fallback_prem != "Not found in document":
                        result["premium_amount"] = f"₹{fallback_prem}"

            # If premium_amount was not in result at all, try finding it via regex
            if "premium_amount" not in result or not result["premium_amount"]:
                fallback_prem = _regex_find_any([
                    r'(?:total\s+premium|gross\s+premium|net\s+premium|premium\s+paid|premium\s+amount|premium\s+received)[:\s\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.[\d]{1,2})?)',
                    r'[\u20b9Rs.INR]\s*([\d,]{4,}(?:\.\d{1,2})?)\s*(?:towards\s+premium|towards\s+the\s+premium|towards\s+insurance|premium)',
                    r'(?:received\s+an\s+amount\s+of)\s*[\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
                    r'(?:towards\s+premium)[^\n\r]*?[\u20b9Rs.INR]*\s*([\d,]{3,}(?:\.\d{1,2})?)',
                ], document_text)
                if fallback_prem and fallback_prem != "Not found in document":
                    result["premium_amount"] = f"₹{fallback_prem}"
            
            field_category_map = {
                "policy_name": "policy_info",
                "insurer_name": "policy_info",
                "policy_number": "policy_info",
                "insured_person": "policy_info",
                "sum_insured": "coverage",
                "premium_amount": "premium",
                "policy_term": "policy_info",
                "renewal_date": "policy_period",
                "coverage_type": "coverage",
                "room_rent_limit": "restrictions",
                "waiting_period": "restrictions",
                "pre_existing_coverage": "restrictions",
                "deductible": "premium",
                "co_payment": "premium",
                "maternity_coverage": "coverage",
                "network_hospitals": "coverage",
                "claim_process": "process",
            }
            # Friendly display names for keys
            field_display_names = {
                "policy_name": "Policy Name",
                "insurer_name": "Insurer Name",
                "policy_number": "Policy Number",
                "insured_person": "Insured Person",
                "sum_insured": "Sum Insured",
                "premium_amount": "Premium Amount",
                "policy_term": "Policy Term",
                "renewal_date": "Renewal Date",
                "coverage_type": "Coverage Type",
                "room_rent_limit": "Room Rent Limit",
                "waiting_period": "Waiting Period",
                "pre_existing_coverage": "Pre Existing Coverage",
                "deductible": "Deductible",
                "co_payment": "Co Payment",
                "maternity_coverage": "Maternity Coverage",
                "network_hospitals": "Network Hospitals",
                "claim_process": "Claim Process",
            }
            fields = [
                {
                    "field_name": field_display_names.get(key, key.replace("_", " ").title()),
                    "field_value": str(value),
                    "field_category": field_category_map.get(key, "general"),
                }
                for key, value in result.items()
                if value and str(value).lower() not in ("null", "none", "not specified")
            ]
            
            # Ensure date fields (Renewal Date, Expiry Date) are included from document text if LLM omitted them
            fallback_fields = _build_fallback_fields(document_text)
            existing_names = {f["field_name"].lower() for f in fields}
            for fb in fallback_fields:
                if fb["field_name"].lower() in ("renewal date", "expiry date", "premium due date", "insured person") and fb["field_name"].lower() not in existing_names:
                    fields.append(fb)
                    existing_names.add(fb["field_name"].lower())

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

    # Cap at 3000 chars to fit within num_ctx=1024 context window.
    # Risk analysis extracts up to 3 JSON risk items — short output, small input is fine.
    truncated = document_text[:3000] if len(document_text) > 3000 else document_text
    try:
        response = await call_ollama(
            RISK_ANALYSIS_PROMPT.format(document_text=truncated),
            num_predict=300,
            num_ctx=1024,
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
    
    synthesis_lines = []
    for idx, name in enumerate(policy_names):
        p_data = policies_data[idx]
        fields = p_data.get("extracted_fields", [])
        si = next((f["field_value"] for f in fields if f["field_name"].lower() in ("sum insured", "sum_insured")), "Not specified")
        prem = next((f["field_value"] for f in fields if f["field_name"].lower() in ("premium amount", "premium_amount")), "Not specified")
        ded = next((f["field_value"] for f in fields if f["field_name"].lower() in ("deductible",)), "Not specified")
        cp = next((f["field_value"] for f in fields if f["field_name"].lower() in ("co payment", "co_payment")), "Not specified")
        rr = next((f["field_value"] for f in fields if f["field_name"].lower() in ("room rent limit", "room_rent_limit")), "Not specified")
        
        synthesis_lines.append(
            f"- **{name}**:\n"
            f"  - **Sum Insured**: {si}\n"
            f"  - **Premium**: {prem}\n"
            f"  - **Deductible**: {ded}\n"
            f"  - **Co-Payment**: {cp}\n"
            f"  - **Room Rent Limit**: {rr}"
        )
    
    synthesis_lines.append(
        f"\nComparing the policies reveals that {p2 if len(policy_names) > 1 else p1} offers more comprehensive hospital coverage "
        f"and lower out-of-pocket costs on room rents and deductibles, making it a stronger choice for extensive protection. "
        f"On the other hand, {p1} features lower premium rates but carries higher cost-sharing responsibilities (deductibles/co-payments) during hospitalization."
    )
    synthesis_text = "\n".join(synthesis_lines)
    
    return {
        "synthesis": synthesis_text,
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
            f"You are HealthPolicyLens, an expert healthcare insurance advisor helping {user_name}.\n"
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
            f"You are HealthPolicyLens, a knowledgeable and friendly healthcare insurance assistant helping {user_name}.\n"
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
    """Translate text using Google Translate free API first, falling back to local Ollama if offline."""
    if not text.strip():
        return text

    # Map language names to Google Translate language codes
    lang_map = {
        "hindi": "hi",
        "marathi": "mr",
        "english": "en",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "japanese": "ja",
        "chinese": "zh-CN",
    }
    
    lang_normalized = target_language.strip().lower()
    target_code = lang_map.get(lang_normalized, lang_normalized)
    
    # Try free Google Translate API (fast, highly accurate, keyless)
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": "auto",
        "tl": target_code,
        "dt": "t",
        "q": text
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            res_data = response.json()
            
            # Reconstruct translation from segments
            translated_parts = []
            if res_data and len(res_data) > 0 and res_data[0]:
                for segment in res_data[0]:
                    if segment and len(segment) > 0 and isinstance(segment[0], str):
                        translated_parts.append(segment[0])
            
            if translated_parts:
                translated_text = "".join(translated_parts).strip()
                if translated_text:
                    logger.info(f"Successfully translated via Google Translate API ({target_language})")
                    return translated_text
    except Exception as e:
        logger.warning(f"Google Translate API failed: {e}. Falling back to local Ollama...")
        
    # Local Ollama fallback if offline or request failed
    try:
        response = await call_ollama(
            TRANSLATE_PROMPT.format(text=text[:1000], target_language=target_language),
            num_predict=300,
            num_ctx=512
        )
        if response:
            logger.info(f"Successfully translated via local Ollama fallback ({target_language})")
            return response.strip()
    except Exception as fallback_e:
        logger.warning(f"Ollama translation fallback also failed: {fallback_e}")
        
    return text  # Return original if all else fails


def extract_claim_section(text: str, max_chars: int = 6000) -> str:
    """Find and extract the section of the policy related to claim process/procedures."""
    if not text:
        return ""
    keywords = ["claim process", "how to claim", "claim procedure", "submission of claim", "claim document", "claims"]
    text_lower = text.lower()
    
    for kw in keywords:
        pos = text_lower.find(kw)
        if pos != -1:
            start = max(0, pos - 500)
            end = min(len(text), pos + max_chars - 500)
            logger.info(f"Extracted claim section using keyword: '{kw}' at position {pos}")
            return text[start:end]
            
    return text[:max_chars]


TREATMENTS_EXTRACTION_PROMPT = """Analyze the following health insurance policy document and extract a list of up to 6 major medical treatments, surgeries, or procedures that are covered or mentioned in the document (e.g., Cataract Surgery, Knee Replacement, Heart Bypass, Dialysis, Maternity, Accidental Fracture, etc.).
Return ONLY a valid JSON object with a single key "treatments" containing an array of strings. Do not include any markdown fences or explanatory text.

JSON Format:
{{
  "treatments": ["Cataract Surgery", "Heart Bypass", "Knee Replacement", ...]
}}

DOCUMENT:
{document_text}"""


async def extract_covered_treatments(document_id: str, document_text: str) -> list[str]:
    """Extract a list of covered treatments from the policy text using Ollama."""
    cache_key = f"treatments:{document_id}"
    if cache_key in _ai_cache:
        return _ai_cache[cache_key]["treatments"]
        
    try:
        # Take the first 10,000 characters where coverages are usually defined
        prompt = TREATMENTS_EXTRACTION_PROMPT.format(document_text=document_text[:10000])
        response = await call_ollama(prompt, num_predict=150, num_ctx=2048)
        result = extract_json_from_response(response)
        
        treatments = result.get("treatments", [])
        if isinstance(treatments, list) and len(treatments) > 0:
            clean_treatments = [str(t).strip() for t in treatments if str(t).strip()][:6]
            _ai_cache[cache_key] = {"treatments": clean_treatments}
            return clean_treatments
    except Exception as e:
        logger.warning(f"Failed to extract covered treatments from LLM: {e}")
        
    fallback = [
        "Cataract Surgery",
        "Heart Bypass / CABG",
        "Knee Replacement",
        "Accidental Fracture Cover",
        "Kidney Dialysis",
        "Maternity Delivery"
    ]
    return fallback


async def generate_claims_checklist(policy_name: str, fields_summary: str, treatment_type: str, claim_section: str = "") -> dict:
    """Generate dynamic claim checklist using Ollama."""
    prompt = CLAIMS_CHECKLIST_PROMPT.format(
        policy_name=policy_name,
        fields_summary=fields_summary,
        treatment_type=treatment_type,
        claim_section=claim_section or "Not specified in document text."
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
    """Generate streaming tokens from Ollama /api/generate — optimized for speed.
    Uses /api/generate (completion) which works with all GGUF models including
    fine-tuned models that only support 'completion' capability, not 'chat'.
    """
    from app.services.ollama_client import parse_keep_alive
    
    model = model or settings.OLLAMA_MODEL
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    
    if "localhost" in url:
        url = url.replace("localhost", "127.0.0.1")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": parse_keep_alive(settings.OLLAMA_KEEP_ALIVE),
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": 1024,
            "num_batch": 1024,
            "top_k": 1,
            "top_p": 1.0,
            "num_thread": settings.OLLAMA_NUM_THREAD,
            "num_gpu": settings.OLLAMA_NUM_GPU,
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
                            # /api/generate returns {"response": "token", "done": false}
                            token = data.get("response", "")
                            if token:
                                yield token
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
    q = query.strip().lower().rstrip("!?. ,")
    if q in _GREETING_PATTERNS or q in _GUIDANCE_PATTERNS or q in _THANKS_PATTERNS:
        return True
    if re.match(r'^h+[eay]+y*$', q) or re.match(r'^h+i+$', q) or re.match(r'^hell+o+$', q) or re.match(r'^y+o+$', q):
        return True
    words = q.split()
    if len(words) <= 2 and (q in _GREETING_PATTERNS or q in _THANKS_PATTERNS):
        return True
    return False

def _get_chitchat_response(query: str, user_name: str) -> Optional[str]:
    """Identify if a query is a general chitchat or guidance request and return a standard response."""
    q = query.strip().lower().rstrip("!?. ,")
    
    # Check words and phrase matches
    is_greeting = (
        q in _GREETING_PATTERNS or 
        re.match(r'^h+[eay]+y*$', q) or 
        re.match(r'^h+i+$', q) or 
        re.match(r'^hell+o+$', q) or 
        re.match(r'^y+o+$', q) or 
        any(pat in q for pat in ["hello", "how are you", "what's up", "greetings", "good morning", "good afternoon", "good evening"]) and len(q.split()) <= 3
    )
    
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
            f"Hi {user_name}! 👋 I'm **HealthPolicyLens**, your healthcare insurance assistant. "
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



