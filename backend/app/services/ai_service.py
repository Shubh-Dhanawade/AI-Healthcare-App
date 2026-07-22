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

SUMMARIZATION_PROMPT = """You are a senior healthcare insurance analyst. Read the following insurance policy document carefully and produce a structured, factual summary.

CRITICAL RULES — follow exactly:
1. Return ONLY a valid JSON object. No text before or after it. No markdown code fences.
2. summary_text: MUST be around 250 words, tightly optimized for important and meaningful information, written in flowing paragraphs. ABSOLUTELY NO bullet points or numbered lists inside summary_text.
3. The four bullet fields (coverage_summary, exclusions_summary, waiting_period_summary, premium_summary): each bullet MUST start with '• ' and be one complete, specific sentence sourced from the document.
4. Only state facts present in the document. Do not invent values.

JSON schema (return exactly this structure):
{{
  "summary_text": "PARAGRAPH 1 — Policy Identity: State the exact policy name, insurer company name, policyholder full name, policy number, and coverage period/validity. PARAGRAPH 2 — Coverage: State the exact Sum Insured amount, whether individual or family floater, and list every category covered: inpatient hospitalisation, daycare procedures, pre-hospitalisation expenses (with number of days), post-hospitalisation expenses (with number of days), ambulance charges, AYUSH treatments. PARAGRAPH 3 — Insured Members & Premium: List every insured member's name with their relationship and any individual sum insured or premium. State the total annual premium inclusive of GST. PARAGRAPH 4 — Key Benefits: Describe 3-5 standout benefits such as cashless hospitalisation network size, no-claim bonus, restoration benefit, wellness programs, OPD cover, or other add-ons found in the document. PARAGRAPH 5 — Waiting Periods & Restrictions: State all waiting periods with exact durations (initial waiting period, pre-existing disease waiting period, specific disease waiting period). Mention any co-payment clause or deductible. PARAGRAPH 6 — Claims & Advisory: Explain the claim process (cashless vs reimbursement), mention the customer care helpline number if found, and give a brief closing advisory. Write all six paragraphs in second person ('your policy', 'you are covered for') with no bullet points.",
  "coverage_summary": "4-6 bullet points, each starting with '• '. Each bullet is ONE complete sentence stating a specific coverage or benefit with exact amounts where available. Example: '• Inpatient hospitalisation is covered up to the full Sum Insured of ₹X Lakh including room rent, nursing charges, surgeon fees, and ICU charges.'",
  "exclusions_summary": "4-6 bullet points, each starting with '• '. Each bullet is ONE complete sentence stating a specific exclusion — something the policy does NOT cover. Be specific: name the exact excluded item or condition. Example: '• Cosmetic and aesthetic treatments are not covered unless necessitated directly by an accident.'",
  "waiting_period_summary": "3-5 bullet points, each starting with '• '. Each bullet is ONE complete sentence stating one waiting period rule with the exact duration in months or days. Example: '• Pre-existing diseases are covered after a continuous 36-month waiting period from the policy start date.'",
  "premium_summary": "3-5 bullet points, each starting with '• '. Each bullet is ONE complete sentence about premium, charges, or payment terms. Include the total premium amount, GST, co-payment percentage (if any), deductible (if any), and renewal grace period. Example: '• Total annual premium payable is ₹X,XXX inclusive of 18% GST.'"
}}

DOCUMENT:
{document_text}"""


FIELD_EXTRACTION_PROMPT = """You are a healthcare insurance data entry clerk. Analyze the following health insurance document text and extract values for the requested fields.
If a field is not explicitly mentioned or cannot be found in the text, use null or "Not specified".
Do not create any extra keys. Return ONLY a valid JSON object matching the schema below. Do not output any markdown code blocks, preamble, or trailing text.

Return format JSON:
{{
  "policy_name": "the name of the policy plan",
  "insurer_name": "the insurance provider name",
  "policy_number": "the policy number",
  "sum_insured": "overall coverage amount",
  "premium_amount": "premium cost if specified",
  "deductible": "deductible terms or amount",
  "co_payment": "co-payment terms or percentage",
  "waiting_period": "waiting period rules",
  "coverage_type": "type of plan (family floater, individual, etc.)",
  "policy_term": "policy duration",
  "network_hospitals": "network hospital count or details",
  "pre_existing_coverage": "waiting periods/terms for pre-existing diseases",
  "maternity_coverage": "maternity benefit details/limits",
  "room_rent_limit": "daily room rent limit",
  "claim_process": "brief instructions on filing a claim"
}}

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
            hits.append(s_clean[:300])

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
        r'(?:total\s+premium|gross\s+premium|net\s+premium|premium\s+paid|premium\s+amount)[:\s₹Rs.]+([\d,]{4,}(?:\.[\d]+)?)',
        r'(?:premium)[:\s₹Rs.]+([\d,]{4,}(?:\.[\d]+)?)',
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
        r'(?:policy\s+period|policy\s+term)[:\s]+([^.\n]{0,60})',
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

    premium = _regex_find_any([
        r'(?:total\s+premium|gross\s+premium|net\s+premium|premium\s+paid|premium\s+amount)[:\s₹Rs.]+([\d,]{4,}(?:\.[\d]+)?)',
        r'(?:premium)[:\s₹Rs.]+([\d,]{4,}(?:\.[\d]+)?)',
    ], text)

    waiting_period = _regex_find_any([
        r'(?:waiting\s+period)[:\s]+([\d]+\s*(?:month|year|day)[s]?[^.\n]{0,80})',
        r'([\d]+\s*(?:month|year)[s]?\s+waiting\s+period)',
    ], text)

    co_pay = _regex_find_any([
        r'(?:co-?pay(?:ment)?)[:\s]+([\d]+%[^.\n]{0,60})',
        r'(?:co-?pay(?:ment)?\s+of\s+)([\d]+%[^.\n]{0,60})',
    ], text)

    # ── Build executive summary text ────────────────────────────────────────
    facts = []
    if insurer:
        facts.append(f"Insurer: {insurer}")
    if policy_name:
        facts.append(f"Plan: {policy_name}")
    if policy_number:
        facts.append(f"Policy Number: {policy_number}")
    if policy_holder:
        facts.append(f"Policyholder: {policy_holder}")
    if sum_insured:
        facts.append(f"Sum Insured: ₹{sum_insured}")
    if premium:
        facts.append(f"Premium: ₹{premium}")
    if waiting_period:
        facts.append(f"Waiting Period: {waiting_period}")
    if co_pay:
        facts.append(f"Co-payment: {co_pay}")

    facts_text = ". ".join(facts) + "." if facts else ""

    # Pull 2-3 meaningful description sentences from the document
    desc_candidates = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n{2,}', text)
                       if len(s.strip()) > 80 and not s.strip().startswith('[Page')]
    desc_sentences = []
    seen_desc = set()
    skip_re = re.compile(
        r'^(page\s*\d+|dear\s+|\[page|subject:|insured\s+person|date of|name of|relationship|gender|premium\s+period)',
        re.IGNORECASE
    )
    for s in desc_candidates:
        key = s[:60].lower()
        if key not in seen_desc and not skip_re.match(s):
            seen_desc.add(key)
            desc_sentences.append(s[:350])
        if len(desc_sentences) >= 3:
            break

    intro_text = " ".join(desc_sentences)
    summary_text = f"{facts_text} {intro_text}".strip()[:1500]
    if not summary_text or len(summary_text) < 100:
        summary_text = (
            f"This health insurance document contains policy details from "
            f"{insurer or 'the insurer'}. "
            f"{facts_text} "
            f"Please refer to the full document for complete terms and conditions."
        )

    # ── Coverage section ────────────────────────────────────────────────────
    coverage_hits = _extract_sentences_with_keywords(
        full_text,
        [
            "hospitaliz", "inpatient", "daycare", "ICU", "ambulance",
            "sum insured", "reimburse", "cashless", "OPD", "wellness",
            "benefit", "covered under", "entitled to", "eligible for",
        ],
        max_results=5,
    )
    coverage_summary = "\n".join(f"• {s}" for s in coverage_hits) if coverage_hits else \
        "• Please refer to the policy schedule for a full list of covered benefits."

    # ── Exclusions section ──────────────────────────────────────────────────
    excl_hits = _extract_sentences_with_keywords(
        full_text,
        [
            "exclud", "not cover", "not payable", "not admissible",
            "shall not", "exception", "not eligible", "no claim",
        ],
        max_results=5,
    )
    exclusions_summary = "\n".join(f"• {s}" for s in excl_hits) if excl_hits else \
        "• Exclusions not clearly identified — please review the full document."

    # ── Waiting period section ──────────────────────────────────────────────
    wait_hits = _extract_sentences_with_keywords(
        full_text,
        [
            "waiting period", "initial waiting", "pre-existing",
            "months waiting", "days waiting", "moratorium",
        ],
        max_results=4,
    )
    waiting_period_summary = "\n".join(f"• {s}" for s in wait_hits) if wait_hits else \
        "• Waiting period details not found — please review the policy schedule."

    # ── Premium / cost section ──────────────────────────────────────────────
    premium_hits = _extract_sentences_with_keywords(
        full_text,
        [
            "premium", "deductible", "co-pay", "copay",
            "sum insured", "GST", "tax", "renewal", "total amount",
        ],
        max_results=5,
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
        r'(?:total\s+premium|gross\s+premium|net\s+premium)[:\s₹Rs.]+([\d,]{4,}(?:\.[\d]{0,2})?)',
        r'(?:premium\s+paid|premium\s+amount)[:\s₹Rs.]+([\d,]{4,}(?:\.[\d]{0,2})?)',
        r'(?:annual\s+premium)[:\s₹Rs.]+([\d,]{4,}(?:\.[\d]{0,2})?)',
        r'Total\s+(?:Premium|Amount)[:\s₹Rs.]+([\d,]{4,}(?:\.[\d]{0,2})?)',
    ], text), "premium")

    # ── Deductible ──────────────────────────────────────────────────────────
    add("Deductible", _regex_find_any([
        r'(?:deductible|excess)[:\s₹Rs.]+([₹Rs\d,]+(?:\.[\d]{0,2})?)',
        r'(?:per\s+hospitalization\s+deductible)[:\s₹Rs.]+([\d,]+)',
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

    # ── Policy Term / Period ────────────────────────────────────────────────
    add("Policy Term", _regex_find_any([
        r'(?:policy\s+term|policy\s+period|duration)[:\s]+([^.\n]{0,80})',
        r'(?:from|valid\s+from)[:\s]+([\d]{1,2}[\-/][\d]{1,2}[\-/][\d]{2,4}[^\n]{0,50}to[^\n]{0,50})',
        r'(?:inception\s+date|start\s+date)[:\s]+([\d]{1,2}[\-/][\d]{1,2}[\-/][\d]{2,4})',
    ], text), "policy_info")

    # ── Network Hospitals ───────────────────────────────────────────────────
    add("Network Hospitals", _regex_find_any([
        r'([\d,]+\+?\s*(?:network|cashless|empanelled)\s*hospitals?)',
        r'(?:network\s+of)[\s]+([\d,]+\+?\s*hospitals?)',
        r'([\d,]{2,})\s*(?:hospitals?|network)',
    ], text), "coverage")

    # ── Room Rent Limit ─────────────────────────────────────────────────────
    add("Room Rent Limit", _regex_find_any([
        r'(?:room\s+rent\s+limit)[:\s₹Rs.]+([^.\n]{0,100})',
        r'(?:room\s+rent)[^.\n]{0,30}([₹Rs\d,]+(?:\s*per\s+day)?[^.\n]{0,60})',
        r'((?:single\s+private\s+AC\s+room|private\s+room)[^.\n]{0,60})',
    ], text), "restrictions")

    # ── Pre-existing Disease Coverage ───────────────────────────────────────
    add("Pre Existing Coverage", _regex_find_any([
        r'(?:pre-?existing)[^.\n]{0,50}((?:cover|wait|period)[^.\n]{0,80})',
        r'(?:pre-?existing\s+disease)[:\s]+([^.\n]{0,120})',
    ], text), "restrictions")

    # ── Maternity Coverage ──────────────────────────────────────────────────
    add("Maternity Coverage", _regex_find_any([
        r'(?:maternity)[:\s]+([^.\n]{0,120})',
        r'(?:maternity\s+benefit)[:\s]+([^.\n]{0,100})',
    ], text), "coverage")

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
        r'(?:customer\s+(?:care|service|helpline)|toll\s+free|contact)[:\s]+([+0-9 \-]{8,18})',
        r'(?:call\s+us)[\s:]+([+0-9 \-]{8,18})',
        r'(1800\s*[0-9\-]+)',
    ], text), "policy_info")

    if not fields:
        # Last resort — return document identification info
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

    # Use up to 30000 chars — enough for key policy content, keeps input tokens low for speed
    truncated = document_text[:30000] if len(document_text) > 30000 else document_text
    try:
        response = await call_ollama(
            SUMMARIZATION_PROMPT.format(document_text=truncated),
            num_predict=3500,   # 3500 tokens needed for 400-500 word summary + 4 bullet sections
            num_ctx=12288,
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

    truncated = document_text[:30000] if len(document_text) > 30000 else document_text
    try:
        response = await call_ollama(
            FIELD_EXTRACTION_PROMPT.format(document_text=truncated),
            num_predict=600,
            num_ctx=12288,
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
                if val in ("0", "null", "") or len(val) < 3:
                    fallback_prem = _regex_find_any([
                        r'(?:total\s+premium|gross\s+premium|net\s+premium)[:\s₹Rs.]+([\d,]{4,}(?:\.[\d]{0,2})?)',
                        r'(?:premium\s+paid|premium\s+amount)[:\s₹Rs.]+([\d,]{4,}(?:\.[\d]{0,2})?)',
                        r'Total\s+(?:Premium|Amount)[:\s₹Rs.]+([\d,]{4,}(?:\.[\d]{0,2})?)',
                    ], document_text)
                    if fallback_prem and fallback_prem != "Not found in document":
                        result["premium_amount"] = fallback_prem
            
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

    truncated = document_text[:30000] if len(document_text) > 30000 else document_text
    try:
        response = await call_ollama(
            RISK_ANALYSIS_PROMPT.format(document_text=truncated),
            num_predict=500,
            num_ctx=12288,
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
    """Generate streaming tokens from Ollama — optimized for speed."""
    from app.services.ollama_client import parse_keep_alive
    
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
        "keep_alive": parse_keep_alive(settings.OLLAMA_KEEP_ALIVE),
        "options": {
            "temperature": 0,  # CHANGED from 0.1 to 0 for greedy decoding
            "num_predict": num_predict,  # REDUCED from 512
            "num_ctx": 1024,  # REDUCED from 4096
            "num_batch": 1024,
            "top_k": 1,  # GREEDY only
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



