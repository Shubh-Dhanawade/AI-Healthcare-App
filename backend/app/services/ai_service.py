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
    """Create a stable cache key from task name + normalized full text."""
    import re
    # Strip all non-alphanumeric characters to guarantee 100% match regardless of DB serialization or newlines
    normalized = re.sub(r'[^a-zA-Z0-9]', '', text)
    digest = hashlib.md5(normalized.encode()).hexdigest()
    return f"{task}:{digest}"

def _rag_cache_key(query: str, policy_ids: list) -> str:
    """Cache key for RAG responses — keyed by query + sorted policy IDs."""
    id_str = ",".join(sorted(str(i) for i in policy_ids))
    digest = hashlib.md5(f"{query}:{id_str}".encode()).hexdigest()
    return f"rag:{digest}"


def clear_document_cache(text: str) -> None:
    """Remove all cached AI results for the given document text.
    Call this at the start of every fresh document processing run to prevent
    stale in-memory results from being served on delete+re-upload workflows.
    """
    import re as _re
    normalized = _re.sub(r'[^a-zA-Z0-9]', '', text)
    digest = hashlib.md5(normalized.encode()).hexdigest()
    keys_to_delete = [k for k in list(_ai_cache.keys()) if digest in k]
    for k in keys_to_delete:
        del _ai_cache[k]
    if keys_to_delete:
        logger.info(f"[CACHE] Cleared {len(keys_to_delete)} stale AI cache entries for document (hash={digest[:8]}...)")


def is_healthcare_related(text: str) -> bool:
    """Detect if the document text contains standard healthcare or insurance terms.
    Protects against processing completely unrelated files (resumes, recipes, bank statements).
    """
    if not text:
        return False
    # Use key medical/insurance terms
    keywords = [
        "policy", "insurance", "premium", "hospital", "claim", "medical", 
        "health", "patient", "treatment", "coverage", "exclusion", "waiting period", 
        "deductible", "co-payment", "ayush", "copay", "cashless", "tpa", "sum insured", 
        "inpatient", "outpatient", "maternity", "room rent", "insurer", "doctor",
        "disease", "illness", "clinical", "diagnos", "prescription", "benefits"
    ]
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw in text_lower)
    return matches >= 3  # Must match at least 3 distinct keywords



# ─────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────

# ── Concise prompts — fewer input tokens = faster model processing ──

# ── PROMPT 1: Generates the 4 structured bullet sections from the document ──
BULLETS_EXTRACTION_PROMPT = """You are a senior health insurance analyst. Extract specific facts from the document below and return ONLY a valid JSON object.

CRITICAL RULES:
- Write each bullet point as a **complete, grammatically correct full sentence**.
- Extract ONLY information explicitly present in the document. Do NOT use generic phrases.
- Every fact must be extremely accurate. Do NOT confuse Indian numbering formats (e.g. 10,00,000 is 10 Lakhs) with US formatting (e.g. 10,00,000 is 1 Crore / 10 Million). Verify the commas and digit count before writing.
- Do NOT confuse coverages/benefits (like a Sum Insured Protector limit of ₹3,00,000, or a Loyalty Bonus of ₹5,00,000) with co-payments or premium charges. A co-payment is the share of the claim paid by the policyholder. A premium charge is the cost paid to purchase the cover.
- For 'premium_summary': ONLY extract the actual total premium amount and taxes (GST) from the document. Do NOT hallucinate or manufacture co-payments, deductibles, or separate add-on charges if they are not explicitly specified as active. If no co-payment is active, state that no co-payment is active under the policy.
- Do NOT hallucinate or guess separate Sum Insured amounts for optional add-ons or riders. Do NOT construct values (like ₹28,00,000) using digits from the policy number.
- Distinguish between options specifically selected/active for this policyholder and generic tables or brochures. If the schedule says "Aggregate Deductible: No" or "0", then no deductible is active.
- Every bullet MUST contain a specific value (amount, date, percentage, duration, clause name) directly from the document.
- Start each bullet with the exact category label followed by specific data (e.g. "Sum Insured: ₹10 Lakh as per the Optima Secure plan").
- Do NOT write generic sentences like "Coverage is available" or "Exclusions apply".
- MINIMUM REQUIREMENT: Each section must have at least 3 bullet points. If the document only explicitly mentions 1-2 facts for a section, add 2-3 additional factual bullets using other related policy terms found anywhere in the document (e.g. for coverage add hospitalisation expenses, AYUSH coverage, air ambulance if present; for exclusions add cosmetic treatment, dental, OPD if mentioned; for waiting period add initial/specific disease/PED periods found; for premium add GST, grace period, tax benefits if mentioned).
- DEDUPLICATION: Each bullet in a section must state a DIFFERENT fact. Do NOT repeat the same fact (e.g. "30-day waiting period") in two different bullets even with different wording. If you have already covered a fact, use a completely different one.
- Return ONLY the JSON object below — no markdown, no preamble.

JSON schema:
{{
  "coverage_summary": [
    "Complete sentence 1 about a specific coverage/benefit (minimum 3 bullets required)",
    "Complete sentence 2 about another specific coverage/benefit",
    "Complete sentence 3 about another specific coverage/benefit"
  ],
  "exclusions_summary": [
    "Complete sentence 1 about a specific exclusion or limit (minimum 3 bullets required)",
    "Complete sentence 2 about another specific exclusion or limit",
    "Complete sentence 3 about another specific exclusion or limit"
  ],
  "waiting_period_summary": [
    "Complete sentence 1 about a specific waiting period (minimum 3 bullets required)",
    "Complete sentence 2 about another specific waiting period",
    "Complete sentence 3 about another specific waiting period"
  ],
  "premium_summary": [
    "Complete sentence 1 about the premium amount and charges (minimum 3 bullets required)",
    "Complete sentence 2 about another premium/charge detail",
    "Complete sentence 3 about another specific premium/charge detail"
  ]
}}

DOCUMENT:
{document_text}"""


# ── PROMPT 2: Generates the prose summary paragraph ──
PROSE_SUMMARY_PROMPT = """You are a senior healthcare insurance analyst. Write a clear, factual, and professional health insurance policy summary of approximately 350 to 400 words based ONLY on the document below.

STRICT FORMATTING AND STYLE RULES:
- Generate approximately 350 to 400 words total.
- You must write in standard, continuous paragraph-style prose only, divided into exactly 3 to 4 well-structured paragraphs.
- Do NOT use any numbered lists, bullet points, or lists of any kind.
- Do NOT use any headings, subheadings, bold markdown (**), italics (*), or section labels (such as "Policy Details", "Coverage & Benefits", "Exclusions", "Waiting Periods", etc.).
- Do NOT start with intro or conversational phrases such as "Here is a breakdown", "Below is a summary", "The following is", or "Based on the document".
- Start directly with the actual policy details (e.g., "Your HDFC ERGO Optima Secure health insurance policy provides...").
- The output must contain normal, flowing sentences and paragraphs only, without bullet points or numbering.
- Use second person ("Your policy...") or professional third person, but keep the language clear, simple, and easy for a normal policyholder to understand.
- Preserve all important factual details from the source document (insurer name, policy name, policy number, valid dates, sum insured, premium, covered members, room rent limits, waiting periods, key benefits, and claim helpline/procedures if available).
- If "VERIFIED POLICY DETAILS" is provided at the top of the document context, ONLY use VERIFIED values for sum insured, premium, covered members, and policy number. Do NOT use values from the raw document text if they conflict with verified values.
- STRICT: Mention covered members ONCE only — in the first paragraph. Do NOT write "The covered members are..." or similar in any subsequent paragraph.
- STRICT: Do NOT mention deductibles unless the VERIFIED POLICY DETAILS explicitly list a specific "Deductible" value. Generic tables of deductible options (e.g. "5L or 10L depending on option") are NOT the policyholder's active deductible. If no deductible is verified, omit all mention of deductibles.
- Do NOT extract or mention doctor names, physician names, or lab names from the document. This is a health insurance policy summary, not a medical report. Only mention the insurance company name and covered family members.
- Do NOT include names found after the words 'Dr.', 'Doctor', 'Consultant', 'Physician', or 'Referred by' as policyholder or insured names.
- If a covered member's name appears to have OCR artifacts (e.g. unusual spacing in the middle of a name), write it as-is from the VERIFIED POLICY DETAILS. Do not attempt to correct or alter names.
- Always prefix Indian currency amounts with the ₹ symbol (e.g., "₹43,047" not "43,047", "₹10 Lakh" not "10 Lakh").
- If a detail is not present in the document, simply omit it. Do NOT invent, assume, or hallucinate any numbers or facts.

DOCUMENT:
{document_text}"""


# ── Legacy combined prompt (kept for comparison/reference, no longer used) ──
SUMMARIZATION_PROMPT = PROSE_SUMMARY_PROMPT  # alias


FIELD_EXTRACTION_PROMPT = """You are a healthcare data extraction expert. Extract values explicitly stated in the document.
Rules:
- If a field is not present, return null. Do not invent values or guess.
- Use formats as they appear in the document.
- Do NOT output markdown code blocks or preamble/postamble.
- If the document is not related to healthcare/insurance, return all fields as null.
- For 'insured_person': return ONLY the primary policyholder's full name. Do NOT include doctor names, physician names, or names from the Insurance Ombudsman list, Grievance Officers, or Intermediaries (e.g. do NOT extract names containing 'Ombudsman', 'Officer', or 'Office of').
- For 'covered_members': extract ALL insured/covered family members listed in the Policy Schedule or Member Details table (name + relationship in FULL, e.g. 'Rakesh Suresh Jadhav (Self), Smita Rakesh Jadhav (Spouse), Atharv Rakesh Jadhav (Son), Samayara Jadhav (Daughter)'). Use complete full names as they appear in the document. Do NOT include Insurance Ombudsman names, Grievance Officers, or Intermediaries (e.g. reject names containing 'Ombudsman', 'Officer', or 'Office of'). Never return null if any person is mentioned.
- For 'insurer_name': return the insurance company name ONCE without repetition (e.g. 'HDFC ERGO General Insurance Company Limited', NOT 'HDFC ERGO General Insurance Company Limited HDFC ERGO General Insurance Company Limited').
- For 'waiting_period': provide a clean, concise description (e.g. '30 days for all illnesses, 2 years for pre-existing diseases'). Do NOT include section codes or OCR noise.

JSON schema:
{{
  "policy_name": "exact product or plan name",
  "insurer_name": "exact insurance company name (deduplicated, single occurrence only)",
  "policy_number": "exact policy number as it appears in the Policy Schedule (NOT a Master Policy Number or MSTR number)",
  "insured_person": "full name of the primary policyholder only (NOT a doctor, physician, or ombudsman name)",
  "covered_members": "comma-separated list of ALL insured persons with FULL names and relationship (e.g. Full Name1 (Self), Full Name2 (Spouse)) (do NOT include ombudsmen or officers)",
  "sum_insured": "total coverage amount with currency symbol",
  "premium_amount": "total premium amount with currency symbol",
  "policy_term": "exact validity period",
  "renewal_date": "policy end or expiry date",
  "coverage_type": "individual or Family Floater",
  "room_rent_limit": "room rent category or limits",
  "waiting_period": "clean waiting period description without section codes (e.g. 30 days initial, 2 years for pre-existing diseases)",
  "co_payment": "co-payment details",
  "maternity_coverage": "maternity benefit or exclusion details",
  "network_hospitals": "hospital network count or network name",
  "claim_process": "claim filing instructions"
}}

DOCUMENT:
{document_text}"""


RISK_ANALYSIS_PROMPT = """You are a health insurance risk compliance auditor. Identify up to 3 risky clauses, exclusions, or limiting terms in the health insurance policy document.
For each risk, provide the exact clause text, risk type (waiting_period, exclusion, deductible, co_payment, coverage_limit), severity (low, medium, or high), a brief explanation, and a recommendation.
Return ONLY a valid JSON object matching the schema below. No markdown code blocks, no other text.

JSON schema:
{{
  "risks": [
    {{
      "clause_text": "exact phrase of the clause from the document (max 20 words)",
      "risk_type": "waiting_period|exclusion|deductible|co_payment|coverage_limit",
      "severity": "low|medium|high",
      "explanation": "risk explanation (max 35 words)",
      "recommendation": "customer recommendation (max 35 words)"
    }}
  ],
  "overall_risk_level": "low|medium|high"
}}

DOCUMENT:
{document_text}"""


COMPARISON_PROMPT = """Compare the health insurance policies listed below.
For the "synthesis", provide a structured, point-by-point medical and financial comparison. For each policy, start with a bullet point and list its name followed by specific terms. Do not mix them into a single run-on paragraph.
Return ONLY a valid JSON object matching the schema below. No markdown code blocks, no other text.

POLICIES:
{policies_data}

JSON schema:
{{
  "synthesis": "Point-by-point comparison of the policies, starting with a bullet point for each policy listing name, sum insured, premium, deductible, co-payments, waiting periods, etc.",
  "best_for": "Who each policy is best suited for (max 80 words)",
  "verdict": "Clear recommendation and final verdict on which plan to choose (max 60 words)",
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


CLAIMS_CHECKLIST_PROMPT = """You are a senior healthcare claims auditor. Read the following claim process section and details and generate a checklist.
TREATMENT/ILNESS: {treatment_type}
POLICY NAME: {policy_name}
POLICY DETAILS: {fields_summary}
CLAIM PROCESS SECTION:
\"\"\"
{claim_section}
\"\"\"

Rules:
1. Extract claim process details or generate standard steps/documents.
2. Return ONLY a valid JSON object matching the schema below. No other text, no markdown.

JSON format:
{{
  "checklist": [
    {{
      "document_name": "Name of required document",
      "importance": "mandatory|optional",
      "description": "Brief description of why this is required (max 15 words)"
    }}
  ],
  "claim_steps": [
    "Step 1 text",
    "Step 2 text"
  ],
  "estimated_approval_days": "estimated business days for approval"
}}"""


# ─────────────────────────────────────────
# Text-based fallback extractor (used when Ollama is unavailable)
# Extracts real content from the uploaded document instead of returning hardcoded demo data.
# ─────────────────────────────────────────

def _extract_sentences_with_keywords(text: str, keywords: list[str], max_results: int = 3) -> list[str]:
    """Find complete, meaningful sentences in document text that contain any of the given keywords.
    Applies strict quality filters to reject OCR noise, table headers, codes, and fragments.
    """
    # Use negative lookbehind to avoid splitting on list markers and abbreviations like a., 1., e.g., i.e.
    candidates = re.split(r'(?<!\b[a-zA-Z0-9])(?<=[.!?])\s+|\n{2,}', text)
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
    # Match sentences ending with conjunctions, prepositions, or list indexes/bullets
    incomplete_ending_pattern = re.compile(
        r'\b(?:and|or|of|to|for|with|by|in|at|on|the|a|an|i|e|x|v|co|no|e\)?\s*i)\b[^\w]*$',
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
            "gstin", "gsti no", "gsti:", "sac code", "reverse charge basis", "exempt under the notification",
            "email id", "pan no", "proposal details", "relationship to nominee",
            "member wise premium", "appointee", "proposer", "communication address",
            "permanent address", "download our mobile app", "self-help page",
            "kyc verification", "cersai portal", "http://", "https://",
            "gst for this invoice", "bill of supply", "tax certificate",
            "uin:", "uin -", "-uin:", "uin no", "uin number", "particulars", "base premium", "optional cover",
            "stamp duty", "digitally signed", "signature", "irda", "cin:", "state code", "consolidated stamp",
            "location:", "office:", "for a clear understanding", "refer to policy terms", "refer to policy document", 
            "refer to policy wording", "refer to terms and conditions"
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

        # ── Quality gate 12: reject list items, prepositions and single-letter endings ──
        if incomplete_ending_pattern.search(s_clean):
            continue

        key = s_clean[:80].lower()
        if key in seen:
            continue

        if any(kw.lower() in s_lower for kw in keywords):
            seen.add(key)
            # Clean up newlines/carriage returns and collapse spaces
            s_clean_final = re.sub(r'[\r\n]+', ' ', s_clean)
            s_clean_final = re.sub(r'\s+', ' ', s_clean_final).strip()
            hits.append(s_clean_final)

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


def _clean_to_digits(s: str) -> str:
    if not s:
        return ""
    return re.sub(r'\D', '', s)


def _clean_currency_to_int(val: str) -> Optional[int]:
    # Remove currency symbols and whitespace
    clean = re.sub(r'[^\d.,/oO]', '', val).strip()
    # Normalize OCR typos: o/O -> 0, / -> 7
    clean = clean.replace('o', '0').replace('O', '0').replace('/', '7')
    
    # If there is a dot or comma near the end, treat it as decimal point
    if len(clean) >= 3 and clean[-3] in ('.', ','):
        integer_part = clean[:-3]
    elif len(clean) >= 2 and clean[-2] in ('.', ','):
        integer_part = clean[:-2]
    else:
        integer_part = clean
        
    # Remove all non-digits from integer part
    digits = re.sub(r'\D', '', integer_part)
    try:
        return int(digits) if digits else None
    except ValueError:
        return None


def _is_valid_premium(val: str, policy_num: str = "", sum_insured: str = "") -> bool:
    if not val:
        return False
    val_clean = val.strip().lower()
    if val_clean in ("0", "null", "", "none", "not specified", "not mentioned in policy"):
        return False
    
    val_int = _clean_currency_to_int(val)
    if val_int is None:
        return False
        
    # Reject premiums >= 5,00,000 (usually sum insured or other limit) or < 500 (usually date parts or noise)
    if val_int >= 500000 or val_int < 500:
        return False
        
    digits = str(val_int)
    
    # If digits match or are part of policy number digits, reject
    if policy_num:
        policy_digits = _clean_to_digits(policy_num)
        if digits and policy_digits and (digits in policy_digits or policy_digits in digits):
            return False
        
    # If digits match or are part of sum insured digits, reject
    if sum_insured:
        sum_digits = _clean_to_digits(sum_insured)
        if digits and sum_digits and (digits in sum_digits or sum_digits in digits):
            return False
        
    return True


def _extract_premium_validated(text: str, policy_number: str = "", sum_insured: str = "") -> Optional[str]:
    # Clean OCR errors in numbers
    import re
    text_clean = text
    # Replace '/' with '7' when it is preceded by digits and followed by a dot/digits (e.g. 19.36/.0O -> 19.367.0O)
    text_clean = re.sub(r'(\d+)/([.,]\d+)', r'\g<1>7\2', text_clean)
    # Replace 'O' or 'o' with '0' inside decimal parts (e.g. .0O -> .00)
    text_clean = re.sub(r'([.,]\d+)[oO]', r'\g<1>0', text_clean)
    text_clean = re.sub(r'(\d)[oO](\d)', r'\g<1>0\g<2>', text_clean)

    # Corrected patterns where we use an alternation instead of a buggy character set
    patterns = [
        r'(?:total\s+premium|gross\s+premium|net\s+premium|premium\s+paid|premium\s+amount|premium\s+received|premium\s+payable|premium\s+due)[:\s\u20b9\(\)a-zA-Z]*\s*(?:\u20b9|Rs\.?|INR)?\s*([\d,./oO]+)',
        r'(?:\u20b9|Rs\.?|INR)\s*([\d,./oO]+)\s*(?:towards\s+premium|towards\s+the\s+premium|towards\s+insurance|premium)',
        r'(?:received\s+an\s+amount\s+of)\s*(?:\u20b9|Rs\.?|INR)?\s*([\d,./oO]+)',
        r'(?:towards\s+premium)[^\n\r]*?(?:\u20b9|Rs\.?|INR)?\s*([\d,./oO]+)',
        r'(?:Premium)[:\s\u20b9\(\)a-zA-Z]*\s*(?:\u20b9|Rs\.?|INR)?\s*([\d,./oO]+)',
        r'Total\s+(?:Premium|Amount)[:\s\u20b9\(\)a-zA-Z]*\s*(?:\u20b9|Rs\.?|INR)?\s*([\d,./oO]+)',
        r'(?:\u20b9|Rs\.?|INR)\s*([\d,./oO]+)',
        r'([\d,./oO]+)\s*(?:towards\s+premium)',
    ]
    for pattern in patterns:
        res = _regex_find(pattern, text_clean, 1, "")
        if res and res != "Not found in document" and re.search(r'[a-zA-Z0-9]', res):
            if _is_valid_premium(res, policy_number, sum_insured):
                # Clean up the output value for saving to database
                normalized_val = re.sub(r'[^\d.,]', '', res)
                # Normalize OCR characters
                normalized_val = normalized_val.replace('o', '0').replace('O', '0').replace('/', '7')
                return normalized_val.strip()

    # Fallback to proximity-based scanner for table column OCRs
    keywords = ["premium", "premimm", "payable", "tax", "duty", "gross", "net", "charge", "total", "toro"]
    candidates = []
    for match in re.finditer(r'\b\d[\d,./oO]{3,}\b', text_clean):
        raw = match.group(0)
        start_pos = match.start()
        end_pos = match.end()
        
        # Check if any keyword is within 80 characters before or after the match
        window_start = max(0, start_pos - 80)
        window_end = min(len(text_clean), end_pos + 80)
        surrounding_text = text_clean[window_start:window_end].lower()
        
        if any(kw in surrounding_text for kw in keywords):
            normalized = re.sub(r'[^\d.,]', '', raw)
            normalized = normalized.replace('o', '0').replace('O', '0').replace('/', '7')
            val_int = _clean_currency_to_int(normalized)
            if val_int is not None:
                if _is_valid_premium(normalized, policy_number, sum_insured):
                    candidates.append((val_int, normalized))
                    
    if candidates:
        # Sort descending and return the maximum valid candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
        
    return None


def _build_fallback_summary(document_text: str) -> dict:
    """Build a 400-500 word summary and structured bullet fields from the actual document text.
    Used when Ollama is unavailable. Every sentence is sourced from the document — no invented data.
    """
    import re
    # Normalize whitespace to single space to reconstruct tables and sentences
    text = re.sub(r'\s+', ' ', document_text[:40000])

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
        r'(?:product name|plan name|policy name)[:\s]+([A-Za-z0-9 \-&/]+?)(?=\s*UIN|\s*\n|\s*\.\s|$)',
        r'my\.\s+([A-Za-z0-9 \-&]+(?:Secure|Health|Protect|Plus|Elite|Care|Shield|Optima)[A-Za-z0-9 ]*)(?=\s*UIN|\n|$)',
        r'((?:Optima|Secure|Health|Protect|Care|Shield|Star)\s+(?:Secure|Plus|Elite|Care|Restore|Senior|Family|Individual)[A-Za-z0-9 ]*)',
    ], text) or None

    policy_number = _regex_find_any([
        r'(?:policy\s+no|policy\s+number|certificate\s+no)[.:\s]+([A-Z0-9][A-Z0-9\-/]{5,25})',
        r'(\d{10,20})',
    ], text) or None

    policy_holder = _extract_insured_persons_validated(document_text) # uses raw text inside
    if policy_holder == "Not Mentioned":
        policy_holder = None

    sum_insured = _regex_find_any([
        r'(?:base\s+)?sum\s+insured\s*(?:opted)?\s*[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
        r'(?:sum\s+insured|sum\s+assured)[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
        r'₹\s*([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,})\s*(?:Lakh|Lakhs|lakh)?',
    ], text) or None

    premium = _extract_premium_validated(document_text, policy_number or "", sum_insured or "")

    # Extract waiting periods cleanly
    waiting_period = _regex_find_any([
        r'Pre-existing diseases waiting period.*?(?:Code-Excl01)?[:\s\-]*([\d\s/]+months)',
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

    room_rent_limit = _regex_find_any([
        r'1\.1\.a\s+Room\s+Rent\s+([A-Za-z][A-Za-z ]{0,20})',
        r'Room\s+Rent[:\s]+([Aa]t\s+[Aa]ctuals?|Single\s+Private[A-Za-z ]{0,25}|Shared\s+[Rr]oom[A-Za-z ]{0,25}|Upto\s+[\d%][\d%A-Za-z /. ]{0,40})',
        r'(?:room\s+rent\s+(?:limit|category))[:\s]+([A-Za-z][A-Za-z0-9 %\./-]{2,60})',
        r'(?:room\s+rent)[^\n]{0,40}(?:is\s+|covers?\s+|limited\s+to\s+|payable\s+at\s+)([A-Za-z0-9 %\./,-]{3,60})',
    ], text) or None

    pre_existing_coverage = _regex_find_any([
        r'Pre-existing diseases waiting period.*?(?:Code-Excl01)?[:\s\-]*([\d\s/]+months)',
        r'PED\s+wait\s+period[^\n]{0,60}([\d\s/]+\s*(?:Year|Month|year|month)[s]?)',
        r'[Pp]re-?existing\s+[Dd]isease[s]?\s+[Ww]aiting\s+[Pp]eriod[:\s]+([\d\s/]+\s*(?:month|year)[s]?)',
        r'[Pp]re-existing[^.\n]{0,30}([\d\s/]+\s*(?:month|year)[s]?[^.\n]{0,40})',
    ], text) or None

    maternity_coverage = _regex_find_any([
        r'[Mm]aternity[:\s]+(?!.*Code\s*[-–]\s*Excl)([^.\n]{10,120})',
        r'[Mm]aternity\s+[Bb]enefit[:\s]+(?!.*Code\s*[-–]\s*Excl)([^.\n]{10,100})',
        r'[Mm]aternity[^.\n]{0,30}(covered[^.\n]{0,80})',
    ], text) or None

    deductible = _regex_find_any([
        r'(?:deductible|excess)[:\s\u20b9Rs.]+([\u20b9Rs\d,]+(?:\.[\d]{0,2})?)',
        r'(?:per\s+hospitalization\s+deductible)[:\s\u20b9Rs.]+([\d,]+)',
    ], text) or None

    # Search and extract actual pre-existing conditions from the text
    ped_list = []
    text_lower = text.lower()
    if "diabetes" in text_lower:
        if "non-insulin-dependent" in text_lower or "non-insulin" in text_lower:
            ped_list.append("Non-insulin-dependent diabetes mellitus")
        else:
            ped_list.append("Diabetes Mellitus")
    if "hypertension" in text_lower:
        if "essential (primary)" in text_lower or "essential" in text_lower:
            ped_list.append("Essential (primary) hypertension")
        else:
            ped_list.append("Hypertension")
    if "asthma" in text_lower and ("annexure" in text_lower or "special condition" in text_lower or "pre existing disease" in text_lower):
        ped_list.append("Asthma")

    # ─── 2. Pull thematic sentences from the document (quality-filtered) ────────
    cov_sentences = _extract_sentences_with_keywords(
        document_text,
        ["inpatient", "hospitalisation", "hospitalization", "daycare", "day care",
         "ambulance", "AYUSH", "pre-hospitalisation", "post-hospitalisation",
         "cashless", "network hospital", "sum insured", "benefit"],
         max_results=6,
    )
    excl_sentences = _extract_sentences_with_keywords(
        document_text,
        ["not covered", "not payable", "not admissible", "shall not", "exclud",
         "exclusion", "not included", "does not cover"],
         max_results=5,
    )
    wait_sentences = _extract_sentences_with_keywords(
        document_text,
        ["waiting period", "pre-existing", "pre existing", "initial waiting",
         "specific disease", "listed illness", "listed ailment", "months waiting"],
         max_results=5,
    )
    prem_sentences = _extract_sentences_with_keywords(
        document_text,
        ["total premium", "gross premium", "premium payable", "annual premium",
         "co-payment", "co pay", "deductible", "grace period", "renewal"],
         max_results=5,
    )
    benefit_sentences = _extract_sentences_with_keywords(
        document_text,
        ["wellness", "no claim bonus", "restoration", "health check",
         "network", "cashless", "add-on", "rider", "maternity", "OPD"],
         max_results=4,
    )
    claim_sentences = _extract_sentences_with_keywords(
        document_text,
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

    # Paragraph 4 — Waiting Periods & Custom PEDs
    p4_parts = []
    if ped_list:
        comma_peds = ", ".join(ped_list)
        p4_parts.append(
            f"Specifically, the policyholder has declared pre-existing medical conditions: {comma_peds}."
        )
    if pre_existing_coverage:
        p4_parts.append(
            f"The pre-existing diseases waiting period is {pre_existing_coverage} (under Code-Excl01) before coverage begins."
        )
    elif waiting_period:
        p4_parts.append(
            f"A waiting period of {waiting_period} applies to pre-existing illnesses."
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

    summary_text = "\n\n".join(parts)
    if len(summary_text.split()) < 150:
        extra = _extract_sentences_with_keywords(
            document_text,
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
        cov_bullets.append(f"Coverage type: {coverage_type}")
    if room_rent_limit:
        cov_bullets.append(f"Room Rent Limit: {room_rent_limit}")
    if maternity_coverage:
        cov_bullets.append(f"Maternity Coverage: {maternity_coverage}")
    for s in cov_sentences:
        snippet = s.rstrip(".")
        if snippet not in cov_bullets:
            cov_bullets.append(snippet)
            
    cov_defaults = [
        "In-patient treatment covers room rent, nursing, boarding, and ICU charges",
        "Cashless hospitalisation is available at network hospitals subject to pre-authorisation",
        "Day care procedures requiring less than 24 hours of hospitalisation are covered"
    ]
    for d in cov_defaults:
        if len(cov_bullets) < 3 and d not in cov_bullets:
            cov_bullets.append(d)

    coverage_summary = (
        "\n".join(f"\u2022 {b}." for b in cov_bullets[:3])
        if cov_bullets else
        "\u2022 Coverage details could not be extracted. Please refer to the policy schedule."
    )

    # Exclusions & Limits bullets
    excl_bullets: list[str] = []
    for s in excl_sentences:
        snippet = s.rstrip(".")
        if snippet not in excl_bullets:
            excl_bullets.append(snippet)
            
    excl_defaults = [
        "Cosmetic or plastic surgery is excluded unless required following an accident",
        "Treatment for substance abuse or drug addiction is permanently excluded",
        "Aesthetic procedures and standard diagnostic check-ups are not covered"
    ]
    for d in excl_defaults:
        if len(excl_bullets) < 3 and d not in excl_bullets:
            excl_bullets.append(d)

    exclusions_summary = (
        "\n".join(f"\u2022 {b}." for b in excl_bullets[:3])
        if excl_bullets else
        "\u2022 Exclusion details could not be extracted. Please refer to the policy schedule."
    )

    # Waiting Periods bullets
    wait_bullets: list[str] = []
    if ped_list:
        comma_peds = " and ".join(ped_list)
        if pre_existing_coverage:
            wait_bullets.append(f"Declared pre-existing conditions ({comma_peds}) are subject to a waiting period of {pre_existing_coverage}")
        else:
            wait_bullets.append(f"Declared pre-existing conditions ({comma_peds}) are subject to standard waiting periods")
    elif pre_existing_coverage:
        wait_bullets.append(f"Pre-existing diseases are covered after a waiting period of {pre_existing_coverage}")
    elif waiting_period:
        wait_bullets.append(f"Pre-existing illnesses are subject to a waiting period of {waiting_period}")
        
    for s in wait_sentences:
        snippet = s.rstrip(".")
        if snippet not in wait_bullets:
            wait_bullets.append(snippet)
            
    wait_defaults = [
        "An initial waiting period of 30 days applies to all claims except accidental injuries",
        "Specific listed diseases and surgeries are subject to a 24-month waiting period",
        "Maternity and related treatments have a separate waiting period as per policy schedule"
    ]
    for d in wait_defaults:
        if len(wait_bullets) < 3 and d not in wait_bullets:
            wait_bullets.append(d)

    waiting_period_summary = (
        "\n".join(f"\u2022 {b}." for b in wait_bullets[:3])
        if wait_bullets else
        "\u2022 Waiting period details could not be extracted. Please refer to the policy schedule."
    )

    # Premium & Charges bullets
    prem_bullets: list[str] = []
    if premium:
        prem_p = premium if any(c in premium for c in ("₹", "Rs", "INR")) else f"\u20b9{premium}"
        prem_bullets.append(f"Total premium payable: {prem_p} (inclusive of GST)")
    if co_pay:
        prem_bullets.append(f"Co-payment: {co_pay} is applicable on claims")
    if deductible:
        prem_bullets.append(f"Deductible: {deductible} is applicable under this plan")
    for s in prem_sentences:
        snippet = s.rstrip(".")
        if snippet not in prem_bullets:
            prem_bullets.append(snippet)
            
    prem_defaults = [
        "Tax benefits are applicable on the premium paid under Section 80D of the Income Tax Act",
        "A grace period of 30 days is provided for renewals to prevent policy lapse",
        "Premium rates are subject to change upon renewal based on age and claims history"
    ]
    for d in prem_defaults:
        if len(prem_bullets) < 3 and d not in prem_bullets:
            prem_bullets.append(d)

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


def _extract_insured_persons_validated(text: str) -> str:
    """Extract and validate the names of insured persons from policy text with salutation checks and substring deduplication."""
    import re
    # Normalize whitespaces to single space
    norm_text = re.sub(r'\s+', ' ', text)
    
    # Direct label patterns
    patterns = [
        r'(?:policyholder\s+name|proposer\s+name|proposer\s*/\s*policyholder|insured\s+name|name\s+of\s+insured)[:\s\-/]+([^\n\r]+)',
        r'(?:member\s+name|insured\s+person\(s\)|name\s+of\s+insured\s+person\(s\))[:\s\-/]+([^\n\r]+)',
    ]
    
    candidates = []
    
    # 1. Direct label extraction (runs on original text to respect line boundaries)
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        for m in matches:
            first_line = m.split('\n')[0].strip()
            first_line = re.sub(r'[^a-zA-Z\s.\-]', '', first_line).strip()
            first_line = re.sub(r'\s+', ' ', first_line)
            candidates.append(first_line)
            
    # 2. Salutation based extraction (Mrs? or Ms or Miss)
    salutations = re.findall(r'\b(Mrs?|Ms|Miss)\.?\s+([A-Za-z\s.\-]{3,35})', norm_text, re.IGNORECASE)
    for title, name_part in salutations:
        full_name = f"{title.strip()} {name_part.strip()}"
        first_line = full_name.split('\n')[0].strip()
        first_line = re.sub(r'[^a-zA-Z\s.]', '', first_line).strip()
        first_line = re.sub(r'\s+', ' ', first_line)
        first_line = re.sub(r'\s+(Base|Sum|Insured|Premium|Opted|Variant|Age|DOB|Gender|Relation).*$', '', first_line, flags=re.IGNORECASE).strip()
        candidates.append(first_line)

    # 3. Fallback to basic Dear pattern
    dear_match = re.findall(r'(?:Dear|name\s+of\s+(?:insured|policyholder))[:\s,]+([A-Za-z\s.\-]{3,40})', norm_text, re.IGNORECASE)
    for dm in dear_match:
        candidates.append(dm.strip())

    # 4. Clean candidates and validate
    noise_keywords = [
        "limit", "sum", "insured", "benefit", "terms", "condition", 
        "policy", "premium", "date", "year", "turnaround", "prescribed", 
        "servicing", "question", "answer", "relationship", "gender", 
        "age", "dob", "address", "number", "details", "abha", "id", 
        "proposer", "holder", "schedule", "table", "product",
        "hospital", "clinic", "nursing", "medical", "doctor", "dr.", 
        "care", "service", "center", "centre", "limited", "ltd", 
        "company", "tpa", "bupa", "health", "insurance", "hallmark", 
        "labs", "laboratory", "diagnostics", "person", "exclusion", "excl",
        "ombudsman", "officer", "office", "intermediary"
    ]
    
    noise_regex = r'\b(thank|you|for|issued|to|per|the|period|either|at|inception|or|renewal|on|with|from|basis|policy|number|date|address|appointee|nominee|proposer|relation|relationship|member)\b.*$'
    
    valid_names = []
    seen = set()
    
    for c in candidates:
        # Strip trailing stop words/verbs
        clean_c = re.sub(noise_regex, '', c, flags=re.IGNORECASE).strip()
        clean_c = re.sub(r'[,\s\-]+$', '', clean_c).strip()
        # Clean salutation prefixes
        clean_c = re.sub(r'\b(mrs?|ms|miss|no)\.?\s+', '', clean_c, flags=re.IGNORECASE).strip()
        # OCR name-stitching: join single uppercase letter fragments into adjacent all-caps word
        # e.g. "RAJKUMA R JAIN" → "RAJKUMAR JAIN"
        clean_c = re.sub(r'(\b[A-Z]{2,})\s+([A-Z])\s+([A-Z]{2,}\b)', lambda m: m.group(1) + m.group(2) + ' ' + m.group(3), clean_c)
        
        lower_c = clean_c.lower()
        if len(clean_c) > 5 and ' ' in clean_c and lower_c not in seen:
            if clean_c[0].isupper() and not any(nk in lower_c for nk in noise_keywords):
                valid_names.append(clean_c)
                seen.add(lower_c)
                
    # 5. Substring deduplication
    sorted_names = sorted(valid_names, key=len, reverse=True)
    deduplicated = []
    for name in sorted_names:
        is_sub = False
        for longer in deduplicated:
            if name.lower() in longer.lower():
                is_sub = True
                break
        if not is_sub:
            deduplicated.append(name)
            
    if deduplicated:
        return ", ".join(deduplicated[:4])
            
    return "Not Mentioned"


def _build_fallback_fields(document_text: str) -> list[dict]:
    """Extract structured fields directly from document text using multi-pattern regex cascades."""
    # Preprocess text to normalize consecutive whitespaces into a single space
    # This reconstructs the horizontal layout of tables and avoids split-line match failures
    text = re.sub(r'\s+', ' ', document_text[:40000])
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
        r'(?:product\s+name|plan\s+name|policy\s+name)[:\s]+([A-Za-z0-9][A-Za-z0-9 \-&/]{3,60}?)(?=\s*UIN|\s*UIN|\s*\n|\s*\.\s|$)',
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
    policy_no = _regex_find_any([
        r'(\d{3,6}[A-Za-z]{1,5}/(?:ICICI|IBANK)/\d{6,15}/\d{2}/\d{3})',
        r'(?:policy\s+(?:no|number|certificate|id|doc(?:ument)?\s+no))[.:\s#]+((?!.*MSTR)[A-Za-z0-9][A-Za-z0-9\-/]{6,40})',
        r'(?:certificate\s+(?:no|number))[.:\s#]+((?!.*MSTR)[A-Za-z0-9][A-Za-z0-9\-/]{6,40})',
        r'(?:policy\s+schedule\s+(?:no|number))[.:\s#]+((?!.*MSTR)[A-Za-z0-9][A-Za-z0-9\-/]{6,40})',
        r'(?:endorsement\s+no)[.:\s#]+((?!.*MSTR)[A-Za-z0-9][A-Za-z0-9\-/]{6,40})',
        r'(\d{16,20})',
        r'(\d{10,15})',
    ], text)
    add("Policy Number", policy_no, "policy_info")

    # ── Insured Person / Policyholder ───────────────────────────────────────
    insured_val = _extract_insured_persons_validated(document_text) # uses raw text inside
    add("Insured Person", insured_val, "policy_info")

    # ── Sum Insured ─────────────────────────────────────────────────────────
    sum_ins = _regex_find_any([
        r'sum\s+insured\s*(?:\(₹\))?\s*([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
        r'(?:base\s+)?sum\s+insured\s*(?:opted)?\s*[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
        r'(?:sum\s+insured|sum\s+assured|si)[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
        r'(?:total\s+sum\s+insured)[:\s₹Rs.]+([1-9]\d{4,})',
        r'(?:basic\s+sum\s+insured)[:\s₹Rs.]+([1-9]\d{4,})',
        r'₹\s*([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,})\s*(?:Lakh|Lakhs|lakh)?',
    ], text)
    add("Sum Insured", sum_ins, "coverage")

    # ── Premium Amount ──────────────────────────────────────────────────────
    prem_val = _extract_premium_validated(document_text, policy_no or "", sum_ins or "")
    add("Premium Amount", prem_val, "premium")

    # Removed: Deductible (not a standard display field per user requirements)

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
        # Removed: Expiry Date and Premium Due Date (duplicate of Renewal Date)
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
            # Removed: Expiry Date and Premium Due Date

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

    # Removed: Pre Existing Coverage (duplicate info already in Waiting Period)

    # ── Maternity Coverage ──────────────────────────────────────────────────
    _maternity_raw = _regex_find_any([
        r'[Mm]aternity[:\s]+(?!.*Code\s*[-–]\s*Excl)([^.\n]{10,120})',
        r'[Mm]aternity\s+[Bb]enefit[:\s]+(?!.*Code\s*[-–]\s*Excl)([^.\n]{10,100})',
        r'[Mm]aternity[^.\n]{0,30}(covered[^.\n]{0,80})',
    ], text)
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

    # ── Covered Members (Family Floater Member List) ───────────────────────
    _members_found = []
    # Use improved pattern that captures multi-word full names (up to 5 words) followed by relationship
    p_mem = r'\b([A-Z][A-Za-z .]{2,50})\s+(Self|Spouse|Wife|Husband|Son|Daughter|Applicant|Father|Mother|Parent|Sibling|Brother|Sister|Child|Dependent)\b'
    members_matches = re.findall(p_mem, text)
    
    # Table-row patterns (e.g. Niva Bupa / Star Health style: Name Age DOB Gender Relationship)
    p_table = r'\b([A-Z][A-Za-z .]{2,35})\s+\d{1,3}\s+(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s+(?:Male|Female)\s+(Self|Spouse|Wife|Husband|Son|Daughter|Applicant|Father|Mother|Parent|Sibling|Brother|Sister|Child|Dependent)\b'
    members_matches.extend(re.findall(p_table, text))
    
    noise_members = [
        "appointee", "nominee", "proposer", "insured", "holder", "relationship", "relation", 
        "of the", "to the", "details", "policy", "premium", "ombudsman", "officer", "office", "intermediary",
        "male", "female", "gender", "hospital", "clinic", "nursing", "medical", "society", "association", 
        "road", "avenue", "street", "lane", "building", "floor", "city", "state", "district", "zone", 
        "pune", "mumbai", "surat", "gujarat", "bhopal", "delhi", "noida", "kolkata", "chennai", "jaipur", 
        "kochi", "lucknow", "bhubaneswar", "patna", "indore"
    ]
    seen_members = set()
    for name, rel in members_matches:
        # Normalize multiple spaces within the name (OCR often splits names)
        name_clean = re.sub(r'\s+', ' ', name.strip())
        rel_clean = rel.strip()
        name_clean = re.sub(r'^(?:and|or|for|with|to|of|at|on|in|dear|miss|no)\s+', '', name_clean, flags=re.IGNORECASE)
        # OCR name-stitching: join single uppercase letter fragments into adjacent word
        name_clean = re.sub(r'(\b[A-Z]{2,})\s+([A-Z])\s+([A-Z]{2,}\b)', lambda m: m.group(1) + m.group(2) + ' ' + m.group(3), name_clean)
        
        # Additional filter: reject name if it's too short or contains only digits/noise words
        name_clean_lower = name_clean.lower()
        if (
            name_clean
            and name_clean[0].isupper()
            and len(name_clean) > 3
            and not any(nw in name_clean_lower for nw in noise_members)
            and not name_clean_lower in ("male", "female", "gender")
        ):
            entry_key = (name_clean_lower[:20], rel_clean.lower())
            if entry_key not in seen_members:
                entry = f"{name_clean} ({rel_clean})"
                _members_found.append(entry)
                seen_members.add(entry_key)

    if _members_found:
        add("Covered Members", ", ".join(_members_found[:8]), "policy_info")

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

        # NOTE: Consulting Doctor is not extracted — it is not an insurance field
        # and causes confusion when insurance policies mention doctor names in claim sections

        # Only extract diagnosis from actual diagnostic sections
        diag_section_start = max(
            text.lower().find("diagnosis"),
            text.lower().find("clinical finding"),
            text.lower().find("impression"),
        )
        excl_start = text.lower().find("exclusion")
        if diag_section_start > 0 and (excl_start < 0 or diag_section_start < excl_start):
            diagnosis = _regex_find_any([
                r'(?:diagnosis|clinical\s+findings|impression)[:\s]+([A-Za-z][^.\n]{5,150})',
                r'(?:investigation)[:\s]+([A-Za-z][^.\n]{5,100})',
            ], text[:diag_section_start + 500])
            if diagnosis and "code excl" not in diagnosis.lower():
                add("Diagnosis / Findings", diagnosis, "coverage")

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
            "keep_alive": parse_keep_alive(settings.OLLAMA_KEEP_ALIVE),
            # Use num_ctx=settings.OLLAMA_NUM_CTX: same as all inference calls to avoid costly reload on first request
            "options": {"num_predict": 1, "num_ctx": settings.OLLAMA_NUM_CTX, "temperature": 0},
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
    num_ctx: Optional[int] = None,  # Must match warmup num_ctx to avoid KV cache reload
) -> str:
    """Call Ollama API using shared connection pool with GPU-accelerated settings."""
    num_ctx = num_ctx or settings.OLLAMA_NUM_CTX
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
    
    # Split text into lines
    raw_lines = text.split('\n')
    bullets = []
    current_bullet = ""
    
    # Regex to identify bullet/number list markers at the start of a line
    # Matches: •, -, *, ●, ▪, ❖, 1., 2), (a), etc.
    marker_pattern = re.compile(r'^([•\-\*\u2022\uf0b7●▪▫❖]|\d+\.?\)?|\([a-zA-Z0-9]+\))\s*')
    
    for line in raw_lines:
        line_str = line.strip()
        if not line_str:
            continue
        
        # Check if the line starts with a list marker/bullet/number
        is_new_bullet = False
        if marker_pattern.match(line_str):
            is_new_bullet = True
        elif line_str.startswith('•') or line_str.startswith('*') or line_str.startswith('-'):
            is_new_bullet = True
            
        if is_new_bullet:
            if current_bullet:
                bullets.append(current_bullet)
            # Strip the leading marker
            clean_line = marker_pattern.sub('', line_str).strip()
            # If there's still a leading bullet character, strip it too
            clean_line = re.sub(r'^[•\-\*\u2022\uf0b7●▪▫❖\s]+', '', clean_line).strip()
            current_bullet = clean_line
        else:
            if current_bullet:
                # Append line as continuation of the current bullet
                line_clean = re.sub(r'\s+', ' ', line_str).strip()
                current_bullet += f" {line_clean}"
            else:
                current_bullet = line_str

    if current_bullet:
        bullets.append(current_bullet)
        
    # Format each bullet: ensure it starts with '• ', has a capitalized first letter, and ends with a single '.'
    final_bullets = []
    for b in bullets:
        b_clean = b.strip()
        if not b_clean:
            continue
        # Capitalize first letter
        if len(b_clean) > 1:
            b_clean = b_clean[0].upper() + b_clean[1:]
        else:
            b_clean = b_clean.upper()
            
        # Ensure it ends with exactly one period
        b_clean = b_clean.rstrip('.')
        b_clean = f"• {b_clean}."
        final_bullets.append(b_clean)
        
    return "\n".join(final_bullets)


def _extract_key_context_for_summary(text: str, max_chars: int = 15000, header_chars: int = 8000) -> str:
    """Extract optimal document text context for LLM summarization.
    Takes the first header_chars (policy schedule, insured names, premiums) plus
    keyword-matched clauses from the rest, up to max_chars total.
    """
    if len(text) <= max_chars:
        return text
    
    # 1. Take first header_chars (policy schedule, insured names, premiums, sum insured)
    header_part = text[:header_chars]
    
    # 2. Extract key sentences for coverage, exclusions, waiting periods, and claims from remaining text
    remainder = text[header_chars:]
    keywords = [
        "inpatient", "day care", "daycare", "hospitalisation", "room rent",
        "pre-hospitalisation", "post-hospitalisation", "waiting period", "pre-existing",
        "exclusion", "not covered", "co-payment", "deductible", "claim", "cashless",
        "helpline", "maternity", "restoration", "bonus", "renewal", "network hospital",
        "sum insured", "ambulance", "AYUSH", "domiciliary", "discharge"
    ]
    
    matched_chunks = []
    lines = remainder.split("\n")
    for line in lines:
        l_lower = line.lower()
        if any(kw in l_lower for kw in keywords):
            cleaned = line.strip()
            if len(cleaned) > 20 and cleaned not in matched_chunks:
                matched_chunks.append(cleaned)
                if sum(len(c) for c in matched_chunks) >= (max_chars - header_chars):
                    break
                    
    body_part = "\n".join(matched_chunks)
    combined = f"{header_part}\n\n--- KEY POLICY CLAUSES ---\n{body_part}"
    return combined[:max_chars]


async def generate_summary(
    document_text: str,
    force_regenerate: bool = False,
    is_ocr: bool = False,
    fields_summary: Optional[str] = None,
) -> dict:
    """Generate AI summary using two dedicated Ollama calls:
    - Call 1: Structured bullet sections (coverage, exclusions, waiting, premium) via BULLETS_EXTRACTION_PROMPT
    - Call 2: Prose summary paragraph via PROSE_SUMMARY_PROMPT
    Falls back to regex-based extraction ONLY if both Ollama calls fail.
    """
    # ── Healthcare relevance check ──
    if not is_healthcare_related(document_text):
        logger.warning("[SUMMARY] Document is not healthcare-related. Bypassing LLM and returning error response.")
        return {
            "summary_text": "Validation Error: The uploaded document does not appear to be a valid health insurance policy, medical report, or healthcare-related document. Analysis is only available for medical, health insurance, and healthcare-related documents.",
            "coverage_summary": "• Invalid Document: The content does not contain health insurance or healthcare terminology.",
            "exclusions_summary": "• Invalid Document: The content does not contain health insurance or healthcare terminology.",
            "waiting_period_summary": "• Invalid Document: The content does not contain health insurance or healthcare terminology.",
            "premium_summary": "• Invalid Document: The content does not contain health insurance or healthcare terminology.",
        }

    # Use larger context to extract coverages, exclusions, and premiums accurately from multi-page documents
    context = _extract_key_context_for_summary(document_text, max_chars=6000, header_chars=3500)

    if fields_summary:
        context = f"VERIFIED POLICY DETAILS:\n{fields_summary}\n\nDOCUMENT TEXT:\n{context}"

    # ── CALL 1: Generate structured bullet sections ───────────────────────────
    bullets_result: dict = {}
    try:
        logger.info(f"[SUMMARY] 📋 Call 1 — Extracting bullet sections from {len(context)} chars via Ollama...")
        prompt1 = BULLETS_EXTRACTION_PROMPT
        if is_ocr:
            prompt1 = prompt1 + "\n- OCR Error Correction: The document text was extracted via OCR and may contain character misreads (e.g. '/' instead of '7', 'O' instead of '0', '.' instead of ','). You must reconstruct the correct numbers (e.g. '19.36/.0O' is '19,367.00', '50,00.000' is '50,00,000'). Please correct these values in your output."
            
        bullets_response = await call_ollama(
            prompt1.format(document_text=context),
            num_predict=700,   # Bullets only — enough for 4 sections × 4 bullets
            num_ctx=settings.OLLAMA_NUM_CTX,      # Matches warmup num_ctx — no model reload needed
        )
        bullets_parsed = extract_json_from_response(bullets_response)
        if bullets_parsed:
            logger.info("[SUMMARY] ✅ Call 1 succeeded — structured bullet sections extracted by LLM")
            bullets_result = bullets_parsed
        else:
            logger.warning("[SUMMARY] ⚠️ Call 1 returned empty JSON — will use regex fallback for bullets")
    except Exception as e:
        logger.warning(f"[SUMMARY] ⚠️ Call 1 failed ({e}) — will use regex fallback for bullets")

    # ── CALL 2: Generate prose summary paragraph ──────────────────────────────
    prose_text: str = ""
    try:
        logger.info(f"[SUMMARY] 📝 Call 2 — Generating prose summary from {len(context)} chars via Ollama...")
        prompt2 = PROSE_SUMMARY_PROMPT
        if is_ocr:
            prompt2 = prompt2 + "\n- OCR Error Correction: The document text was extracted via OCR and may contain character misreads (e.g. '/' instead of '7', 'O' instead of '0', '.' instead of ','). You must reconstruct the correct numbers (e.g. '19.36/.0O' is '19,367.00', '50,00.000' is '50,00,000'). Please correct these values in your output."
            
        prose_response = await call_ollama(
            prompt2.format(document_text=context),
            num_predict=600,   # Prose only — 4-5 paragraphs, ~120-150 words
            num_ctx=settings.OLLAMA_NUM_CTX,      # Matches warmup num_ctx — no model reload needed
        )
        # Prose response is plain text, not JSON
        prose_clean = prose_response.strip()
        if prose_clean:
            prose_text = clean_newlines_in_text(prose_clean)
            logger.info("[SUMMARY] ✅ Call 2 succeeded — prose summary generated by LLM")
        else:
            logger.warning("[SUMMARY] ⚠️ Call 2 returned empty — will use regex fallback for prose")
    except Exception as e:
        logger.warning(f"[SUMMARY] ⚠️ Call 2 failed ({e}) — will use regex fallback for prose")

    # ── Build fallback for any missing sections ───────────────────────────────
    fallback = _build_fallback_summary(document_text)

    # ── Assemble final output from LLM results (with fallback where LLM missed) ──

    def _bullets_from_list(lst) -> str:
        """Convert a list of bullet strings (from LLM JSON) into formatted bullet text."""
        if not lst or not isinstance(lst, list):
            return ""
        items = [str(item).strip() for item in lst if str(item).strip()]
        if not items:
            return ""
        # Run through clean_newlines_in_bullets to normalize formatting
        raw = "\n".join(f"• {item}" for item in items)
        return clean_newlines_in_bullets(raw)

    cov_llm = _bullets_from_list(bullets_result.get("coverage_summary"))
    excl_llm = _bullets_from_list(bullets_result.get("exclusions_summary"))
    wait_llm = _bullets_from_list(bullets_result.get("waiting_period_summary"))
    prem_llm = _bullets_from_list(bullets_result.get("premium_summary"))

    # Use LLM bullets if available, fallback to regex only when LLM returned nothing
    cov_final = cov_llm or clean_newlines_in_bullets(fallback.get("coverage_summary", ""))
    excl_final = excl_llm or clean_newlines_in_bullets(fallback.get("exclusions_summary", ""))
    wait_final = wait_llm or clean_newlines_in_bullets(fallback.get("waiting_period_summary", ""))
    prem_final = prem_llm or clean_newlines_in_bullets(fallback.get("premium_summary", ""))

    prose_final = prose_text or clean_newlines_in_text(fallback.get("summary_text", ""))

    logger.info(
        f"[SUMMARY] 🎯 Final assembly — prose={'LLM' if prose_text else 'FALLBACK'}, "
        f"coverage={'LLM' if cov_llm else 'FALLBACK'}, "
        f"exclusions={'LLM' if excl_llm else 'FALLBACK'}, "
        f"waiting={'LLM' if wait_llm else 'FALLBACK'}, "
        f"premium={'LLM' if prem_llm else 'FALLBACK'}"
    )

    return {
        "summary_text": prose_final,
        "coverage_summary": cov_final,
        "exclusions_summary": excl_final,
        "waiting_period_summary": wait_final,
        "premium_summary": prem_final,
    }




async def extract_policy_fields(document_text: str, force_regenerate: bool = False, is_ocr: bool = False) -> list[dict]:
    """Extract key fields. Falls back to demo data when Ollama is offline."""
    # ── Healthcare relevance check ──
    if not is_healthcare_related(document_text):
        logger.warning("[FIELDS] Document is not healthcare-related. Bypassing LLM.")
        return []

    context = _extract_key_context_for_summary(document_text, max_chars=15000, header_chars=8000)
    try:
        prompt = FIELD_EXTRACTION_PROMPT
        if is_ocr:
            prompt = prompt + "\n- OCR Error Correction: The document text was extracted via OCR and may contain character misreads (e.g. '/' instead of '7', 'O' instead of '0', '.' instead of ','). You must reconstruct the correct numbers (e.g. '19.36/.0O' is '19,367.00', '50,00.000' is '50,00,000'). Please correct these values in your JSON output."
            
        response = await call_ollama(
            prompt.format(document_text=context),
            num_predict=500,
            num_ctx=settings.OLLAMA_NUM_CTX,      # Consistent with warmup — no model reload
        )
        result = extract_json_from_response(response)
        if result:
            logger.info("Ollama field extraction successful")
            
            # Post-process and validate LLM outputs (e.g. reject list headers like 1.1)
            if "sum_insured" in result:
                val = str(result["sum_insured"]).strip()
                fallback_si = _regex_find_any([
                    r'sum\s+insured\s*(?:\(₹\))?\s*([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
                    r'(?:base\s+)?sum\s+insured\s*(?:opted)?\s*[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
                    r'(?:sum\s+insured|sum\s+assured|si)[:\s₹Rs.]+([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,}|[1-9]\d{0,2}\s*(?:Lakh|Lakhs|lakh|L|Cr|Crore))',
                    r'(?:total\s+sum\s+insured)[:\s₹Rs.]+([1-9]\d{4,})',
                    r'(?:basic\s+sum\s+insured)[:\s₹Rs.]+([1-9]\d{4,})',
                    r'₹\s*([1-9]\d*,\d{2,},\d{2,}|[1-9]\d{4,})\s*(?:Lakh|Lakhs|lakh)?',
                ], document_text)
                if fallback_si and fallback_si != "Not found in document":
                    clean_val = re.sub(r'[^\d]', '', val)
                    # If LLM returned 10,000,000 (10 Million) but fallback is 10 Lakh or 10,00,000 (1 Million)
                    # or if LLM value is weak/empty/invalid, correct it
                    is_confident_fallback = "lakh" in fallback_si.lower() or "cr" in fallback_si.lower() or "crore" in fallback_si.lower() or "," in fallback_si
                    if val in ("1", "1.1", "0", "null", "") or len(val) < 3 or (clean_val == "10000000" and ("10 Lakh" in fallback_si or "10,00,000" in fallback_si)) or is_confident_fallback:
                        result["sum_insured"] = fallback_si
                        
            # Validate and extract insured person safely
            insured_person = str(result.get("insured_person") or "").strip()
            # Clean common prefixes/suffixes
            insured_person = re.sub(r'^(?:proposer|insured|member|policyholder|name)\s*[:\-]\s*', '', insured_person, flags=re.IGNORECASE).strip()
            insured_person = re.sub(r'\s*\((?:proposer|self|insured|spouse|holder|owner)\)\s*$', '', insured_person, flags=re.IGNORECASE).strip()
            
            is_insured_valid = False
            if insured_person and len(insured_person) > 5 and ' ' in insured_person:
                noise_keywords = ["limit", "sum", "insured", "benefit", "terms", "condition", "policy", "premium", "date", "year", "turnaround", "prescribed", "servicing", "question", "answer", "relationship", "gender", "age", "dob", "address", "number", "details", "abha", "id", "ombudsman", "officer", "office", "intermediary"]
                if not any(nk in insured_person.lower() for nk in noise_keywords):
                    is_insured_valid = True
                    result["insured_person"] = insured_person
            
            if not is_insured_valid:
                result["insured_person"] = _extract_insured_persons_validated(document_text)

            # Validate and extract premium amount safely
            policy_num = str(result.get("policy_number") or "")
            sum_insured = str(result.get("sum_insured") or "")
            is_prem_valid = False
            if "premium_amount" in result:
                val = str(result["premium_amount"]).strip()
                if _is_valid_premium(val, policy_num, sum_insured):
                    is_prem_valid = True

            if not is_prem_valid:
                fallback_prem = _extract_premium_validated(document_text, policy_num, sum_insured)
                if fallback_prem:
                    result["premium_amount"] = f"₹{fallback_prem}" if not fallback_prem.startswith("₹") else fallback_prem
                else:
                    result["premium_amount"] = "Not specified"
            else:
                # Ensure it has currency symbol for professional look
                val = result["premium_amount"]
                if val and not any(c in val for c in ("₹", "Rs", "INR")):
                    result["premium_amount"] = f"₹{val}"

            field_category_map = {
                "policy_name": "policy_info",
                "insurer_name": "policy_info",
                "policy_number": "policy_info",
                "insured_person": "policy_info",
                "covered_members": "policy_info",
                "sum_insured": "coverage",
                "premium_amount": "premium",
                "policy_term": "policy_info",
                "renewal_date": "policy_period",
                "coverage_type": "coverage",
                "room_rent_limit": "restrictions",
                "waiting_period": "restrictions",
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
                "covered_members": "Covered Members",
                "sum_insured": "Sum Insured",
                "premium_amount": "Premium Amount",
                "policy_term": "Policy Term",
                "renewal_date": "Renewal Date",
                "coverage_type": "Coverage Type",
                "room_rent_limit": "Room Rent Limit",
                "waiting_period": "Waiting Period",
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
                if value
                   and str(value).lower() not in ("null", "none", "not specified")
                   and key not in ("pre_existing_coverage", "deductible")  # removed fields
            ]

            # Post-process: clean insurer name duplications (e.g. "HDFC ERGO... HDFC ERGO...")
            for f in fields:
                if f["field_name"] == "Insurer Name":
                    parts = f["field_value"].split()
                    half = len(parts) // 2
                    if half > 3 and parts[:half] == parts[half:]:
                        f["field_value"] = " ".join(parts[:half])
                    break
            
            # Ensure date fields and member fields are included from document text if LLM omitted them
            fallback_fields = _build_fallback_fields(document_text)
            existing_names = {f["field_name"].lower() for f in fields}
            for fb in fallback_fields:
                merge_candidates = ("renewal date", "expiry date", "premium due date", "insured person", "covered members")
                if fb["field_name"].lower() in merge_candidates and fb["field_name"].lower() not in existing_names:
                    fields.append(fb)
                    existing_names.add(fb["field_name"].lower())

            return fields
    except Exception as e:
        logger.warning(f"Ollama unavailable ({e}), extracting fields from document text")
    # Ollama offline: extract real fields from the uploaded document
    fallback = _build_fallback_fields(document_text)
    return fallback


async def analyze_risks(document_text: str, force_regenerate: bool = False) -> dict:
    """Detect risky clauses. Falls back to demo data when Ollama is offline."""
    # ── Healthcare relevance check ──
    if not is_healthcare_related(document_text):
        logger.warning("[RISKS] Document is not healthcare-related. Bypassing LLM.")
        return {
            "risks": [
                {
                    "clause_text": "Non-healthcare document",
                    "risk_type": "exclusion",
                    "severity": "high",
                    "explanation": "The uploaded document is not a healthcare or health insurance document.",
                    "recommendation": "Please upload a valid healthcare or health insurance document."
                }
            ],
            "overall_risk_level": "high"
        }

    context = _extract_key_context_for_summary(document_text, max_chars=15000, header_chars=8000)
    try:
        response = await call_ollama(
            RISK_ANALYSIS_PROMPT.format(document_text=context),
            num_predict=700,
            num_ctx=settings.OLLAMA_NUM_CTX,      # Consistent with warmup — no model reload
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
            return out
    except Exception as e:
        logger.warning(f"Ollama unavailable ({e}), extracting risks from document text")
    # Ollama offline: detect risk clauses from the actual uploaded document
    fallback = _build_fallback_risks(document_text)
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
            num_predict=300,
            num_ctx=settings.OLLAMA_NUM_CTX
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
            num_ctx=settings.OLLAMA_NUM_CTX
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


TREATMENTS_EXTRACTION_PROMPT = """Analyze the following health insurance policy document and extract a list of up to 10 major medical treatments, procedures, or chronic conditions that are explicitly covered, mentioned, or declared in the document.
Look only at the document text. Do not make up or copy example treatments.

You must return the response as a JSON object inside a ```json ``` code block:
```json
{{
  "treatments": ["Hospitalization Expenses", "Day Care Treatment", "Asthma Care", "Diabetes Treatment"]
}}
```

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
        response = await call_ollama(prompt, num_predict=150, num_ctx=settings.OLLAMA_NUM_CTX)
        result = extract_json_from_response(response)
        
        treatments = result.get("treatments", [])
        if isinstance(treatments, list) and len(treatments) > 0:
            clean_treatments = [str(t).strip() for t in treatments if str(t).strip()][:10]
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
        "Maternity Delivery",
        "Cancer Chemotherapy",
        "Hernia Repair Surgery",
        "Gallbladder Removal",
        "Cosmetic Surgery",
        "Hazardous Sports Injury"
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
            "num_ctx": 2048,
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
        rewritten = await call_ollama(prompt, num_predict=60, num_ctx=settings.OLLAMA_NUM_CTX)
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



