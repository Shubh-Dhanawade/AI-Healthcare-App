import re
from typing import List, Dict, Any, Optional

def clean_exclusion_codes(text: str) -> str:
    """
    Remove IRDAI exclusion codes (e.g., Code - Excl08, Excl08, Code-Excl12)
    from context so the LLM does not see or repeat them in responses.
    """
    if not text:
        return ""
    text = re.sub(r"(?i)\(?Code\s*[-\u2013]\s*Excl\d+\)?", "", text)
    text = re.sub(r"(?i)\bExcl\d+\b", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_chat_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]],
    policies: List[Dict[str, Any]],
    user_name: str = "there",
    is_comparison: bool = False,
    structured_context: str = ""
) -> str:
    """Build an optimized prompt for RAG Chat."""

    # 1. Format retrieved chunks context
    context_lines = []
    if structured_context:
        first_policy_filename = policies[0].get("filename", "Policy Document") if policies else "Policy Document"
        cleaned_structured = clean_exclusion_codes(structured_context)
        context_lines.append(
            f"[{first_policy_filename} - Page 1 - Policy Schedule Summary Details]\n{cleaned_structured}\n"
        )

    if is_comparison:
        by_source: Dict[str, List[str]] = {}
        for c in retrieved_chunks:
            src = c["source"]
            by_source.setdefault(src, []).append(c["text"])
        for src, texts in by_source.items():
            context_lines.append(f"=== {src} ===")
            for t in texts:
                context_lines.append(clean_exclusion_codes(t[:800]))
            context_lines.append("")
    else:
        for c in retrieved_chunks:
            cleaned_chunk_text = clean_exclusion_codes(c["text"][:2000])
            context_lines.append(f"[{c['source']} - Page {c.get('page', 1)}]\n{cleaned_chunk_text}")

    context_block = (
        "\n---\n".join(context_lines)
        if context_lines
        else "No relevant policy chunks found in index. Use the structured data and summaries below to answer."
    )

    # 2. Format stored policy summaries
    summary_lines = []
    for policy in policies:
        summary_info = policy.get("summary")
        if summary_info:
            parts = []
            if summary_info.get("summary_text"):
                parts.append(f"Summary: {clean_exclusion_codes(summary_info['summary_text'][:500])}")
            if summary_info.get("premium_summary"):
                parts.append(f"Premium: {clean_exclusion_codes(summary_info['premium_summary'][:200])}")
            if summary_info.get("coverage_summary"):
                parts.append(f"Coverage: {clean_exclusion_codes(summary_info['coverage_summary'][:300])}")
            if summary_info.get("exclusions_summary"):
                parts.append(f"Exclusions: {clean_exclusion_codes(summary_info['exclusions_summary'][:300])}")
            if summary_info.get("waiting_period_summary"):
                parts.append(f"Waiting Periods: {clean_exclusion_codes(summary_info['waiting_period_summary'][:200])}")
            if parts:
                summary_lines.append(f"- {policy.get('filename')}:\n  " + "\n  ".join(parts))
    summaries_block = "\n".join(summary_lines) if summary_lines else "No policy summary available."

    # 3. Format recent conversation history
    history_lines = []
    if history:
        for msg in history[-6:]:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")[:200]
            history_lines.append(f"{role}: {content}")
    history_str = "\n".join(history_lines) if history_lines else "No previous conversation."

    # 4. Structured DB details string
    structured_str = (
        f"STRUCTURED DATABASE DETAILS (Exact facts from database - treat as ground truth):\n"
        f"{clean_exclusion_codes(structured_context)}\n\n"
    ) if structured_context else ""

    # 5. Extract Schedule of Benefits block specifically if present
    schedule_blocks = []
    for policy in policies:
        raw_text = (policy.get("text") or "").strip()
        if raw_text:
            pos = raw_text.find("SCHEDULE OF BENEFITS")
            if pos == -1:
                pos = raw_text.find("CUSTOMER INFORMATION SHEET")
            if pos != -1:
                sched_text = raw_text[pos:pos + 6000]
                filename = policy.get("filename", "Policy")
                schedule_blocks.append(f"=== {filename} SCHEDULE OF BENEFITS ===\n{clean_exclusion_codes(sched_text)}")
    schedule_str = (
        "SCHEDULE OF BENEFITS & COVERAGE TABLE (Ground-truth benefit table extracted from policy schedule - read every row carefully):\n"
        + "\n\n".join(schedule_blocks)
        + "\n\n"
    ) if schedule_blocks else ""

    # 6. FULL DOCUMENT TEXT fallback block (first 10,000 chars of raw OCR text)
    # This guarantees that table-based fields (nominee name, insured persons, DOB,
    # policy number, schedule of benefits) are always visible to the LLM.
    full_text_blocks = []
    for policy in policies:
        raw_text = (policy.get("text") or "").strip()
        if raw_text:
            cleaned_raw = clean_exclusion_codes(raw_text[:10000])
            filename = policy.get("filename", "Policy")
            full_text_blocks.append(f"=== {filename} ===\n{cleaned_raw}")
    full_text_str = (
        "FULL DOCUMENT TEXT (Complete raw text - use this as the primary source for any "
        "table-row fields such as nominee name, insured person name, date of birth, policy number, "
        "premium breakdown, or relationship. Read every line carefully):\n"
        + "\n\n".join(full_text_blocks)
        + "\n\n"
    ) if full_text_blocks else ""

    # 7. Construct final prompt
    if is_comparison:
        policy_names = [p.get("filename", "Policy") for p in policies]
        names_str = ", ".join(policy_names)
        prompt = (
            f"You are HealthPolicyLens, an expert healthcare insurance advisor helping {user_name}.\n"
            f"You are comparing the following insurance policies: {names_str}.\n"
            "\n"
            "Instructions:\n"
            "1. Compare the policies directly based on the provided POLICY CONTEXT, STORED SUMMARIES, and PREVIOUS CONVERSATION.\n"
            "2. Clearly specify which details belong to which policy by name.\n"
            "3. Use a markdown comparison table or bullet lists grouped by policy name for readability.\n"
            "4. Highlight differences in key terms (deductibles, co-pays, waiting periods, room rent caps).\n"
            "5. Maintain a professional tone and end with a concise recommendation.\n"
            "6. Do NOT include any inline references, page numbers, or source citations inside your response text.\n"
            "7. Do NOT output ASSISTANT:, USER:, or context: labels in your response.\n"
            "8. IMPORTANT: Never output exclusion codes under any circumstances. Always refer to exclusions using their descriptive names.\n"
            "\n"
            f"{full_text_str}"
            f"POLICY CONTEXT:\n{context_block}\n\n"
            f"STORED POLICY SUMMARIES:\n{summaries_block}\n\n"
            f"{schedule_str}"
            f"{structured_str}"
            f"PREVIOUS CONVERSATION:\n{history_str}\n\n"
            f"User Query: {query}\n"
            "\n"
            "Comparison Response:"
        )
    else:
        prompt = (
            f"You are HealthPolicyLens, a knowledgeable and friendly healthcare insurance assistant helping {user_name}.\n"
            "\n"
            "Instructions:\n"
            "1. Answer the user query using the STRUCTURED DATABASE DETAILS, SCHEDULE OF BENEFITS, STORED POLICY SUMMARIES, POLICY CONTEXT, and FULL DOCUMENT TEXT blocks below. Treat structured details and schedule of benefits as verified ground-truth facts.\n"
            "2. READ EVERY CONTEXT BLOCK and structured detail carefully before answering.\n"
            "3. IMPORTANT for Hospitalization: Hospitalization expenses (including Room Rent at actuals, ICU at actuals, Day Care, etc.) ARE COVERED under the policy up to the Sum Insured. Do NOT confuse previous claim history entries (e.g. 'Hospitalization claim made in last policy year NA') with coverage! 'NA' under claim history means zero claims were submitted in past years, NOT that hospitalization is uncovered!\n"
            "4. IMPORTANT for Dental Treatment: Check the SCHEDULE OF BENEFITS & COVERAGE TABLE or POLICY CONTEXT carefully. If it specifies 'Dental Treatment (Accidental Hospitalization Only) Covered upto sum insured' or similar, state clearly that dental treatment is COVERED specifically for accidental hospitalization up to the full sum insured.\n"
            "5. IMPORTANT for Maternity Coverage: Check the SCHEDULE OF BENEFITS, STRUCTURED DATABASE DETAILS, and STORED POLICY SUMMARIES carefully. If they specify 'Maternity Coverage' is covered (or list limits/waiting periods for it, or if 'Parenthood' add-on is active), state clearly that maternity is covered and mention any specific limits/waiting periods. If it is marked as excluded, not covered, or if Parenthood is not opted, state clearly that maternity coverage is excluded.\n"
            "6. IMPORTANT for AYUSH Treatment: Check the SCHEDULE OF BENEFITS table. If it specifies 'Covered upto sum insured', state clearly that AYUSH treatment (Ayurveda, Yoga, Unani, Siddha, Homeopathy) is COVERED up to the full Base Sum Insured (e.g., ₹20,00,000).\n"
            "7. IMPORTANT: For questions about nominee name, insured person name, date of birth, relationship, policy number, or any table-based field - ALWAYS scan the SCHEDULE OF BENEFITS and FULL DOCUMENT TEXT blocks first.\n"
            "7. If you find the term anywhere in the context - even in a table row - report the EXACT value (e.g. Covered upto sum insured, At Actuals, Not Covered, specific Rupee limit, specific person name).\n"
            "8. If the topic appears in an EXCLUSIONS section, state clearly: This is listed under Exclusions - it is NOT covered under the policy.\n"
            "9. NEVER say a benefit is not mentioned or absent if it appears in the SCHEDULE OF BENEFITS, summaries, POLICY CONTEXT, or FULL DOCUMENT TEXT.\n"
            "10. If the information is genuinely absent from ALL sources, say: I could not find that specific detail in the provided document. You may refer to the full policy document for this information.\n"
            "11. Do NOT include any inline references, page numbers, or source citations inside your response text.\n"
            "12. Do NOT output ASSISTANT:, USER:, or context: labels in your response.\n"
            "13. Never output curly braces in your answer.\n"
            "14. IMPORTANT: Never output exclusion codes under any circumstances. Always refer to exclusions using their descriptive names.\n"
            "\n"
            f"{full_text_str}"
            f"POLICY CONTEXT (direct document excerpts - read all blocks carefully):\n{context_block}\n\n"
            f"STORED POLICY SUMMARIES:\n{summaries_block}\n\n"
            f"{schedule_str}"
            f"{structured_str}"
            f"PREVIOUS CONVERSATION:\n{history_str}\n\n"
            f"User Query: {query}\n"
            "\n"
            "Response:"
        )

    return prompt