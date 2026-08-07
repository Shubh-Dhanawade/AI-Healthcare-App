"""
Prompt Builder Service
Constructs optimized, token-saving prompt templates for chat queries,
including the stored policy summary, retrieved chunks, and the last 6 turns of history.
"""

from typing import List, Dict, Any, Optional

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
    
    # 1. Format the retrieved chunks context
    # Raised to 2000 chars per chunk: insurance benefit schedules are dense tables
    # that can span 1500+ characters. Old 1200-char limit was cutting key rows.
    context_lines = []
    
    # Prepend structured data as virtual context chunk so the LLM reads it as verified ground-truth document facts
    if structured_context:
        first_policy_filename = policies[0].get("filename", "Policy Document")
        context_lines.append(f"[{first_policy_filename} - Page 1 - Policy Schedule Summary Details]\n{structured_context}\n")

    if is_comparison:
        by_source: Dict[str, List[str]] = {}
        for c in retrieved_chunks:
            src = c["source"]
            by_source.setdefault(src, []).append(c["text"])
            
        for src, texts in by_source.items():
            context_lines.append(f"=== {src} ===")
            for t in texts:
                context_lines.append(t[:800])  # Comparison keeps shorter per-source to stay token-safe
            context_lines.append("")
    else:
        for c in retrieved_chunks:
            context_lines.append(f"[{c['source']} - Page {c.get('page', 1)}]\n{c['text'][:2000]}")
            
    context_block = "\n---\n".join(context_lines) if context_lines else "No relevant policy chunks found in index. Use the structured data and summaries below to answer."
    
    # 2. Format the stored summaries of the policies being queried
    summary_lines = []
    for policy in policies:
        summary_info = policy.get("summary")
        if summary_info:
            parts = []
            if summary_info.get("summary_text"):
                parts.append(f"Summary: {summary_info['summary_text'][:500]}")
            if summary_info.get("premium_summary"):
                parts.append(f"Premium: {summary_info['premium_summary'][:200]}")
            if summary_info.get("coverage_summary"):
                parts.append(f"Coverage: {summary_info['coverage_summary'][:300]}")
            if summary_info.get("exclusions_summary"):
                parts.append(f"Exclusions: {summary_info['exclusions_summary'][:300]}")
            if summary_info.get("waiting_period_summary"):
                parts.append(f"Waiting Periods: {summary_info['waiting_period_summary'][:200]}")
            if parts:
                summary_lines.append(f"- {policy.get('filename')}:\n  " + "\n  ".join(parts))
    summaries_block = "\n".join(summary_lines) if summary_lines else "No policy summary available."
    
    # 3. Format the recent conversation history (last 6 messages / 3 exchanges to keep context small)
    history_lines = []
    if history:
        for msg in history[-6:]:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")[:200]  # Slightly longer history snippets
            history_lines += [f"{role}: {content}"]
    history_str = "\n".join(history_lines) if history_lines else "No previous conversation."
    
    # Format optional SQL database details
    structured_str = f"STRUCTURED DATABASE DETAILS (Exact facts from database — treat as ground truth):\n{structured_context}\n\n" if structured_context else ""

    # 4. Construct prompt
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
            "7. Do NOT output 'ASSISTANT:', 'USER:', or 'context:' labels in your response.\n"
            "\n"
            f"{structured_str}"
            f"STORED POLICY SUMMARIES:\n{summaries_block}\n\n"
            f"POLICY CONTEXT:\n{context_block}\n\n"
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
            "1. Answer the user's query using the STRUCTURED DATABASE DETAILS, STORED POLICY SUMMARIES, and POLICY CONTEXT blocks below. Treat structured details as verified ground-truth facts.\n"
            "2. READ EVERY CONTEXT BLOCK and structured detail carefully before answering.\n"
            "3. IMPORTANT: The policy may have a SCHEDULE OF BENEFITS table listing all covered items with section numbers (e.g. 1.1, 1.1.a, 1.1.1.i, 1.1.1.ii etc.). If the user asks about a treatment or benefit (e.g. 'dental', 'room rent', 'ambulance', 'AYUSH'), scan ALL context blocks for that exact section/row. Even a single matching row like '1.1.1.ii Dental Treatment - Covered upto sum insured' is sufficient to confirm coverage.\n"
            "4. If you find the term anywhere in the context — even in a table row — report the EXACT coverage value (e.g. 'Covered upto sum insured', 'At Actuals', 'Not Covered', specific Rupee limit).\n"
            "5. If the user's topic appears in an EXCLUSIONS section (e.g., listed under 'Excl18'), state clearly: 'This is listed under Exclusions — it is NOT covered under the policy.'\n"
            "6. NEVER say the topic is 'not mentioned' or 'absent' if it appears in the STRUCTURED DATABASE DETAILS, summaries, or any context block.\n"
            "7. If the information is genuinely absent from ALL sources, say: 'I could not find that specific detail in the provided document excerpts. You may refer to the full policy document for this information.'\n"
            "8. Do NOT include any inline references, page numbers, or source citations inside your response text.\n"
            "9. Do NOT output 'ASSISTANT:', 'USER:', or 'context:' labels in your response.\n"
            "10. Never output curly braces in your answer.\n"
            "\n"
            f"{structured_str}"
            f"STORED POLICY SUMMARIES:\n{summaries_block}\n\n"
            f"POLICY CONTEXT (direct document excerpts — read all blocks carefully):\n{context_block}\n\n"
            f"PREVIOUS CONVERSATION:\n{history_str}\n\n"
            f"User Query: {query}\n"
            "\n"
            "Response:"
        )
        
    return prompt
