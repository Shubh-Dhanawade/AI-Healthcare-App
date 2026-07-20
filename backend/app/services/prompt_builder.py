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
    is_comparison: bool = False
) -> str:
    """Build an optimized prompt for RAG Chat."""
    
    # 1. Format the retrieved chunks context
    # Truncation raised to 1200 chars (was 600): insurance clauses can be 700-900 chars;
    # the old limit was cutting maternity/coverage details before the key sentence.
    context_lines = []
    if is_comparison:
        by_source: Dict[str, List[str]] = {}
        for c in retrieved_chunks:
            src = c["source"]
            by_source.setdefault(src, []).append(c["text"])
            
        for src, texts in by_source.items():
            context_lines.append(f"=== {src} ===")
            for t in texts:
                context_lines.append(t[:600])  # Comparison keeps shorter per-source to stay token-safe
            context_lines.append("")
    else:
        for c in retrieved_chunks:
            context_lines.append(f"[{c['source']} - Page {c.get('page', 1)}]\n{c['text'][:1200]}")
            
    context_block = "\n---\n".join(context_lines) if context_lines else "No relevant policy details found in indexed chunks — use the stored summaries below to answer."
    
    # 2. Format the stored summaries of the policies being queried
    summary_lines = []
    for policy in policies:
        summary_info = policy.get("summary")
        if summary_info:
            parts = []
            if summary_info.get("summary_text"):
                parts.append(f"Summary: {summary_info['summary_text'][:300]}")
            if summary_info.get("premium_summary"):
                parts.append(f"Premium: {summary_info['premium_summary'][:150]}")
            if summary_info.get("coverage_summary"):
                parts.append(f"Coverage: {summary_info['coverage_summary'][:150]}")
            if summary_info.get("exclusions_summary"):
                parts.append(f"Exclusions: {summary_info['exclusions_summary'][:150]}")
            if summary_info.get("waiting_period_summary"):
                parts.append(f"Waiting Periods: {summary_info['waiting_period_summary'][:150]}")
            if parts:
                summary_lines.append(f"- {policy.get('filename')}:\n  " + "\n  ".join(parts))
    summaries_block = "\n".join(summary_lines) if summary_lines else "No policy summary available."
    
    # 3. Format the recent conversation history (last 6 messages / 3 exchanges to keep context small)
    history_lines = []
    if history:
        for msg in history[-6:]:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")[:150]  # Truncate long history messages
            history_lines += [f"{role}: {content}"]
    history_str = "\n".join(history_lines) if history_lines else "No previous conversation."
    
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
            "6. Do NOT include any inline references, page numbers, or source citations (such as 'Reference: ...', 'Page X', 'In filename.pdf - Page Y') inside your response text. Citations are automatically appended in a separate section below your response, so any inline citations are redundant.\n"
            "7. Do NOT include any 'ASSISTANT:', 'USER:', or 'context:' labels in your response.\n"
            "\n"
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
            "1. Answer the user's query using the POLICY CONTEXT blocks below. Each block is a direct excerpt from the policy document.\n"
            "2. READ EVERY CONTEXT BLOCK carefully before answering. The answer may appear in any block, not just the first one.\n"
            "3. If the user asks about a specific topic (e.g. maternity, deductible, room rent, co-payment, waiting period) — search ALL context blocks for that exact word or any related synonym. Even if it appears only once in one block, report it.\n"
            "4. Inspect the Exclusions sections in the context. If a topic (e.g., maternity, pregnancy, childbirth) is listed under exclusions (e.g., Excl18), state clearly that it is explicitly mentioned in the exclusions section, and explain that this means it is NOT covered under the policy. Do not say it is 'not mentioned' or 'not covered' generically if it is listed as an exclusion.\n"
            "5. NEVER say the topic is 'not mentioned' unless you have read through every context block above and found zero relevant text AND the stored summaries also have no mention.\n"
            "6. If the information is genuinely absent from BOTH context AND summaries, say: 'I could not find that specific detail in the provided document excerpts. You may refer to the full policy document for this information.'\n"
            "7. Do NOT include any inline references, page numbers, or source citations inside your response text.\n"
            "8. Do NOT output 'ASSISTANT:', 'USER:', or 'context:' labels in your response.\n"
            "9. Never output curly braces in your answer.\n"
            "\n"
            f"STORED POLICY SUMMARIES:\n{summaries_block}\n\n"
            f"POLICY CONTEXT (direct document excerpts — read all blocks):\n{context_block}\n\n"
            f"PREVIOUS CONVERSATION:\n{history_str}\n\n"
            f"User Query: {query}\n"
            "\n"
            "Response:"
        )
        
    return prompt

