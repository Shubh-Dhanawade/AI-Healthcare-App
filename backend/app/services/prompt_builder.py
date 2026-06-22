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
    context_lines = []
    if is_comparison:
        by_source: Dict[str, List[str]] = {}
        for c in retrieved_chunks:
            src = c["source"]
            by_source.setdefault(src, []).append(c["text"])
            
        for src, texts in by_source.items():
            context_lines.append(f"=== {src} ===")
            for t in texts:
                # Truncate context per block to save tokens
                context_lines.append(t[:400])
            context_lines.append("")
    else:
        for c in retrieved_chunks:
            context_lines.append(f"[{c['source']} - Page {c.get('page', 1)}]\n{c['text'][:450]}")
            
    context_block = "\n---\n".join(context_lines) if context_lines else "No relevant policy details found."
    
    # 2. Format the stored summaries of the policies being queried
    summary_lines = []
    for policy in policies:
        summary_info = policy.get("summary")
        if summary_info and summary_info.get("summary_text"):
            summary_lines.append(f"- {policy.get('filename')}: {summary_info['summary_text'][:200]}")
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
            f"You are HealthAI, an expert healthcare insurance advisor helping {user_name}.\n"
            f"You are comparing the following insurance policies: {names_str}.\n"
            "\n"
            "Instructions:\n"
            "1. Compare the policies directly based on the provided POLICY CONTEXT, STORED SUMMARIES, and PREVIOUS CONVERSATION.\n"
            "2. Clearly specify which details belong to which policy by name.\n"
            "3. Use a markdown comparison table or bullet lists grouped by policy name for readability.\n"
            "4. Highlight differences in key terms (deductibles, co-pays, waiting periods, room rent caps).\n"
            "5. Maintain a professional tone and end with a concise recommendation.\n"
            "6. Do NOT include any 'ASSISTANT:', 'USER:', or 'context:' labels in your response.\n"
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
            f"You are HealthAI, a knowledgeable and friendly healthcare insurance assistant helping {user_name}.\n"
            "\n"
            "Instructions:\n"
            "1. Answer the user's query clearly and concisely using the provided POLICY CONTEXT, STORED SUMMARIES, and PREVIOUS CONVERSATION.\n"
            "2. If the user is asking for clarification, explanation of terms, or a follow-up question on previous responses, use the PREVIOUS CONVERSATION and general insurance knowledge to answer directly and politely.\n"
            "3. When referencing specific policy facts, always mention the source document name (e.g., 'In Star_Health.pdf...').\n"
            "4. If the query asks for specific policy details (such as coverage limits, deductibles, premium amounts, etc.) that are not in the context, and cannot be inferred from history, state: 'I could not find this specific information in the selected policies.' For general inquiries, greetings, or questions about who you are, answer them directly based on your instructions.\n"
            "5. Do NOT output 'ASSISTANT:', 'USER:', or 'context:' labels in your response.\n"
            "6. Never output curly braces in your answer.\n"
            "\n"
            f"STORED POLICY SUMMARIES:\n{summaries_block}\n\n"
            f"POLICY CONTEXT:\n{context_block}\n\n"
            f"PREVIOUS CONVERSATION:\n{history_str}\n\n"
            f"User Query: {query}\n"
            "\n"
            "Response:"
        )
        
    return prompt
