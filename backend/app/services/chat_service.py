"""
Chat Service
Orchestrates the entire Conversational RAG pipeline. Handles chitchat routing,
history-based query rewriting, FAISS vector store retrieval, and token streaming.
"""

import asyncio
import re
import time
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.cache_manager import CacheManager
from app.services.ollama_client import call_ollama, call_ollama_stream
from app.services.vector_store import search_vector_store
from app.services.prompt_builder import build_chat_prompt
from app.services.ai_service import (
    _get_chitchat_response, _needs_query_rewriting, rewrite_query_with_history,
    _is_comparison_query
)


# ─────────────────────────────────────────
# Similarity threshold — lowered to -1.0 (no threshold)
# so keyword-boosted chunks are not filtered out before the 
# re-ranking pass in vector_store has a chance to elevate them.
# The LLM is smart enough to reject irrelevant context.
# ─────────────────────────────────────────
SIMILARITY_THRESHOLD = -1.0

_STOP_WORDS = {"the", "and", "for", "with", "that", "this", "what", "how", "are", "you", "can", "does", "did", "was", "has", "have"}

def _text_search_fallback(query: str, policies: List[Dict[str, Any]], top_k: int = 6) -> List[Dict[str, Any]]:
    """
    Fast keyword search over policy document text with table-row awareness and synonym expansion.
    Used concurrently with vector search in hybrid retrieval.
    """
    query_lower = query.lower()
    # Expand query with common insurance synonyms for maximum recall
    SYNONYMS = {
        "dental": ["teeth", "tooth", "oral", "dentist", "dental treatment"],
        "maternity": ["pregnancy", "childbirth", "delivery", "newborn", "obstetric", "miscarriage"],
        "covered": ["covers", "coverage", "eligible", "payable", "admissible", "benefit"],
        "room rent": ["room charges", "accommodation", "hospital room", "icu", "shared room"],
        "pre-existing": ["pre existing", "ped", "pre-existing disease"],
        "waiting period": ["waiting", "moratorium", "initial period"],
        "premium": ["amount paid", "annual cost", "total premium"],
        "claim": ["reimbursement", "cashless", "hospital discharge"],
        "deductible": ["aggregate deductible", "excess", "deduct"],
        "ayush": ["ayurveda", "homeopathy", "unani", "siddha"],
        "ambulance": ["air ambulance", "transport", "emergency travel"],
        "cataract": ["eye", "lens", "vision"],
        "member": ["insured person", "covered person", "family member", "beneficiary",
                    "dependent", "self", "spouse", "son", "daughter", "insured persons",
                    "members covered", "persons covered", "member details", "insured members"],
        "who is covered": ["insured person", "covered person", "insured members", "member list",
                            "family members", "policy members", "persons insured"],
    }
    original_tokens = set(re.findall(r'\w+', query_lower))
    original_words = [w for w in original_tokens if len(w) > 2 and w not in _STOP_WORDS]
    
    synonym_words = set()
    for base_term, syns in SYNONYMS.items():
        if any(s in query_lower for s in [base_term] + syns):
            for s in syns:
                for w in s.split():
                    if w not in _STOP_WORDS and w not in original_words:
                        synonym_words.add(w)

    if not original_words:
        return []

    hits: List[Dict[str, Any]] = []

    for policy in policies:
        text = policy.get("text", "") or ""
        filename = policy.get("filename", "Policy")
        if not text:
            continue

        # Use 1000-char chunks with 200-char overlap
        chunk_size = 1000
        overlap = 200
        start = 0
        chunks = []
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            # Don't cut in the middle of a line — extend to next newline
            if end < len(text) and '\n' in text[end:end+100]:
                nl_pos = text.index('\n', end)
                chunk = text[start:nl_pos]
            chunks.append((start, chunk))
            start += chunk_size - overlap

        for pos, chunk in chunks:
            chunk_lower = chunk.lower()
            # Original words count for 5 points each
            orig_score = sum(chunk_lower.count(w) * 5 for w in original_words)
            # Synonyms count for 1 point each
            syn_score = sum(chunk_lower.count(w) for w in synonym_words)
            score = orig_score + syn_score
            
            if score > 0:
                # Boost chunks from benefit schedule / coverage sections (tables)
                schedule_boost = 0
                if any(k in chunk_lower for k in ["schedule of benefits", "section", "covered upto", "covered up to", "at actuals", "1.1", "1.2"]):
                    schedule_boost = 15
                # Additional boost if exact query terms appear in chunk
                if any(qw in chunk_lower for qw in original_words if len(qw) > 3):
                    schedule_boost += 10

                page_num = 1
                page_match = re.search(r'\[Page (\d+)\]', chunk)
                if page_match:
                    page_num = int(page_match.group(1))
                hits.append({
                    "text": chunk,
                    "source": filename,
                    "page": page_num,
                    "score": float(score + schedule_boost),
                    "document_id": policy.get("id", "")
                })

    hits.sort(key=lambda x: x["score"], reverse=True)
    # De-duplicate overlapping chunks (same text within first 80 chars)
    seen_starts: set = set()
    deduped = []
    for h in hits:
        key = h["text"][:80].lower().strip()
        if key not in seen_starts:
            seen_starts.add(key)
            deduped.append(h)
    return deduped[:top_k]


def classify_question(query: str) -> str:
    """
    Classify user query into FACTUAL, SEMANTIC, or COMPLEX.
    """
    query_lower = query.lower()
    factual_keywords = [
        "policy number", "premium", "due date", "expiry", "renewal date",
        "deductible", "co-pay", "copay", "co-payment", "holder",
        "network", "company", "issuer", "what is my number", "policy end date",
        "member", "members", "who is covered", "family member", "insured person",
        "covered person", "who are covered", "list of members", "policy members",
        "spouse", "son", "daughter", "dependent", "beneficiary",
    ]
    complex_keywords = [
        "summarize", "summary", "explain my policy", "will diabetes", "covered", 
        "limitations", "comparison", "compare", "likely be covered", "pre-existing"
    ]
    if any(k in query_lower for k in complex_keywords):
        return "COMPLEX"
    if any(k in query_lower for k in factual_keywords):
        return "FACTUAL"
    return "SEMANTIC"


async def fetch_structured_policy_data(db: AsyncSession, policy_ids: List[str]) -> str:
    """Fetch structured metadata, extracted fields, summaries, and risk analyses for target policies in a single query."""
    if not policy_ids:
        return ""
    from app.models.document import Document, ExtractedField
    from app.models.risk_analysis import Summary
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    
    stmt = select(Document).where(Document.id.in_(policy_ids)).options(
        selectinload(Document.extracted_fields),
        selectinload(Document.summary)
    )
    res = await db.execute(stmt)
    docs = res.scalars().all()
    
    structured_blocks = []
    for doc in docs:
        block = [
            f"=== STRUCTURED DETAILS FOR {doc.original_filename} ===",
            f"Policy Name: {doc.original_filename}",
            f"Renewal Date: {doc.renewal_date.strftime('%Y-%m-%d') if doc.renewal_date else 'Not Mentioned'}",
            f"Premium Due Date: {doc.premium_due_date.strftime('%Y-%m-%d') if doc.premium_due_date else 'Not Mentioned'}"
        ]
        
        for field in doc.extracted_fields:
            if field.field_name and field.field_value:
                block.append(f"{field.field_name}: {field.field_value}")

        sum_obj = doc.summary
        if sum_obj:
            # Always inject full summary_text first — it contains member names, policy details, exclusions
            if sum_obj.summary_text:
                block.append(f"\n--- Full Policy Summary ---\n{sum_obj.summary_text}\n---")
            if sum_obj.coverage_summary:
                block.append(f"Coverage Summary: {sum_obj.coverage_summary}")
            if sum_obj.exclusions_summary:
                block.append(f"Exclusions Summary: {sum_obj.exclusions_summary}")
            if sum_obj.waiting_period_summary:
                block.append(f"Waiting Period Summary: {sum_obj.waiting_period_summary}")
            if sum_obj.premium_summary:
                block.append(f"Premium Summary: {sum_obj.premium_summary}")
            
        structured_blocks.append("\n".join(block))
        
    return "\n\n---\n\n".join(structured_blocks) if structured_blocks else ""


async def run_chat_query(
    policies: List[Dict[str, Any]],
    query: str,
    db: AsyncSession,
    history: List[Dict[str, str]],
    user_name: str = "there",
    user_id: Optional[str] = None,
) -> str:
    """Non-streaming Conversational RAG execution."""
    start_time = time.time()
    
    # 1. Intent check: handle greetings, guidance, and thanks immediately
    chitchat_reply = _get_chitchat_response(query, user_name)
    if chitchat_reply:
        logger.info(f"Chitchat/Guidance query short-circuited: '{query}'")
        return chitchat_reply
        
    # 2. Fast query setup (Conversation history is built into prompt directly — no extra blocking LLM call)
    search_query = query
        
    # 3. Cache lookup
    policy_ids = [p.get("id") for p in policies if p.get("id")]
    cache_key = CacheManager.get_rag_cache_key(search_query, policy_ids)
    cached_response = CacheManager.get(cache_key)
    if cached_response:
        logger.info(f"Cache hit for query: '{search_query}'")
        return cached_response
        
    # 4. Parallel Hybrid RAG Retrieval (Vector Search + Keyword Match + Structured DB Data)
    is_comparison = _is_comparison_query(search_query) and len(policies) > 1
    top_k = 10 if is_comparison else 8
    
    retrieval_start = time.time()
    keyword_chunks = _text_search_fallback(search_query, policies, top_k=top_k)
    vector_chunks = await search_vector_store(db, search_query, policies, top_k=top_k)
    structured_context = await fetch_structured_policy_data(db, policy_ids)
    
    # Merge & Deduplicate vector + keyword results
    seen_texts: set = set()
    combined_chunks: List[Dict[str, Any]] = []
    
    for kc in keyword_chunks:
        kc_key = kc["text"][:80].lower().strip()
        if kc_key not in seen_texts:
            seen_texts.add(kc_key)
            combined_chunks.append(kc)

    for vc in vector_chunks:
        vc_key = vc["text"][:80].lower().strip()
        if vc_key not in seen_texts:
            seen_texts.add(vc_key)
            combined_chunks.append(vc)

    combined_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    filtered_chunks = combined_chunks[:top_k]
    retrieval_time = time.time() - retrieval_start
    
    logger.info(
        f"Hybrid retrieval retrieved {len(combined_chunks)} combined chunks in {retrieval_time:.4f}s. "
        f"Selected top {len(filtered_chunks)} chunks for LLM context."
    )
        
    # 5. Build prompt
    prompt = build_chat_prompt(
        query=query,
        retrieved_chunks=filtered_chunks,
        history=history,
        policies=policies,
        user_name=user_name,
        is_comparison=is_comparison,
        structured_context=structured_context
    )
    
    # 6. Call LLM — using num_ctx=8192 (to prevent context truncation for complex health policies)
    llm_start = time.time()
    response = await call_ollama(prompt, num_predict=600 if is_comparison else 450, num_ctx=8192)
    llm_time = time.time() - llm_start
    
    total_time = time.time() - start_time
    logger.info(f"⏱️ Non-streaming response generated in {total_time:.4f}s [Retrieval: {retrieval_time:.4f}s, LLM: {llm_time:.4f}s]")
    # Append structured sources footer so the backend can parse and save it
    if filtered_chunks:
        source_pages = {}
        for chunk in filtered_chunks:
            src = chunk.get("source", "")
            if src:
                pg = chunk.get("page", 1)
                if src not in source_pages:
                    source_pages[src] = pg
                else:
                    source_pages[src] = min(source_pages[src], pg)
        ordered_sources = []
        for src, min_pg in source_pages.items():
            ordered_sources.append(f"{src} - Page {min_pg}")
        if ordered_sources:
            response += "\n[SOURCES:" + "|".join(ordered_sources) + "]"

    # Cache successful response (valid for 10 minutes)
    CacheManager.set(cache_key, response, ttl_seconds=600)
    return response


async def prepare_chat_rag_prompt(
    policies: List[Dict[str, Any]],
    query: str,
    db: AsyncSession,
    history: List[Dict[str, str]],
    user_name: str = "there",
) -> Tuple[str, List[Dict[str, Any]], bool]:
    """
    Perform hybrid retrieval and build RAG prompt while DB session is open.
    Returns (prompt_or_chitchat, filtered_chunks, is_chitchat).
    """
    try:
        # 1. Intent check: handle greetings, guidance, and thanks immediately
        chitchat_reply = _get_chitchat_response(query, user_name)
        if chitchat_reply:
            logger.info(f"Chitchat/Guidance query short-circuited: '{query}'")
            return chitchat_reply, [], True
            
        search_query = query
        policy_ids = [p.get("id") for p in policies if p.get("id")]

        # Check cache
        cache_key = CacheManager.get_rag_cache_key(search_query, policy_ids)
        cached_response = CacheManager.get(cache_key)
        if cached_response:
            return "CACHED:" + cached_response, [], True

        # Hybrid Retrieval
        is_comparison = _is_comparison_query(search_query) and len(policies) > 1
        top_k = 10 if is_comparison else 8

        keyword_chunks = _text_search_fallback(search_query, policies, top_k=top_k)
        try:
            vector_chunks = await search_vector_store(db, search_query, policies, top_k=top_k)
        except Exception as ve:
            logger.error(f"Vector search failed: {ve}")
            vector_chunks = []
        try:
            structured_context = await fetch_structured_policy_data(db, policy_ids)
        except Exception as se:
            logger.error(f"Structured context fetch failed: {se}")
            structured_context = ""

        # Reciprocal Rank Fusion (RRF) to combine results
        rrf_scores = {}
        
        for rank, kc in enumerate(keyword_chunks):
            key = kc["text"][:80].lower().strip()
            rrf_scores[key] = {"chunk": kc, "k_rank": rank, "v_rank": None}
            
        for rank, vc in enumerate(vector_chunks):
            key = vc["text"][:80].lower().strip()
            if key not in rrf_scores:
                rrf_scores[key] = {"chunk": vc, "k_rank": None, "v_rank": rank}
            else:
                rrf_scores[key]["v_rank"] = rank

        combined_chunks = []
        for key, info in rrf_scores.items():
            k_score = 1.0 / (60 + info["k_rank"]) if info["k_rank"] is not None else 0.0
            v_score = 1.0 / (60 + info["v_rank"]) if info["v_rank"] is not None else 0.0
            info["chunk"]["rrf_score"] = k_score + v_score
            combined_chunks.append(info["chunk"])
            
        combined_chunks.sort(key=lambda x: x["rrf_score"], reverse=True)
        filtered_chunks = combined_chunks[:top_k]

        prompt = build_chat_prompt(
            query=query,
            retrieved_chunks=filtered_chunks,
            history=history,
            policies=policies,
            user_name=user_name,
            is_comparison=is_comparison,
            structured_context=structured_context
        )

        return prompt, filtered_chunks, False
    except Exception as prep_err:
        logger.error(f"Error in prepare_chat_rag_prompt: {prep_err}", exc_info=True)
        fallback_prompt = (
            f"You are HealthPolicyLens, a healthcare insurance assistant helping there.\n"
            f"User Query: query\n\nAnswer directly based on standard health insurance policy rules."
        )
        return fallback_prompt, [], False


async def run_chat_query_stream_with_prompt(
    prompt: str,
    filtered_chunks: List[Dict[str, Any]],
    is_chitchat: bool = False,
    is_comparison: bool = False,
    policies: List[Dict[str, Any]] = None,
    query: str = "",
) -> AsyncGenerator[str, None]:
    """Stream LLM tokens using a pre-constructed prompt (zero DB dependency during streaming)."""
    start_time = time.time()

    if is_chitchat:
        text_to_yield = prompt[7:] if prompt.startswith("CACHED:") else prompt
        for word in text_to_yield.split(" "):
            yield word + " "
            await asyncio.sleep(0.012)
        return

    full_response_parts: List[str] = []
    first_token_time: Optional[float] = None

    try:
        max_tokens = 700 if is_comparison else 500
        async for token in call_ollama_stream(prompt, num_predict=max_tokens, num_ctx=8192):
            if first_token_time is None:
                first_token_time = time.time() - start_time
                logger.info(f"⚡ Time to first token: {first_token_time:.4f}s")
            full_response_parts.append(token)
            yield token
            
        # Yield structured sources footer so the frontend can render document badges
        if filtered_chunks:
            source_pages = {}
            for chunk in filtered_chunks:
                src = chunk.get("source", "")
                if src:
                    pg = chunk.get("page", 1)
                    if src not in source_pages:
                        source_pages[src] = pg
                    else:
                        source_pages[src] = min(source_pages[src], pg)
            ordered_sources: List[str] = []
            for src, min_pg in source_pages.items():
                ordered_sources.append(f"{src} - Page {min_pg}")
            if ordered_sources:
                sources_tag = "\n[SOURCES:" + "|".join(ordered_sources) + "]"
                yield sources_tag
                full_response_parts.append(sources_tag)
            
        total_time = time.time() - start_time
        logger.info(f"⏱️ Streaming complete in {total_time:.4f}s [First Token: {first_token_time:.4f}s]")

    except Exception as e:
        logger.error(f"Streaming call to Ollama failed (falling back to fast smart QA): {e}")
        
        # Call smart fallback QA engine
        fallback_msg = generate_smart_fallback_answer(query, policies)
        
        # Stream the fallback message token-by-token
        import re
        for token in re.split(r'(\s+)', fallback_msg):
            if token:
                yield token
                await asyncio.sleep(0.01)


def generate_smart_fallback_answer(query: str, policies: List[Dict[str, Any]]) -> str:
    """Generate a highly accurate, rule-based QA response based on extracted fields and document text when LLM fails."""
    if not policies:
        return "I couldn't access your policy data to answer that. Please make sure your policy document is uploaded."

    query_lower = query.lower()
    
    # Identify the primary target document (first one by default, or matched by filename)
    doc = policies[0]
    for p in policies:
        if p.get("filename", "").lower() in query_lower:
            doc = p
            break
            
    fields = doc.get("extracted_fields", [])
    text = doc.get("text", "")
    summary = doc.get("summary", {})
    
    # Helper to look up a field value by name
    def get_field(field_name: str) -> Optional[str]:
        for f in fields:
            if f.get("field_name", "").lower() == field_name.lower():
                return f.get("field_value")
        return None

    # 1. SUM INSURED
    if any(k in query_lower for k in ["sum insured", "sum assured", "coverage limit", "si ", " assured"]):
        si_val = get_field("Sum Insured")
        if si_val:
            return f"The Sum Insured under the policy **{doc.get('filename')}** is **{si_val}**."
        
    # 2. PREMIUM
    if any(k in query_lower for k in ["premium", "cost", "pay"]):
        prem_val = get_field("Premium Amount")
        if prem_val:
            return f"The premium for the policy **{doc.get('filename')}** is **₹{prem_val}**."
            
    # 3. COVERED MEMBERS
    if any(k in query_lower for k in ["who is covered", "covered member", "members", "person", "insured person", "proposer", "policyholder"]):
        members = get_field("Covered Members")
        insured = get_field("Insured Person")
        resp = ""
        if insured:
            resp += f"The primary policyholder/insured person is **{insured}**.\n"
        if members:
            resp += f"The covered members under this policy are: **{members}**."
        if resp:
            return resp

    # 4. WAITING PERIOD
    if any(k in query_lower for k in ["waiting period", "ped", "pre-existing", "pre existing"]):
        ped_wait = get_field("Pre Existing Coverage")
        wait_wp = get_field("Waiting Period")
        resp = "Here are the waiting periods for **" + doc.get('filename') + "**:\n"
        
        # Extract using regex
        import re
        text_clean = re.sub(r'\s+', ' ', text)
        m1 = re.search(r'Pre-existing diseases waiting period.*?(?:Code-Excl01)?[:\s\-]*([\d\s/]+months)', text_clean, re.IGNORECASE)
        ped_reg = m1.group(1).strip() if m1 else None
        
        m2 = re.search(r'Specified Disease/Procedure waiting period.*?[:\s\-]*(\d+\s*months)', text_clean, re.IGNORECASE)
        spec_reg = m2.group(1).strip() if m2 else None
        
        m3 = re.search(r'Initial waiting Period.*?[:\s\-]*(\d+\s*days)', text_clean, re.IGNORECASE)
        init_reg = m3.group(1).strip() if m3 else None
        
        found_any = False
        if ped_reg or ped_wait:
            resp += f"- **Pre-existing diseases waiting period**: {ped_reg or ped_wait}\n"
            found_any = True
        if spec_reg:
            resp += f"- **Specified diseases/procedures waiting period**: {spec_reg}\n"
            found_any = True
        if init_reg or wait_wp:
            resp += f"- **Initial waiting period**: {init_reg or wait_wp}\n"
            found_any = True
            
        if found_any:
            return resp
        if summary.get("waiting_period_summary"):
            return f"Here is the waiting period summary:\n{summary.get('waiting_period_summary')}"

    # 5. EXCLUSIONS
    if any(k in query_lower for k in ["exclusion", "not covered", "exclude", "not pay"]):
        mat = get_field("Maternity Coverage")
        room = get_field("Room Rent Limit")
        resp = "Here are the exclusions and limits for **" + doc.get('filename') + "**:\n"
        if mat:
            resp += f"- **Maternity Coverage**: {mat}\n"
        if room:
            resp += f"- **Room Rent Limit**: {room}\n"
        if summary.get("exclusions_summary"):
            resp += f"\nSpecific exclusions:\n{summary.get('exclusions_summary')}"
            return resp
        if mat or room:
            return resp

    # 6. RENEWAL / EXPIRY / TERM
    if any(k in query_lower for k in ["expiry", "expiration", "renew", "due date", "term", "valid"]):
        term = get_field("Policy Term")
        renew = get_field("Renewal Date")
        resp = ""
        if term:
            resp += f"The Policy Term/Period of Insurance is **{term}**.\n"
        if renew:
            resp += f"The policy is due for renewal on **{renew}**."
        if resp:
            return resp

    # 7. INSURER / POLICY NAME
    if any(k in query_lower for k in ["insurer", "insurance company", "policy name", "product name", "plan"]):
        insurer_name = get_field("Insurer Name")
        pol_name = get_field("Policy Name")
        resp = ""
        if pol_name:
            resp += f"The plan/policy name is **{pol_name}**.\n"
        if insurer_name:
            resp += f"It is issued by **{insurer_name}**."
        if resp:
            return resp

    # 8. NETWORK HOSPITALS
    if any(k in query_lower for k in ["hospital", "network", "cashless"]):
        hosp = get_field("Network Hospitals")
        if hosp:
            return f"The policy covers cashless treatment at **{hosp}** network hospitals."

    # 9. GENERAL TEXT RAG FALLBACK
    from app.services.rag_service import generate_mock_qa_answer
    fallback_rag = generate_mock_qa_answer(query, text)
    if fallback_rag and len(fallback_rag) > 30 and "failed" not in fallback_rag.lower() and "unavailable" not in fallback_rag.lower():
        return fallback_rag

    # 10. Summary fallback
    sum_text = summary.get("summary_text", "")
    if sum_text:
        return f"Here is a summary of the policy **{doc.get('filename')}**:\n\n{sum_text[:400]}..."
        
    return "I couldn't find a specific answer to that in the policy document. Please refer to the Policy Schedule or terms for detailed information."


async def run_chat_query_stream(
    policies: List[Dict[str, Any]],
    query: str,
    db: AsyncSession,
    history: List[Dict[str, str]],
    user_name: str = "there",
    user_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Streaming Conversational RAG execution yielding tokens progressively."""
    prompt, filtered_chunks, is_short = await prepare_chat_rag_prompt(
        policies=policies,
        query=query,
        db=db,
        history=history,
        user_name=user_name
    )
    is_comparison = _is_comparison_query(query) and len(policies) > 1
    async for token in run_chat_query_stream_with_prompt(
        prompt,
        filtered_chunks,
        is_chitchat=is_short,
        is_comparison=is_comparison,
        policies=policies,
        query=query
    ):
        yield token


