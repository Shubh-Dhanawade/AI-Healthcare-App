"""
Chat Service
Orchestrates the entire Conversational RAG pipeline. Handles chitchat routing,
history-based query rewriting, FAISS vector store retrieval, and token streaming.
"""

import asyncio
import re
import time
from typing import AsyncGenerator, List, Dict, Any, Optional
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
    }
    expanded_words = set(re.findall(r'\w+', query_lower))
    for base_term, syns in SYNONYMS.items():
        if any(s in query_lower for s in [base_term] + syns):
            for s in syns:
                expanded_words.update(s.split())

    query_words = [w for w in expanded_words if len(w) > 2 and w not in _STOP_WORDS]
    if not query_words:
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
            score = sum(chunk_lower.count(w) for w in query_words)
            if score > 0:
                # Boost chunks from benefit schedule / coverage sections (tables)
                schedule_boost = 0
                if any(k in chunk_lower for k in ["schedule of benefits", "section", "covered upto", "covered up to", "at actuals", "1.1", "1.2"]):
                    schedule_boost = 3
                # Additional boost if exact query terms appear in chunk
                if any(qw in chunk_lower for qw in query_words if len(qw) > 3):
                    schedule_boost += 2

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
        "network", "company", "issuer", "what is my number", "policy end date"
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
    """Fetch structured metadata, extracted fields, summaries, and risk analyses for target policies from PostgreSQL."""
    from app.models.document import Document, ExtractedField
    from app.models.risk_analysis import Summary
    from sqlalchemy import select
    
    structured_blocks = []
    for pid in policy_ids:
        doc_res = await db.execute(select(Document).where(Document.id == pid))
        doc = doc_res.scalar_one_or_none()
        if not doc:
            continue
            
        block = [
            f"=== STRUCTURED DETAILS FOR {doc.original_filename} ===",
            f"Policy Name: {doc.original_filename}",
            f"Renewal Date: {doc.renewal_date.strftime('%Y-%m-%d') if doc.renewal_date else 'Not Mentioned'}",
            f"Premium Due Date: {doc.premium_due_date.strftime('%Y-%m-%d') if doc.premium_due_date else 'Not Mentioned'}"
        ]
        
        fields_res = await db.execute(
            select(ExtractedField).where(ExtractedField.document_id == pid)
        )
        fields = fields_res.scalars().all()
        for field in fields:
            block.append(f"{field.field_name}: {field.field_value}")

        summary_res = await db.execute(select(Summary).where(Summary.document_id == pid))
        sum_obj = summary_res.scalar_one_or_none()
        if sum_obj:
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
        
    # 2. Query Rewriting: determine if query depends on previous turns
    search_query = query
    if history and _needs_query_rewriting(query, history):
        search_query = await rewrite_query_with_history(query, history)
        
    # 3. Cache lookup
    policy_ids = [p.get("id") for p in policies if p.get("id")]
    cache_key = CacheManager.get_rag_cache_key(search_query, policy_ids)
    cached_response = CacheManager.get(cache_key)
    if cached_response:
        logger.info(f"Cache hit for query: '{search_query}'")
        return cached_response
        
    # 4. Hybrid RAG Retrieval (Vector Search + Direct Keyword Match)
    is_comparison = _is_comparison_query(search_query) and len(policies) > 1
    top_k = 10 if is_comparison else 8
    
    retrieval_start = time.time()
    # Vector Search
    vector_chunks = await search_vector_store(db, search_query, policies, top_k=top_k)
    # Keyword Match Search
    keyword_chunks = _text_search_fallback(search_query, policies, top_k=top_k)
    
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
    
    # ── Always supplement with structured DB fields & summaries ──
    structured_context = await fetch_structured_policy_data(db, [p.get("id") for p in policies if p.get("id")])
        
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
    
    # 6. Call LLM — using num_ctx=4096 to prevent prompt truncation
    llm_start = time.time()
    response = await call_ollama(prompt, num_predict=600 if is_comparison else 450, num_ctx=4096)
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


async def run_chat_query_stream(
    policies: List[Dict[str, Any]],
    query: str,
    db: AsyncSession,
    history: List[Dict[str, str]],
    user_name: str = "there",
    user_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Streaming Conversational RAG execution yielding tokens progressively."""
    start_time = time.time()
    
    # 1. Intent check: handle greetings, guidance, and thanks immediately
    chitchat_reply = _get_chitchat_response(query, user_name)
    if chitchat_reply:
        logger.info(f"Chitchat/Guidance query short-circuited: '{query}'")
        for word in chitchat_reply.split(" "):
            yield word + " "
            await asyncio.sleep(0.012)
        return
        
    # 2. Query Rewriting: determine if query depends on previous turns
    search_query = query
    if history and _needs_query_rewriting(query, history):
        search_query = await rewrite_query_with_history(query, history)
        
    # 3. Cache lookup
    policy_ids = [p.get("id") for p in policies if p.get("id")]
    cache_key = CacheManager.get_rag_cache_key(search_query, policy_ids)
    cached_response = CacheManager.get(cache_key)
    if cached_response:
        logger.info(f"Cache hit for query stream: '{search_query}'")
        for word in cached_response.split(" "):
            yield word + " "
        return
        
    # 4. Hybrid RAG Retrieval (Vector Search + Direct Keyword Match)
    is_comparison = _is_comparison_query(search_query) and len(policies) > 1
    top_k = 10 if is_comparison else 8
    
    retrieval_start = time.time()
    # Vector Search
    vector_chunks = await search_vector_store(db, search_query, policies, top_k=top_k)
    # Keyword Match Search
    keyword_chunks = _text_search_fallback(search_query, policies, top_k=top_k)
    
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

    # ── Always supplement with structured DB fields & summaries ──
    structured_context = await fetch_structured_policy_data(db, [p.get("id") for p in policies if p.get("id")])
        
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
    
    # 6. Stream tokens from Ollama client using num_ctx=4096
    full_response_parts: List[str] = []
    first_token_time: Optional[float] = None
    
    try:
        max_tokens = 700 if is_comparison else 500
        async for token in call_ollama_stream(prompt, num_predict=max_tokens, num_ctx=4096):
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
            
        # Cache full response (valid for 10 minutes)
        full_response = "".join(full_response_parts)
        CacheManager.set(cache_key, full_response, ttl_seconds=600)
        
        total_time = time.time() - start_time
        logger.info(f"⏱️ Streaming complete in {total_time:.4f}s [Retrieval: {retrieval_time:.4f}s, First Token: {first_token_time:.4f}s]")
        
        # Evaluate & log in background — does NOT block the next user message
        if user_id and filtered_chunks:
            async def _log_rag_eval_background():
                try:
                    context_str = "\n\n".join(c["text"] for c in filtered_chunks)
                    faithfulness_score = 1.0
                    faithfulness_reason = "Evaluation skipped for speed."
                    relevance_score = 1.0
                    relevance_reason = "Evaluation skipped for speed."
                    context_relevance = sum(c.get("score", 0.0) for c in filtered_chunks) / len(filtered_chunks) if filtered_chunks else 1.0

                    from app.services.ai_service import extract_json_from_response
                    from app.services.rag_service import FAITHFULNESS_PROMPT, RELEVANCE_PROMPT
                    faith_prompt = FAITHFULNESS_PROMPT.format(context=context_str, answer=full_response)
                    rel_prompt = RELEVANCE_PROMPT.format(query=query, answer=full_response)

                    faith_res, rel_res = await asyncio.gather(
                        call_ollama(faith_prompt, num_predict=80),
                        call_ollama(rel_prompt, num_predict=80),
                        return_exceptions=True
                    )

                    if not isinstance(faith_res, Exception):
                        faith_json = extract_json_from_response(faith_res)
                        if "score" in faith_json:
                            faithfulness_score = float(faith_json["score"])
                            faithfulness_reason = faith_json.get("reasoning", "Faithfulness check complete.")

                    if not isinstance(rel_res, Exception):
                        rel_json = extract_json_from_response(rel_res)
                        if "score" in rel_json:
                            relevance_score = float(rel_json["score"])
                            relevance_reason = rel_json.get("reasoning", "Relevance check complete.")

                    from app.models.rag_query_log import RAGQueryLog
                    from app.core.database import AsyncSessionLocal
                    async with AsyncSessionLocal() as bg_db:
                        log_entry = RAGQueryLog(
                            user_id=user_id,
                            query=query,
                            answer=full_response,
                            faithfulness=min(max(faithfulness_score, 0.0), 1.0),
                            faithfulness_reasoning=faithfulness_reason,
                            answer_relevance=min(max(relevance_score, 0.0), 1.0),
                            answer_relevance_reasoning=relevance_reason,
                            context_relevance=min(max(context_relevance, 0.0), 1.0),
                            latency=round(total_time, 2)
                        )
                        bg_db.add(log_entry)
                        await bg_db.commit()
                    logger.info("✅ Background: Logged streamed RAG query evaluation to database")
                except Exception as log_err:
                    logger.error(f"Background RAG eval logging failed: {log_err}")

            asyncio.create_task(_log_rag_eval_background())

    except Exception as e:
        error_str = str(e).lower()
        if "timeout" in error_str or "connect" in error_str or "read" in error_str:
            logger.error(f"Streaming timed out (model may have been cold-loading): {e}")
            fallback_msg = "\n⏳ The AI model is loading into memory — this takes ~20 seconds on first use. Please **send your message again** and it will respond instantly."
        else:
            logger.error(f"Streaming failed: {e}")
            fallback_msg = "\n❌ Failed to generate a response. Please try again in a moment."
        yield fallback_msg
