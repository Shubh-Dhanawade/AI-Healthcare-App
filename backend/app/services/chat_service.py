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
# Similarity threshold — lowered to capture more relevant chunks.
# Higher values (0.38+) were causing all chunks to be filtered out.
# ─────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.10


def _text_search_fallback(query: str, policies: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Fast TF-IDF keyword fallback when FAISS vector retrieval fails or returns no results.
    Chunks the extracted policy text and ranks by keyword overlap with the query.
    """
    query_words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
    if not query_words:
        return []

    hits: List[Dict[str, Any]] = []

    for policy in policies:
        text = policy.get("text", "") or ""
        filename = policy.get("filename", "Policy")
        if not text:
            continue

        # Simple 400-char chunks with 50-char overlap
        chunk_size = 400
        overlap = 50
        start = 0
        chunks = []
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap

        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = sum(chunk_lower.count(w) for w in query_words)
            if score > 0:
                hits.append({
                    "text": chunk,
                    "source": filename,
                    "page": 1,
                    "score": float(score),
                    "document_id": policy.get("id", "")
                })

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:top_k]

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
        
    # 4. RAG Retrieval using FAISS
    is_comparison = _is_comparison_query(search_query) and len(policies) > 1
    top_k = 6 if is_comparison else 5
    
    retrieval_start = time.time()
    retrieved_chunks = await search_vector_store(db, search_query, policies, top_k=top_k)
    retrieval_time = time.time() - retrieval_start
    
    # Similarity threshold filtering — lowered to 0.10 to capture more relevant chunks
    filtered_chunks = [c for c in retrieved_chunks if c.get("score", 0.0) >= SIMILARITY_THRESHOLD]
    logger.info(
        f"Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.4f}s. "
        f"Filtered to {len(filtered_chunks)} chunks with score >= {SIMILARITY_THRESHOLD}"
    )
    if retrieved_chunks:
        scores_str = ", ".join(f"{c['source']}(p{c['page']}): {c['score']:.4f}" for c in retrieved_chunks[:3])
        logger.debug(f"Top chunks: {scores_str}")

    # ── Fallback: if FAISS returned nothing, use direct keyword search over raw policy text ──
    if not filtered_chunks:
        logger.warning("FAISS returned no results above threshold — falling back to keyword text search")
        filtered_chunks = _text_search_fallback(search_query, policies, top_k=top_k)
        if filtered_chunks:
            logger.info(f"Keyword fallback found {len(filtered_chunks)} chunks")
        
    # 5. Build prompt
    prompt = build_chat_prompt(
        query=query,
        retrieved_chunks=filtered_chunks,
        history=history,
        policies=policies,
        user_name=user_name,
        is_comparison=is_comparison
    )
    
    # 6. Call LLM
    llm_start = time.time()
    response = await call_ollama(prompt, num_predict=500 if is_comparison else 350)
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
    
    # Dynamic RAG Evaluation Logging
    if user_id and filtered_chunks:
        try:
            context_str = "\n\n".join(c["text"] for c in filtered_chunks)
            faithfulness_score = 1.0
            faithfulness_reason = "No context retrieved."
            relevance_score = 1.0
            relevance_reason = "Answer relevance check completed."
            context_relevance = sum(c.get("score", 0.0) for c in filtered_chunks) / len(filtered_chunks) if filtered_chunks else 1.0
            
            from app.services.ai_service import FAITHFULNESS_PROMPT, RELEVANCE_PROMPT, extract_json_from_response
            faith_prompt = FAITHFULNESS_PROMPT.format(context=context_str, answer=response)
            rel_prompt = RELEVANCE_PROMPT.format(query=query, answer=response)
            
            faith_res, rel_res = await asyncio.gather(
                call_ollama(faith_prompt, num_predict=128),
                call_ollama(rel_prompt, num_predict=128),
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
            log_entry = RAGQueryLog(
                user_id=user_id,
                query=query,
                answer=response,
                faithfulness=min(max(faithfulness_score, 0.0), 1.0),
                faithfulness_reasoning=faithfulness_reason,
                answer_relevance=min(max(relevance_score, 0.0), 1.0),
                answer_relevance_reasoning=relevance_reason,
                context_relevance=min(max(context_relevance, 0.0), 1.0),
                latency=round(total_time, 2)
            )
            db.add(log_entry)
            await db.commit()
            logger.info("✅ Logged RAG query evaluation to database successfully")
        except Exception as log_err:
            logger.error(f"Error logging RAG query evaluation: {log_err}")

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
        
    # 4. RAG Retrieval using FAISS
    is_comparison = _is_comparison_query(search_query) and len(policies) > 1
    top_k = 6 if is_comparison else 5
    
    retrieval_start = time.time()
    retrieved_chunks = await search_vector_store(db, search_query, policies, top_k=top_k)
    retrieval_time = time.time() - retrieval_start
    
    # Similarity threshold filtering — lowered to 0.10 to capture more relevant chunks
    filtered_chunks = [c for c in retrieved_chunks if c.get("score", 0.0) >= SIMILARITY_THRESHOLD]
    logger.info(
        f"Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.4f}s. "
        f"Filtered to {len(filtered_chunks)} chunks with score >= {SIMILARITY_THRESHOLD}"
    )
    if retrieved_chunks:
        scores_str = ", ".join(f"{c['source']}(p{c['page']}): {c['score']:.4f}" for c in retrieved_chunks[:3])
        logger.debug(f"Top chunks: {scores_str}")

    # ── Fallback: if FAISS returned nothing, use direct keyword search over raw policy text ──
    if not filtered_chunks:
        logger.warning("FAISS returned no results above threshold — falling back to keyword text search")
        filtered_chunks = _text_search_fallback(search_query, policies, top_k=top_k)
        if filtered_chunks:
            logger.info(f"Keyword fallback found {len(filtered_chunks)} chunks")
        
    # 5. Build prompt
    prompt = build_chat_prompt(
        query=query,
        retrieved_chunks=filtered_chunks,
        history=history,
        policies=policies,
        user_name=user_name,
        is_comparison=is_comparison
    )
    
    # 6. Stream tokens from Ollama client
    full_response_parts: List[str] = []
    first_token_time: Optional[float] = None
    
    try:
        max_tokens = 550 if is_comparison else 380
        async for token in call_ollama_stream(prompt, num_predict=max_tokens):
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
        
        # Evaluate & log in background so streaming is not delayed for the user
        if user_id and filtered_chunks:
            try:
                context_str = "\n\n".join(c["text"] for c in filtered_chunks)
                faithfulness_score = 1.0
                faithfulness_reason = "No context retrieved."
                relevance_score = 1.0
                relevance_reason = "Answer relevance check completed."
                context_relevance = sum(c.get("score", 0.0) for c in filtered_chunks) / len(filtered_chunks) if filtered_chunks else 1.0
                
                from app.services.ai_service import FAITHFULNESS_PROMPT, RELEVANCE_PROMPT, extract_json_from_response
                faith_prompt = FAITHFULNESS_PROMPT.format(context=context_str, answer=full_response)
                rel_prompt = RELEVANCE_PROMPT.format(query=query, answer=full_response)
                
                faith_res, rel_res = await asyncio.gather(
                    call_ollama(faith_prompt, num_predict=128),
                    call_ollama(rel_prompt, num_predict=128),
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
                db.add(log_entry)
                await db.commit()
                logger.info("✅ Logged streamed RAG query evaluation to database successfully")
            except Exception as log_err:
                logger.error(f"Error logging streamed RAG query evaluation: {log_err}")
                
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        fallback_msg = "\n❌ Failed to generate streamed response. Local Ollama server may be overloaded."
        yield fallback_msg
