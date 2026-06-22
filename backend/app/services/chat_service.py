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
    
    # Cache successful response (valid for 10 minutes)
    CacheManager.set(cache_key, response, ttl_seconds=600)
    return response

async def run_chat_query_stream(
    policies: List[Dict[str, Any]],
    query: str,
    db: AsyncSession,
    history: List[Dict[str, str]],
    user_name: str = "there",
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
            
        # Cache full response (valid for 10 minutes)
        full_response = "".join(full_response_parts)
        CacheManager.set(cache_key, full_response, ttl_seconds=600)
        
        total_time = time.time() - start_time
        logger.info(f"⏱️ Streaming complete in {total_time:.4f}s [Retrieval: {retrieval_time:.4f}s, First Token: {first_token_time:.4f}s]")
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        fallback_msg = "\n❌ Failed to generate streamed response. Local Ollama server may be overloaded."
        yield fallback_msg
