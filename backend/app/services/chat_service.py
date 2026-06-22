"""
Chat Service
Orchestrates the entire Conversational RAG pipeline. Handles chitchat routing,
history-based query rewriting, FAISS vector store retrieval, and token streaming.
"""

import asyncio
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
    top_k = 6 if is_comparison else 4
    
    retrieval_start = time.time()
    retrieved_chunks = await search_vector_store(db, search_query, policies, top_k=top_k)
    retrieval_time = time.time() - retrieval_start
    
    # Similarity threshold filtering (threshold = 0.38)
    SIMILARITY_THRESHOLD = 0.38
    filtered_chunks = [c for c in retrieved_chunks if c.get("score", 0.0) >= SIMILARITY_THRESHOLD]
    logger.info(
        f"Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.4f}s. "
        f"Filtered to {len(filtered_chunks)} chunks with score >= {SIMILARITY_THRESHOLD}"
    )
    if retrieved_chunks:
        scores_str = ", ".join(f"{c['source']}(p{c['page']}): {c['score']:.4f}" for c in retrieved_chunks[:3])
        logger.debug(f"Top chunks: {scores_str}")
        
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
    response = await call_ollama(prompt, num_predict=350 if is_comparison else 250)
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
    top_k = 6 if is_comparison else 4
    
    retrieval_start = time.time()
    retrieved_chunks = await search_vector_store(db, search_query, policies, top_k=top_k)
    retrieval_time = time.time() - retrieval_start
    
    # Similarity threshold filtering (threshold = 0.38)
    SIMILARITY_THRESHOLD = 0.38
    filtered_chunks = [c for c in retrieved_chunks if c.get("score", 0.0) >= SIMILARITY_THRESHOLD]
    logger.info(
        f"Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.4f}s. "
        f"Filtered to {len(filtered_chunks)} chunks with score >= {SIMILARITY_THRESHOLD}"
    )
    if retrieved_chunks:
        scores_str = ", ".join(f"{c['source']}(p{c['page']}): {c['score']:.4f}" for c in retrieved_chunks[:3])
        logger.debug(f"Top chunks: {scores_str}")
        
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
        max_tokens = 450 if is_comparison else 280
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
