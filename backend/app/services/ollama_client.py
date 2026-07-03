"""
Ollama Client Service
Manages HTTP connection pooling and optimized parameters for fast local LLM inference.
GPU-accelerated: passes num_gpu=999 to all requests so Ollama offloads all layers
to the detected Vulkan/ROCm device (AMD Radeon RX 6500M).
"""

import json
from typing import AsyncGenerator, List, Optional
import httpx
from loguru import logger
from app.core.config import settings

# Shared HTTP client for connection pooling
_client_instance: Optional[httpx.AsyncClient] = None

def get_httpx_client() -> httpx.AsyncClient:
    """Return or initialize the shared async HTTP client with custom pooling settings."""
    global _client_instance
    if _client_instance is None or _client_instance.is_closed:
        # Configure limits for connection pooling: pool up to 20 connections
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
        # Windows performance: always use 127.0.0.1 — avoids DNS/IPv6 resolution latency
        base_url = settings.OLLAMA_BASE_URL.replace("localhost", "127.0.0.1")
        _client_instance = httpx.AsyncClient(
            base_url=base_url,
            limits=limits,
            # 300s timeout: GPU inference is fast but streaming long responses takes time.
            # 180s was too short for CPU fallback causing mid-stream disconnects.
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
        )
        logger.info(f"🔌 Shared HTTPX Client initialized for Ollama at {base_url}")
    return _client_instance

async def close_httpx_client() -> None:
    """Close the shared client instance on application shutdown."""
    global _client_instance
    if _client_instance is not None and not _client_instance.is_closed:
        await _client_instance.aclose()
        logger.info("🔌 Shared HTTPX Client closed.")

def parse_keep_alive(val: str):
    """Convert numeric keep alive string to integer, or return string duration."""
    try:
        return int(val)
    except ValueError:
        return val


async def warmup_model(model: Optional[str] = None) -> None:
    """Keep the model resident in VRAM by sending a dummy fast-completion prompt."""
    model_name = model or settings.OLLAMA_MODEL
    logger.info(f"🔥 Warming up model {model_name}...")
    try:
        client = get_httpx_client()
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "keep_alive": parse_keep_alive(settings.OLLAMA_KEEP_ALIVE),
            "options": {
                "num_predict": 1,
                "num_ctx": 128,
                "temperature": 0,
                "num_thread": settings.OLLAMA_NUM_THREAD,
                "num_gpu": settings.OLLAMA_NUM_GPU,
            },
        }
        await client.post("/api/chat", json=payload)
        logger.info("✅ Ollama model is now loaded and resident in VRAM.")
    except Exception as e:
        logger.warning(f"Ollama model warmup skipped: {e}")

async def generate_embeddings_nomic(text: str) -> List[float]:
    """Generate 768-dimensional semantic vector embedding using nomic-embed-text from Ollama."""
    client = get_httpx_client()
    payload = {
        "model": "nomic-embed-text",
        "prompt": text,
        "keep_alive": parse_keep_alive(settings.OLLAMA_KEEP_ALIVE),
        "options": {
            "num_gpu": settings.OLLAMA_NUM_GPU,
        }
    }
    
    try:
        response = await client.post("/api/embeddings", json=payload)
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if embedding:
            return embedding
    except Exception as e:
        logger.warning(f"Failed to generate vector embedding via nomic-embed-text /api/embeddings: {e}. Trying /api/embed fallback...")
        
    # Fallback to /api/embed
    payload_embed = {
        "model": "nomic-embed-text",
        "input": text,
        "keep_alive": parse_keep_alive(settings.OLLAMA_KEEP_ALIVE),
        "options": {
            "num_gpu": settings.OLLAMA_NUM_GPU,
        }
    }
    try:
        response = await client.post("/api/embed", json=payload_embed)
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
    except Exception as fallback_e:
        logger.error(f"Fallback embeddings call also failed: {fallback_e}")
        
    # Return zero vector fallback
    logger.warning("Ollama embeddings service offline. Returning zero vector fallback.")
    return [0.0] * 768

async def call_ollama(
    prompt: str,
    model: Optional[str] = None,
    num_predict: int = 512,
    num_ctx: int = 1536,
) -> str:
    """Call Ollama chat API synchronously with dynamic CPU/GPU layer allocation."""
    client = get_httpx_client()
    model_name = model or settings.OLLAMA_MODEL
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": parse_keep_alive(settings.OLLAMA_KEEP_ALIVE),
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "top_k": 1,
            "top_p": 1.0,
            "num_thread": settings.OLLAMA_NUM_THREAD,
            "num_gpu": settings.OLLAMA_NUM_GPU,
        },
    }
    
    logger.debug(f"Ollama Call: {model_name} (predict={num_predict}, ctx={num_ctx})")
    response = await client.post("/api/chat", json=payload)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "").strip()

async def call_ollama_stream(
    prompt: str,
    model: Optional[str] = None,
    num_predict: int = 250,
    num_ctx: int = 1024,
) -> AsyncGenerator[str, None]:
    """Generate streaming tokens from Ollama with GPU-accelerated speed optimizations.
    
    Key changes vs original:
    - num_ctx reduced 2048→1024: halves prefill time, key for fast first-token delivery
    - num_predict reduced 380→250: chat answers should be concise; prevents runaway generation
    - num_gpu: 999 forces all layers to GPU
    - use_mmap REMOVED: mmap conflicts with Vulkan GPU backend and causes slow cold-loads
    """
    client = get_httpx_client()
    model_name = model or settings.OLLAMA_MODEL
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "keep_alive": parse_keep_alive(settings.OLLAMA_KEEP_ALIVE),
        "options": {
            "temperature": 0,           # Greedy for maximum speed
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "top_k": 1,
            "top_p": 1.0,
            "num_thread": settings.OLLAMA_NUM_THREAD,
            "num_gpu": settings.OLLAMA_NUM_GPU,
        },
    }
    
    logger.debug(f"Ollama Stream Call: {model_name} (predict={num_predict}, ctx={num_ctx})")
    
    # Use build_request + send with stream=True for async token-by-token streaming
    req = client.build_request("POST", "/api/chat", json=payload)
    response = await client.send(req, stream=True)
    try:
        response.raise_for_status()
        async for chunk in response.aiter_lines():
            if chunk:
                try:
                    data = json.loads(chunk)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
                except Exception as parse_e:
                    logger.error(f"Error parsing streaming chunk: {parse_e}")
    except Exception as e:
        logger.error(f"Ollama streaming failure: {e}")
        raise
    finally:
        await response.aclose()
