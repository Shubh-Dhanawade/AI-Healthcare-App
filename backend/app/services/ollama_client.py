"""
Ollama Client Service
Manages HTTP connection pooling and optimized parameters for fast local LLM inference.
GPU-accelerated: passes num_gpu=999 to all requests so Ollama offloads all layers
to the detected Vulkan/ROCm device (AMD Radeon RX 6500M).

NOTE: Uses /api/generate (completion endpoint) instead of /api/chat because the
fine-tuned model hf.co/kkross/gemma-3-4b-cord19-finetuned-new:latest only exposes
"completion" capability, not "chat" format. /api/generate accepts a plain `prompt`
string and returns `response`, which is compatible with all GGUF completion models.
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
        _client_instance = httpx.AsyncClient(
            base_url=settings.OLLAMA_BASE_URL,
            limits=limits,
            # connect=60s: handles cold model load (~20s) on first request after idle.
            # read=360s: long enough for full streaming responses even if KV cache misses occur.
            # pool=30s: prevents pool exhaustion during concurrent AI pipeline tasks.
            timeout=httpx.Timeout(connect=60.0, read=360.0, write=30.0, pool=30.0),
        )
        logger.info(f"🔌 Shared HTTPX Client initialized for Ollama at {settings.OLLAMA_BASE_URL}")
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
    """Keep the model resident in VRAM by sending a dummy fast-completion prompt.
    Uses keep_alive=-1 so the model stays loaded indefinitely — eliminates the
    ~20s cold-start penalty that causes streaming timeouts during first chat query.
    Uses /api/generate which is supported by all GGUF completion models.
    """
    model_name = model or settings.OLLAMA_MODEL
    logger.info(f"🔥 Warming up model {model_name}...")
    try:
        client = get_httpx_client()
        payload = {
            "model": model_name,
            "prompt": "hi",
            "stream": False,
            # keep_alive read from settings (e.g. 5m to prevent locked RAM)
            "keep_alive": parse_keep_alive(settings.OLLAMA_KEEP_ALIVE),
            "options": {
                "num_predict": 1,
                # CRITICAL: num_ctx must match ALL inference calls to avoid KV cache reload
                "num_ctx": settings.OLLAMA_NUM_CTX,
                "temperature": 0,
                "num_thread": settings.OLLAMA_NUM_THREAD,
                # Pin model weights in RAM — prevents paging to disk under memory pressure
                "use_mmap": True,
                "use_mlock": False,
            },
        }
        if settings.OLLAMA_NUM_GPU is not None:
            payload["options"]["num_gpu"] = settings.OLLAMA_NUM_GPU
        await client.post("/api/generate", json=payload)
        logger.info("✅ Ollama model is now loaded and permanently resident in VRAM.")
    except Exception as e:
        logger.warning(f"Ollama model warmup skipped: {e}")

async def generate_embeddings_nomic(text: str) -> List[float]:
    """Generate 768-dimensional semantic vector embedding using nomic-embed-text from Ollama."""
    client = get_httpx_client()
    payload = {
        "model": "nomic-embed-text",
        "prompt": text,
        "keep_alive": "0s",
        "options": {}
    }
    
    try:
        if settings.OLLAMA_NUM_GPU is not None:
            payload["options"]["num_gpu"] = settings.OLLAMA_NUM_GPU
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
        "keep_alive": "0s",
        "options": {}
    }
    try:
        if settings.OLLAMA_NUM_GPU is not None:
            payload_embed["options"]["num_gpu"] = settings.OLLAMA_NUM_GPU
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
    num_ctx: Optional[int] = None,
) -> str:
    """Call Ollama /api/generate endpoint synchronously with dynamic CPU/GPU layer allocation."""
    client = get_httpx_client()
    model_name = model or settings.OLLAMA_MODEL
    resolved_ctx = num_ctx if num_ctx is not None else settings.OLLAMA_NUM_CTX
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "keep_alive": parse_keep_alive(settings.OLLAMA_KEEP_ALIVE),
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": resolved_ctx,
            "top_k": 1,
            "top_p": 1.0,
            "num_thread": settings.OLLAMA_NUM_THREAD,
            # Pin weights in RAM — prevents paging to disk under 95% RAM pressure
            "use_mmap": True,
            "use_mlock": False,
        },
    }
    
    logger.debug(f"Ollama Call: {model_name} (predict={num_predict}, ctx={resolved_ctx})")
    if settings.OLLAMA_NUM_GPU is not None:
        payload["options"]["num_gpu"] = settings.OLLAMA_NUM_GPU
    response = await client.post("/api/generate", json=payload)
    response.raise_for_status()
    return response.json().get("response", "").strip()

async def call_ollama_stream(
    prompt: str,
    model: Optional[str] = None,
    num_predict: int = 450,
    num_ctx: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """Generate streaming tokens from Ollama /api/generate with GPU-accelerated speed optimizations."""
    client = get_httpx_client()
    model_name = model or settings.OLLAMA_MODEL
    resolved_ctx = num_ctx if num_ctx is not None else settings.OLLAMA_NUM_CTX
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        # keep_alive read from settings (e.g. 5m to prevent locked RAM)
        "keep_alive": parse_keep_alive(settings.OLLAMA_KEEP_ALIVE),
        "options": {
            "temperature": 0,           # Greedy for maximum speed
            "num_predict": num_predict,
            "num_ctx": resolved_ctx,
            "top_k": 1,
            "top_p": 1.0,
            "num_thread": settings.OLLAMA_NUM_THREAD,
            # Pin weights in RAM — prevents paging to disk under 95% RAM pressure
            "use_mmap": True,
            "use_mlock": False,
        },
    }
    
    logger.debug(f"Ollama Stream Call: {model_name} (predict={num_predict}, ctx={resolved_ctx})")
    
    if settings.OLLAMA_NUM_GPU is not None:
        payload["options"]["num_gpu"] = settings.OLLAMA_NUM_GPU
    # Use build_request + send with stream=True for async token-by-token streaming
    req = client.build_request("POST", "/api/generate", json=payload)
    response = await client.send(req, stream=True)
    try:
        response.raise_for_status()
        async for chunk in response.aiter_lines():
            if chunk:
                try:
                    data = json.loads(chunk)
                    # /api/generate returns {"response": "token", "done": false}
                    token = data.get("response", "")
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
