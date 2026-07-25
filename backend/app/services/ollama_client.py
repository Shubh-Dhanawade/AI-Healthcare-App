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
        # Windows performance: always use 127.0.0.1 — avoids DNS/IPv6 resolution latency
        base_url = settings.OLLAMA_BASE_URL.replace("localhost", "127.0.0.1")
        _client_instance = httpx.AsyncClient(
            base_url=base_url,
            limits=limits,
            # connect=60s: handles cold model load (~20s) on first request after idle.
            # read=300s: long enough for full streaming responses even on CPU fallback.
            # pool=30s: prevents pool exhaustion during concurrent AI pipeline tasks.
            timeout=httpx.Timeout(connect=60.0, read=300.0, write=30.0, pool=30.0),
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
            # keep_alive=-1: model stays loaded indefinitely, no idle eviction.
            "keep_alive": -1,
            "options": {
                "num_predict": 1,
                # IMPORTANT: Use num_ctx=2048 here — same as all inference calls.
                # If warmup uses a different num_ctx (e.g. 128), Ollama reloads the
                # model on every actual inference call (15-20s cold-start per request).
                "num_ctx": 2048,
                "temperature": 0,
                "num_thread": settings.OLLAMA_NUM_THREAD,
                "num_gpu": settings.OLLAMA_NUM_GPU,
            },
        }
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
    num_ctx: int = 2048,
) -> str:
    """Call Ollama /api/generate endpoint synchronously with dynamic CPU/GPU layer allocation.
    Uses /api/generate (completion) which works with all GGUF models including
    fine-tuned models that only support 'completion' capability, not 'chat'.
    
    num_ctx capped at 2048: AMD RX 6500M has 4GB VRAM; model weights use ~3443 MiB,
    leaving ~637 MiB for KV cache. At num_ctx=2048 the KV cache is ~272 MiB (safe).
    At num_ctx=4096 the KV cache is ~544 MiB causing VRAM OOM (HTTP 500).
    """
    client = get_httpx_client()
    model_name = model or settings.OLLAMA_MODEL
    
    payload = {
        "model": model_name,
        "prompt": prompt,
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
    response = await client.post("/api/generate", json=payload)
    response.raise_for_status()
    return response.json().get("response", "").strip()

async def call_ollama_stream(
    prompt: str,
    model: Optional[str] = None,
    num_predict: int = 450,
    num_ctx: int = 2048,
) -> AsyncGenerator[str, None]:
    """Generate streaming tokens from Ollama /api/generate with GPU-accelerated speed optimizations.
    
    Uses /api/generate instead of /api/chat because the fine-tuned model
    hf.co/kkross/gemma-3-4b-cord19-finetuned-new:latest only supports 'completion'
    capability. /api/generate returns streaming JSON with a 'response' field.
    
    Key settings:
    - num_ctx=2048: safe for AMD RX 6500M (4GB VRAM); model weights ~3443 MiB + KV ~272 MiB = ~3758 MiB total
    - num_predict=450: concise, complete policy responses
    - num_gpu=999: forces all layers to GPU (Ollama auto-adjusts to fit VRAM)
    - keep_alive=-1: keeps model permanently in VRAM after first load, avoids cold-start
    """
    client = get_httpx_client()
    model_name = model or settings.OLLAMA_MODEL
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        # keep_alive=-1: model stays loaded in VRAM indefinitely after first use.
        # This prevents the ~20s cold-load penalty on every chat request after idle.
        "keep_alive": -1,
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
