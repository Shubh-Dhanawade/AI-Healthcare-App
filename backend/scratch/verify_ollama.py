import asyncio
import httpx
import json
import sys
from app.core.config import settings

async def test_ollama():
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    if "localhost" in url:
        url = url.replace("localhost", "127.0.0.1")
        
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": "Summarize this: hi",
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0,
            "num_predict": 100,
            "num_ctx": 8192,
            "num_thread": settings.OLLAMA_NUM_THREAD,
            "use_mmap": True,
            "use_mlock": False,
        }
    }
    if settings.OLLAMA_NUM_GPU is not None:
        payload["options"]["num_gpu"] = settings.OLLAMA_NUM_GPU
        
    print(f"Sending request to {url} with model {settings.OLLAMA_MODEL}...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            print(f"Response status: {resp.status_code}")
            if resp.status_code != 200:
                print(f"Error body: {resp.text}")
            else:
                print(f"Success: {resp.json().get('response')}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama())
