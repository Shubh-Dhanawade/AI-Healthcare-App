import asyncio
import os
import sys
import httpx
import time

# Add backend directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings

async def test_ollama(name, options):
    url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    if "localhost" in url:
        url = url.replace("localhost", "127.0.0.1")
        
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": "Hello, write a 1-sentence welcome message."}],
        "stream": False,
        "options": options
    }
    
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            duration = time.time() - start
            print(f"[{name}] SUCCESS in {duration:.2f}s: {content.strip()}")
            return duration
    except Exception as e:
        duration = time.time() - start
        print(f"[{name}] FAILED in {duration:.2f}s: {e}")
        return None

async def main():
    print("Warm up...")
    # Warm up first
    await test_ollama("Warmup", {"num_predict": 10, "num_ctx": 128})
    
    # Test 1: use_mlock=True (current settings in app)
    print("\n--- Running Test 1: use_mlock=True ---")
    await test_ollama("Test 1 (use_mlock=True)", {
        "temperature": 0,
        "num_predict": 50,
        "num_ctx": 1024,
        "num_batch": 1024,
        "use_mmap": True,
        "use_mlock": True,
        "repeat_penalty": 1.0,
        "top_k": 1,
        "top_p": 1.0,
    })
    
    # Test 2: use_mlock=False
    print("\n--- Running Test 2: use_mlock=False ---")
    await test_ollama("Test 2 (use_mlock=False)", {
        "temperature": 0,
        "num_predict": 50,
        "num_ctx": 1024,
        "num_batch": 1024,
        "use_mmap": True,
        "use_mlock": False,
        "repeat_penalty": 1.0,
        "top_k": 1,
        "top_p": 1.0,
    })
    
    # Test 3: use_mlock=False, num_thread=8
    print("\n--- Running Test 3: use_mlock=False, num_thread=8 ---")
    await test_ollama("Test 3 (mlock=False, thread=8)", {
        "temperature": 0,
        "num_predict": 50,
        "num_ctx": 1024,
        "num_batch": 1024,
        "use_mmap": True,
        "use_mlock": False,
        "num_thread": 8,
        "repeat_penalty": 1.0,
        "top_k": 1,
        "top_p": 1.0,
    })

    # Test 4: use_mlock=False, num_thread=4
    print("\n--- Running Test 4: use_mlock=False, num_thread=4 ---")
    await test_ollama("Test 4 (mlock=False, thread=4)", {
        "temperature": 0,
        "num_predict": 50,
        "num_ctx": 1024,
        "num_batch": 1024,
        "use_mmap": True,
        "use_mlock": False,
        "num_thread": 4,
        "repeat_penalty": 1.0,
        "top_k": 1,
        "top_p": 1.0,
    })

if __name__ == "__main__":
    asyncio.run(main())
