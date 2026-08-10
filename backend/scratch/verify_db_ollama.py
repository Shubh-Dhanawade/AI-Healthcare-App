import asyncio
import httpx
import json
import traceback
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select
from app.core.config import settings

async def verify_db_ollama():
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == '7fd1d803-bd6d-4be4-9e9b-3c915022ba16')
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc:
        print("Document not found.")
        return
        
    text = doc.extracted_text
    print(f"Loaded document: {doc.original_filename}")
    
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    if "localhost" in url:
        url = url.replace("localhost", "127.0.0.1")
        
    prompt = f"""Analyze the following health insurance policy document and extract a list of up to 6 major medical treatments...
DOCUMENT:
{text[:10000]}"""

    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0,
            "num_predict": 150,
            "num_ctx": 2048,
            "num_thread": settings.OLLAMA_NUM_THREAD,
            "use_mmap": True,
            "use_mlock": False,
        }
    }
    if settings.OLLAMA_NUM_GPU is not None:
        payload["options"]["num_gpu"] = settings.OLLAMA_NUM_GPU

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            print(f"Response status: {resp.status_code}")
            if resp.status_code != 200:
                print(f"Error body: {resp.text}")
            else:
                print(f"Success: {resp.json().get('response')[:200]}")
    except Exception as e:
        print("Exception occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_db_ollama())
