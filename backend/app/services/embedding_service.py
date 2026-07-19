"""
Embedding Service
Provides batched semantic embedding generation using nomic-embed-text on local Ollama,
leveraging parallel processing to significantly reduce upload-to-indexing latency.
"""

from typing import List
from loguru import logger
from app.services.ollama_client import get_httpx_client

async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate semantic embeddings for a list of text strings in batch.
    Uses the optimized /api/embed endpoint of Ollama.
    """
    if not texts:
        return []
        
    client = get_httpx_client()
    
    # Process in sub-batches of 20 to prevent huge payload issues
    batch_size = 20
    all_embeddings: List[List[float]] = []
    
    logger.info(f"Generating embeddings for {len(texts)} chunks in batches of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        sub_batch = texts[i:i+batch_size]
        payload = {
            "model": "nomic-embed-text",
            "input": sub_batch
        }
        
        try:
            response = await client.post("/api/embed", json=payload)
            response.raise_for_status()
            embeddings = response.json().get("embeddings")
            if embeddings and len(embeddings) == len(sub_batch):
                all_embeddings.extend(embeddings)
            else:
                # Fallback to single-call if batch size mismatch or empty
                logger.warning(f"Batch embedding returned mismatched size. Falling back to single-item queries.")
                for text in sub_batch:
                    single_emb = await generate_single_embedding(text)
                    all_embeddings.append(single_emb)
        except Exception as batch_err:
            logger.error(f"Batch embedding failed for sub-batch {i // batch_size}: {batch_err}. Falling back to single-item queries.")
            # Fallback to single-call on failure
            for text in sub_batch:
                try:
                    single_emb = await generate_single_embedding(text)
                    all_embeddings.append(single_emb)
                except Exception as single_err:
                    logger.error(f"Single-item embedding fallback failed: {single_err}")
                    all_embeddings.append([0.0] * 768)  # Zero vector fallback
                    
    logger.info(f"✅ Generated {len(all_embeddings)} embeddings successfully.")
    return all_embeddings

async def generate_single_embedding(text: str) -> List[float]:
    """Generate a single embedding vector."""
    from app.services.ollama_client import generate_embeddings_nomic
    return await generate_embeddings_nomic(text)
