"""
Embedding Service
Provides batched semantic embedding generation using nomic-embed-text on local Ollama.
Uses /api/embed for efficient batch processing with retry logic for resilience.
"""

from typing import List
from loguru import logger
from app.services.ollama_client import get_httpx_client


async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate semantic embeddings for a list of text strings in batch.

    Uses the optimised /api/embed endpoint of Ollama with a 2-attempt retry
    per sub-batch so transient connection issues are recovered automatically.
    Falls back to single-item calls if the batch endpoint misbehaves.
    """
    if not texts:
        return []

    client = get_httpx_client()

    # 50 texts per sub-batch balances payload size vs. number of round-trips
    batch_size = 50
    all_embeddings: List[List[float]] = []

    logger.info(
        f"[EMBEDDING] Generating embeddings for {len(texts)} chunks "
        f"(sub-batch size={batch_size})..."
    )

    for i in range(0, len(texts), batch_size):
        sub_batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        # nomic-embed-text performs better with the search_document prefix
        prefixed = [f"search_document: {t}" for t in sub_batch]

        succeeded = False
        for attempt in range(1, 3):   # up to 2 attempts per sub-batch
            try:
                response = await client.post(
                    "/api/embed",
                    json={
                        "model": "nomic-embed-text",
                        "input": prefixed,
                        "keep_alive": "0s",
                    },
                )
                response.raise_for_status()
                embeddings = response.json().get("embeddings", [])

                if len(embeddings) == len(sub_batch):
                    all_embeddings.extend(embeddings)
                    succeeded = True
                    break
                else:
                    logger.warning(
                        f"[EMBEDDING] Sub-batch {batch_num} size mismatch "
                        f"(got {len(embeddings)}, expected {len(sub_batch)}). "
                        "Falling back to single-item calls."
                    )
                    break   # fall through to single-item fallback below

            except Exception as batch_err:
                if attempt < 2:
                    logger.warning(
                        f"[EMBEDDING] Sub-batch {batch_num} attempt {attempt} failed: {batch_err}. Retrying..."
                    )
                    import asyncio
                    await asyncio.sleep(5)
                else:
                    logger.error(
                        f"[EMBEDDING] Sub-batch {batch_num} failed after 2 attempts: {batch_err}. "
                        "Falling back to single-item calls."
                    )

        if not succeeded:
            # Single-item fallback for this sub-batch
            for txt in sub_batch:
                try:
                    single_emb = await generate_single_embedding(f"search_document: {txt}")
                    all_embeddings.append(single_emb)
                except Exception as single_err:
                    logger.error(f"[EMBEDDING] Single-item embedding failed: {single_err}. Storing zero-vector.")
                    all_embeddings.append([0.0] * 768)

    logger.info(f"[EMBEDDING] ✅ Generated {len(all_embeddings)} embeddings for {len(texts)} chunks.")
    return all_embeddings


async def generate_single_embedding(text: str) -> List[float]:
    """Generate a single 768-dim embedding vector via nomic-embed-text."""
    from app.services.ollama_client import generate_embeddings_nomic
    return await generate_embeddings_nomic(text)

