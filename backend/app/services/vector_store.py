"""
Vector Store Service
Manages local FAISS index files for high-speed top-K vector similarity search.
Integrates with SQLite chunks database.
"""

import os
import json
import re
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk
from app.services.cache_manager import CacheManager
from app.services.embedding_service import generate_single_embedding

def get_faiss_index_path(user_id: str, doc_id: str) -> str:
    """Get the standard path to save/load a document's FAISS index."""
    from app.core.config import settings
    user_upload_dir = os.path.join(settings.UPLOAD_DIR, str(user_id))
    os.makedirs(user_upload_dir, exist_ok=True)
    return os.path.join(user_upload_dir, f"{doc_id}.faiss")

async def build_faiss_index(
    db: AsyncSession,
    user_id: str,
    doc_id: str,
    chunks: List[DocumentChunk]
) -> None:
    """Build a FAISS IndexFlatIP (Inner Product / Cosine Similarity) and save to disk."""
    if not chunks:
        logger.warning(f"No chunks provided to build FAISS index for document {doc_id}")
        return
        
    logger.info(f"Building FAISS index for document {doc_id} with {len(chunks)} chunks...")
    
    # 1. Gather all embedding lists
    embeddings_list = []
    for chunk in sorted(chunks, key=lambda c: c.chunk_index):
        emb = json.loads(chunk.embedding)
        embeddings_list.append(emb)
        
    # 2. Vectorize and normalize for Cosine Similarity (IndexFlatIP)
    vectors = np.array(embeddings_list).astype("float32")
    faiss.normalize_L2(vectors)
    
    # 3. Create FAISS index
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    
    # 4. Save to disk
    index_path = get_faiss_index_path(user_id, doc_id)
    faiss.write_index(index, index_path)
    logger.info(f"✅ FAISS index saved to disk at: {index_path}")
    
    # 5. Cache index in memory
    CacheManager.set_faiss_index(doc_id, index)

async def load_faiss_index(db: AsyncSession, user_id: str, doc_id: str) -> faiss.Index:
    """Load a FAISS index from memory cache or disk fallback. Rebuilds from DB if missing."""
    # Try cache first
    cached_index = CacheManager.get_faiss_index(doc_id)
    if cached_index is not None:
        return cached_index
        
    # Load from disk
    index_path = get_faiss_index_path(user_id, doc_id)
    if not os.path.exists(index_path):
        logger.info(f"FAISS index file missing on disk: {index_path}. Rebuilding from SQLite chunks...")
        # Query database chunks
        result = await db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == doc_id)
        )
        chunks = result.scalars().all()
        if not chunks:
            # If no chunks, check if we have extracted text to generate chunks
            doc_res = await db.execute(select(Document).where(Document.id == doc_id))
            doc = doc_res.scalar_one_or_none()
            if doc and doc.extracted_text:
                logger.info(f"Generating chunks and embeddings for document {doc_id} on-the-fly...")
                from app.services.rag_service import generate_document_chunks
                await generate_document_chunks(doc_id, doc.extracted_text, db)
                # Re-query chunks
                result = await db.execute(
                    select(DocumentChunk)
                    .where(DocumentChunk.document_id == doc_id)
                )
                chunks = result.scalars().all()
            else:
                raise FileNotFoundError(f"FAISS index file not found and no chunks or text available in DB for doc {doc_id}")
        
        if chunks:
            await build_faiss_index(db, user_id, doc_id, chunks)
            
    logger.info(f"Loading FAISS index from disk: {index_path}")
    index = faiss.read_index(index_path)
    CacheManager.set_faiss_index(doc_id, index)
    return index

def _keyword_boost_score(chunk_text: str, query: str, base_score: float) -> float:
    """
    Boost a chunk's similarity score when it contains exact words from the query.
    This ensures specific terms like 'maternity', 'deductible', or 'co-payment'
    rank highly even when their vector cosine score is modest.
    """
    query_words = set(
        w.lower() for w in re.findall(r'\w+', query)
        if len(w) > 3  # Skip short stop words
    )
    if not query_words:
        return base_score
    chunk_lower = chunk_text.lower()
    # Count how many distinct query keywords appear in this chunk
    matched = sum(1 for w in query_words if w in chunk_lower)
    if matched == 0:
        return base_score
    # +0.3 per matched keyword, capped at +0.6 total boost
    boost = min(0.6, matched * 0.3)
    return base_score + boost


async def search_vector_store(
    db: AsyncSession,
    query: str,
    policies: List[Dict[str, Any]],
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """
    Search vector store for top matching chunks across multiple policy documents.
    Pipeline:
      1. FAISS cosine similarity search (retrieves top_k * 2 candidates)
      2. Keyword boost re-ranking (boosts chunks with exact query terms by +0.3)
      3. Returns top_k results after re-ranking

    Returns: List of dicts containing chunk text, source document filename, page, and similarity score.
    """
    if not policies:
        return []
        
    # Generate query embedding
    query_emb = await generate_single_embedding(query)
    query_vec = np.array([query_emb]).astype("float32")
    faiss.normalize_L2(query_vec)
    
    all_hits: List[Dict[str, Any]] = []
    
    for policy in policies:
        doc_id = policy.get("id")
        filename = policy.get("filename", "Policy")
        
        # We need user_id to locate the FAISS index file on disk
        # Fetch document from DB to verify user_id
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc:
            logger.warning(f"Document {doc_id} not found in DB during search_vector_store")
            continue
            
        try:
            # 1. Load index
            index = await load_faiss_index(db, doc.user_id, doc_id)
            
            # 2. Retrieve more candidates than needed (top_k * 2) for re-ranking
            fetch_k = min(top_k * 2, index.ntotal)
            if fetch_k <= 0:
                continue
                
            distances, indices = index.search(query_vec, fetch_k)
            
            # 3. Retrieve chunk text contents from SQLite
            chunk_indices = [int(idx) for idx in indices[0] if idx >= 0]
            if not chunk_indices:
                continue
                
            chunk_res = await db.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == doc_id, DocumentChunk.chunk_index.in_(chunk_indices))
            )
            db_chunks = chunk_res.scalars().all()
            chunk_map = {c.chunk_index: c.text_content for c in db_chunks}

            # ── Auto re-index stale documents (chunked with old 500-char size) ──
            if db_chunks:
                avg_chunk_len = sum(len(c.text_content) for c in db_chunks) / len(db_chunks)
                if avg_chunk_len < 700 and doc.extracted_text:
                    logger.info(
                        f"📦 Document {doc_id} has stale small chunks (avg {avg_chunk_len:.0f} chars < 700). "
                        f"Scheduling background re-index with new chunk_size=800..."
                    )
                    async def _reindex_background(d_id=doc_id, d_text=doc.extracted_text):
                        try:
                            from app.core.database import AsyncSessionLocal
                            from app.services.rag_service import generate_document_chunks
                            async with AsyncSessionLocal() as bg_db:
                                await generate_document_chunks(d_id, d_text, bg_db)
                                await bg_db.commit()
                            # Evict stale FAISS index from cache so next query uses new index
                            CacheManager.evict_faiss_index(d_id)
                            logger.info(f"✅ Background re-index complete for {d_id}")
                        except Exception as re_err:
                            logger.error(f"Background re-index failed for {d_id}: {re_err}")
                    import asyncio
                    asyncio.create_task(_reindex_background())
            
            # 4. Apply keyword boost re-ranking
            for score, chunk_idx in zip(distances[0], indices[0]):
                chunk_idx = int(chunk_idx)
                if chunk_idx in chunk_map:
                    chunk_text_content = chunk_map[chunk_idx]
                    # Parse page number if present in chunk text, e.g. "[Page 1]"
                    page_num = 1
                    match = re.search(r"\[Page (\d+)\]", chunk_text_content)
                    if match:
                        page_num = int(match.group(1))

                    # Apply keyword boost to raw cosine score
                    boosted_score = _keyword_boost_score(chunk_text_content, query, float(score))
                        
                    all_hits.append({
                        "text": chunk_text_content,
                        "source": filename,
                        "page": page_num,
                        "score": boosted_score,
                        "raw_score": float(score),
                        "document_id": doc_id
                    })
        except Exception as e:
            logger.error(f"Error searching FAISS index for document {doc_id}: {e}")
            
    # Sort all hits descending by boosted score, return top top_k
    all_hits.sort(key=lambda x: x["score"], reverse=True)
    return all_hits[:top_k]

