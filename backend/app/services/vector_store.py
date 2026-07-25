"""
Vector Store Service
Manages local FAISS index files for backward-compatibility,
PostgreSQL pgvector SQL similarity searches, and primary Qdrant collection searches.
"""

import os
import json
import re
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk
from app.services.cache_manager import CacheManager
from app.services.embedding_service import generate_single_embedding
from app.core.config import settings

# ─────────────────────────────────────────
# Qdrant Client Initialization
# ─────────────────────────────────────────

_qdrant_client = None

def get_qdrant_client() -> Optional[Any]:
    """Initialize and retrieve the global Qdrant client connection."""
    global _qdrant_client
    if _qdrant_client is None:
        qdrant_host = settings.QDRANT_HOST
        qdrant_port = settings.QDRANT_PORT
        qdrant_url = settings.QDRANT_URL
        
        try:
            from qdrant_client import QdrantClient
            if qdrant_url:
                _qdrant_client = QdrantClient(url=qdrant_url, timeout=5)
            else:
                _qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=5)
            logger.info("✅ Qdrant client connection established successfully")
        except Exception as q_err:
            logger.warning(f"⚠️ Qdrant client connection failed: {q_err}. Running in pgvector/FAISS fallback mode.")
            _qdrant_client = None
    return _qdrant_client


def init_qdrant_collection() -> None:
    """Ensure the target collection is configured inside Qdrant."""
    client = get_qdrant_client()
    if not client:
        return
        
    collection_name = "healthcare_policy_chunks"
    try:
        from qdrant_client.http import models as qmodels
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        if not exists:
            logger.info(f"Creating Qdrant collection: '{collection_name}' (768 dimensions for nomic-embed-text)...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=768,
                    distance=qmodels.Distance.COSINE
                )
            )
            logger.info(f"✅ Qdrant collection '{collection_name}' initialized.")
    except Exception as e:
        logger.error(f"Failed to check/create Qdrant collection: {e}")


# ─────────────────────────────────────────
# Indexing (Dual-Write) Pipeline
# ─────────────────────────────────────────

def get_faiss_index_path(user_id: str, doc_id: str) -> str:
    """Get the standard path to save/load a document's FAISS index."""
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


async def index_chunks_in_vector_stores(
    db: AsyncSession,
    user_id: str,
    doc_id: str,
    chunks: List[DocumentChunk]
) -> None:
    """
    Simultaneously index policy chunks across all vectors systems (FAISS, pgvector, and Qdrant).
    Ensures data consistency and full hybrid retrieval.
    """
    # 1. Build local FAISS index (for local development fallback)
    await build_faiss_index(db, user_id, doc_id, chunks)
    
    # 2. Ensure pgvector compatibility: update embedding_vector values
    from app.core.database import HAS_PGVECTOR
    for chunk in chunks:
        if chunk.embedding_vector is None:
            # Parse serialized string back into list
            chunk.embedding_vector = json.loads(chunk.embedding)
            
    # 3. Upload to Qdrant collection
    client = get_qdrant_client()
    if not client:
        logger.warning("Qdrant server not connected. Skipping Qdrant upload fallback.")
        return
        
    init_qdrant_collection()
    
    collection_name = "healthcare_policy_chunks"
    
    # Fetch original filename and hash for metadata
    doc_res = await db.execute(select(Document).where(Document.id == doc_id))
    doc = doc_res.scalar_one_or_none()
    original_filename = doc.original_filename if doc else "Policy"
    doc_hash = doc.file_hash if doc else ""
    
    try:
        from qdrant_client.http import models as qmodels
        
        points = []
        for chunk in chunks:
            emb = json.loads(chunk.embedding)
            
            # Detect page number
            page_num = 1
            match = re.search(r"\[Page (\d+)\]", chunk.text_content)
            if match:
                page_num = int(match.group(1))
                
            # Build payload metadata schema
            payload = {
                "user_id": str(user_id),
                "document_id": str(doc_id),
                "page_number": int(page_num),
                "chunk_id": str(chunk.id),
                "chunk_index": int(chunk.chunk_index),
                "document_hash": str(doc_hash),
                "filename": str(original_filename),
                "text": chunk.text_content
            }
            
            points.append(qmodels.PointStruct(
                id=str(chunk.id),
                vector=emb,
                payload=payload
            ))
            
        if points:
            logger.info(f"Uploading {len(points)} points into Qdrant collection '{collection_name}'...")
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info("✅ Qdrant indexing complete!")
    except Exception as q_err:
        logger.error(f"Failed to upload embeddings to Qdrant: {q_err}")


# ─────────────────────────────────────────
# Retrieval Layer (Hybrid Routing Engine)
# ─────────────────────────────────────────

async def load_faiss_index(db: AsyncSession, user_id: str, doc_id: str) -> faiss.Index:
    """Load a FAISS index from memory cache or disk fallback. Rebuilds from DB if missing."""
    cached_index = CacheManager.get_faiss_index(doc_id)
    if cached_index is not None:
        return cached_index
        
    index_path = get_faiss_index_path(user_id, doc_id)
    if not os.path.exists(index_path):
        logger.info(f"FAISS index file missing on disk: {index_path}. Rebuilding from SQLite chunks...")
        result = await db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == doc_id)
        )
        chunks = result.scalars().all()
        if not chunks:
            doc_res = await db.execute(select(Document).where(Document.id == doc_id))
            doc = doc_res.scalar_one_or_none()
            if doc and doc.extracted_text:
                logger.info(f"Generating chunks and embeddings for document {doc_id} on-the-fly...")
                from app.services.rag_service import generate_document_chunks
                await generate_document_chunks(doc_id, doc.extracted_text, db)
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
    """Boost similarity scores of chunks containing exact keywords from the query."""
    query_words = set(
        w.lower() for w in re.findall(r'\w+', query)
        if len(w) >= 3
    )
    if not query_words:
        return base_score
    chunk_lower = chunk_text.lower()
    matched = sum(1 for w in query_words if w in chunk_lower)
    if matched == 0:
        return base_score
    boost = min(1.0, matched * 0.4)
    return base_score + boost


async def search_qdrant_store(
    query_vector: List[float],
    policies: List[Dict[str, Any]],
    top_k: int = 6
) -> List[Dict[str, Any]]:
    """Retrieve top-K matching chunks from Qdrant collection with policy filtering."""
    client = get_qdrant_client()
    if not client:
        return []
        
    collection_name = "healthcare_policy_chunks"
    policy_ids = [p.get("id") for p in policies if p.get("id")]
    
    try:
        from qdrant_client.http import models as qmodels
        
        # Filter queries to target policy IDs
        qfilter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchAny(any=policy_ids)
                )
            ]
        )
        
        if hasattr(client, "query_points"):
            search_res_obj = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=qfilter,
                limit=top_k
            )
            search_res = search_res_obj.points
        else:
            search_res = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=qfilter,
                limit=top_k,
                with_payload=True
            )
        
        hits = []
        for hit in search_res:
            payload = hit.payload or {}
            hits.append({
                "text": payload.get("text", ""),
                "source": payload.get("filename", "Policy"),
                "page": payload.get("page_number", 1),
                "score": float(hit.score),
                "raw_score": float(hit.score),
                "document_id": payload.get("document_id", ""),
                "chunk_id": payload.get("chunk_id", ""),
            })
        return hits
    except Exception as e:
        logger.warning(f"Qdrant search execution failed: {e}")
        return []


async def search_pgvector_store(
    db: AsyncSession,
    query_vector: List[float],
    policies: List[Dict[str, Any]],
    top_k: int = 6
) -> List[Dict[str, Any]]:
    """Retrieve top-K matching chunks from PostgreSQL using pgvector cosine similarity search."""
    from app.core.database import HAS_PGVECTOR
    if not HAS_PGVECTOR:
        return []
        
    policy_ids = [p.get("id") for p in policies if p.get("id")]
    query_vector_str = str(query_vector)
    
    try:
        from sqlalchemy import text
        # Cosine distance operator is '<=>' in pgvector
        # Cosine similarity is 1 - Cosine distance
        sql_query = text(
            "SELECT dc.text_content, dc.chunk_index, dc.document_id, d.original_filename, "
            "1 - (dc.embedding_vector <=> :query_vector::vector) AS similarity "
            "FROM document_chunks dc "
            "JOIN documents d ON dc.document_id = d.id "
            "WHERE dc.document_id = ANY(:policy_ids) "
            "ORDER BY dc.embedding_vector <=> :query_vector::vector "
            "LIMIT :limit"
        )
        
        res = await db.execute(
            sql_query,
            {
                "query_vector": query_vector_str,
                "policy_ids": policy_ids,
                "limit": top_k
            }
        )
        
        hits = []
        for row in res.all():
            text_content, chunk_index, document_id, original_filename, similarity = row
            hits.append({
                "text": text_content,
                "source": original_filename,
                "page": 1,  # Default fallback
                "score": float(similarity) if similarity is not None else 0.0,
                "raw_score": float(similarity) if similarity is not None else 0.0,
                "document_id": document_id,
                "chunk_id": f"pgv_{chunk_index}"
            })
        return hits
    except Exception as e:
        logger.warning(f"pgvector search execution failed: {e}")
        return []


async def search_vector_store(
    db: AsyncSession,
    query: str,
    policies: List[Dict[str, Any]],
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """
    Enterprise hybrid retrieval gateway.
    Resolves vectors using Qdrant (primary), pgvector (secondary SQL), or FAISS (local file fallback).
    Applies re-ranking keyword boosts to elevate exact contextual vocabulary.
    """
    if not policies:
        return []
        
    try:
        # Generate search-prefixed query embedding for nomic-embed-text
        query_emb = await generate_single_embedding(f"search_query: {query}")
        
        # ── 1. PRIMARY SEARCH: Qdrant ──
        retrieved = await search_qdrant_store(query_emb, policies, top_k=top_k * 2)
        if retrieved:
            logger.info(f"🟢 [HYBRID-RAG] Retrieved {len(retrieved)} matching chunks from Qdrant")
        
        # ── 2. SECONDARY SEARCH: pgvector ──
        if not retrieved:
            retrieved = await search_pgvector_store(db, query_emb, policies, top_k=top_k * 2)
            if retrieved:
                logger.info(f"🔵 [HYBRID-RAG] Retrieved {len(retrieved)} matching chunks from pgvector (PostgreSQL)")

        # ── 3. FALLBACK SEARCH: Local FAISS Index files ──
        if not retrieved:
            logger.warning("🟡 [HYBRID-RAG] Qdrant and pgvector unavailable. Running local FAISS fallback search.")
            query_vec = np.array([query_emb]).astype("float32")
            faiss.normalize_L2(query_vec)
            
            for policy in policies:
                doc_id = policy.get("id")
                filename = policy.get("filename", "Policy")
                
                result = await db.execute(select(Document).where(Document.id == doc_id))
                doc = result.scalar_one_or_none()
                if not doc:
                    continue
                    
                try:
                    index = await load_faiss_index(db, doc.user_id, doc_id)
                    fetch_k = min(top_k * 2, index.ntotal)
                    if fetch_k <= 0:
                        continue
                        
                    distances, indices = index.search(query_vec, fetch_k)
                    chunk_indices = [int(idx) for idx in indices[0] if idx >= 0]
                    if not chunk_indices:
                        continue
                        
                    chunk_res = await db.execute(
                        select(DocumentChunk)
                        .where(DocumentChunk.document_id == doc_id, DocumentChunk.chunk_index.in_(chunk_indices))
                    )
                    db_chunks = chunk_res.scalars().all()
                    chunk_map = {c.chunk_index: c.text_content for c in db_chunks}
                    
                    for score, chunk_idx in zip(distances[0], indices[0]):
                        chunk_idx = int(chunk_idx)
                        if chunk_idx in chunk_map:
                            chunk_text_content = chunk_map[chunk_idx]
                            page_num = 1
                            match = re.search(r"\[Page (\d+)\]", chunk_text_content)
                            if match:
                                page_num = int(match.group(1))
                                
                            retrieved.append({
                                "text": chunk_text_content,
                                "source": filename,
                                "page": page_num,
                                "score": float(score),
                                "raw_score": float(score),
                                "document_id": doc_id,
                                "chunk_id": f"faiss_{chunk_idx}"
                            })
                except Exception as fe:
                    logger.error(f"FAISS fallback search failed for {doc_id}: {fe}")
                    
        # ── 4. Apply Keyword Boost Re-ranking ──
        for hit in retrieved:
            hit["score"] = _keyword_boost_score(hit["text"], query, hit["raw_score"])
            
        retrieved.sort(key=lambda x: x["score"], reverse=True)
        return retrieved[:top_k]
    except Exception as search_err:
        logger.warning(f"search_vector_store encountered error (falling back to text search): {search_err}")
        return []
