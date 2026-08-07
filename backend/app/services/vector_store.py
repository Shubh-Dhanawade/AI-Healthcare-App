"""
Vector Store Service
Manages Qdrant vector database storage and searches.
"""

import json
import re
from typing import List, Dict, Any, Optional
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk
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
            logger.error(f"❌ Qdrant client connection failed: {q_err}.")
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
        
        # Ensure payload index on document_id exists for fast O(log N) filtered queries.
        # Without this index, Qdrant does a full scan over all vectors on every query.
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="document_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD
            )
            logger.debug("Qdrant payload index on 'document_id' is ready.")
        except Exception:
            # Index already exists — ignore the error
            pass
    except Exception as e:
        logger.error(f"Failed to check/create Qdrant collection: {e}")


# ─────────────────────────────────────────
# Indexing Pipeline
# ─────────────────────────────────────────

async def index_chunks_in_vector_stores(
    db: AsyncSession,
    user_id: str,
    doc_id: str,
    chunks: List[DocumentChunk]
) -> None:
    """
    Index policy chunks strictly in Qdrant collection.
    """
    client = get_qdrant_client()
    if not client:
        logger.error("❌ Qdrant client connection unavailable. Cannot index chunks in vector store.")
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
        skipped_zero = 0
        for chunk in chunks:
            emb = json.loads(chunk.embedding)
            
            # Fix 5: Skip zero-vector chunks — they cannot be retrieved via cosine
            # similarity and indicate a failed embedding generation.
            if all(v == 0.0 for v in emb):
                skipped_zero += 1
                logger.warning(f"Skipping zero-vector chunk {chunk.id} (embedding failed) — will not be indexed in Qdrant.")
                continue
            
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
            
        if skipped_zero:
            logger.warning(f"⚠️ Skipped {skipped_zero} zero-vector chunks for document {doc_id}. Check nomic-embed-text availability.")
        
        if points:
            logger.info(f"Uploading {len(points)} points into Qdrant collection '{collection_name}'...")
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"✅ Qdrant indexing complete! ({len(points)} valid vectors stored)")
        else:
            logger.error(f"❌ No valid vectors to upload for document {doc_id} — all chunks were zero-vectors!")
    except Exception as q_err:
        logger.error(f"Failed to upload embeddings to Qdrant: {q_err}")


# ─────────────────────────────────────────
# Retrieval Layer (Qdrant Retrieval Gateway)
# ─────────────────────────────────────────

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


async def search_vector_store(
    db: AsyncSession,
    query: str,
    policies: List[Dict[str, Any]],
    top_k: int = 6
) -> List[Dict[str, Any]]:
    """
    Enterprise hybrid retrieval gateway.
    Resolves vectors strictly using Qdrant (primary).
    Applies re-ranking keyword boosts to elevate exact contextual vocabulary.
    Fetches top_k*2 candidates then re-ranks to return top_k best results.
    """
    if not policies:
        return []
        
    try:
        # Generate search-prefixed query embedding for nomic-embed-text
        query_emb = await generate_single_embedding(f"search_query: {query}")
        
        # ── PRIMARY SEARCH: Qdrant ──
        retrieved = await search_qdrant_store(query_emb, policies, top_k=top_k * 2)
        if retrieved:
            logger.info(f"🟢 [HYBRID-RAG] Retrieved {len(retrieved)} matching chunks from Qdrant")
        else:
            logger.warning("🟡 [HYBRID-RAG] Qdrant search returned no results.")
            
        # ── Apply Keyword Boost Re-ranking ──
        for hit in retrieved:
            hit["score"] = _keyword_boost_score(hit["text"], query, hit["raw_score"])
            
        retrieved.sort(key=lambda x: x["score"], reverse=True)
        return retrieved[:top_k]
    except Exception as search_err:
        logger.error(f"search_vector_store encountered error: {search_err}")
        return []


async def delete_document_from_vector_store(document_id: str) -> None:
    """Delete all chunks associated with a document ID from Qdrant vector database."""
    client = get_qdrant_client()
    if not client:
        return
        
    collection_name = "healthcare_policy_chunks"
    try:
        from qdrant_client.http import models as qmodels
        logger.info(f"Removing vectors for document {document_id} from Qdrant...")
        client.delete(
            collection_name=collection_name,
            points_selector=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=str(document_id)),
                    )
                ]
            )
        )
        logger.info(f"Vectors for document {document_id} removed successfully")
    except Exception as q_err:
        logger.error(f"Failed to delete document vectors from Qdrant: {q_err}")
