"""
Cache Manager Service
Provides in-memory caching for prompt templates, document texts, FAISS indices, and LLM responses.
"""

import hashlib
import time
from typing import Any, Dict, Optional
from loguru import logger

# Global in-memory caches
_faiss_index_cache: Dict[str, Any] = {}
_document_text_cache: Dict[str, str] = {}
_summary_cache: Dict[str, Dict[str, Any]] = {}
_general_cache: Dict[str, Any] = {}
_general_cache_expiry: Dict[str, float] = {}

class CacheManager:
    @staticmethod
    def get_cache_key(task: str, text: str) -> str:
        """Generate a stable cache key using MD5 hashing of first 4000 characters."""
        digest = hashlib.md5(text[:4000].encode("utf-8")).hexdigest()
        return f"{task}:{digest}"

    @staticmethod
    def get_rag_cache_key(query: str, policy_ids: list) -> str:
        """Generate a RAG cache key based on query and sorted policy IDs."""
        id_str = ",".join(sorted(str(i) for i in policy_ids))
        digest = hashlib.md5(f"{query}:{id_str}".encode("utf-8")).hexdigest()
        return f"rag:{digest}"

    @staticmethod
    def get_faiss_index(doc_id: str) -> Optional[Any]:
        """Retrieve a cached FAISS index."""
        return _faiss_index_cache.get(doc_id)

    @staticmethod
    def set_faiss_index(doc_id: str, index: Any) -> None:
        """Cache a loaded FAISS index in memory."""
        _faiss_index_cache[doc_id] = index
        logger.info(f"💾 FAISS index for document {doc_id} cached in memory.")

    @staticmethod
    def clear_faiss_index(doc_id: str) -> None:
        """Remove a FAISS index from cache."""
        if doc_id in _faiss_index_cache:
            del _faiss_index_cache[doc_id]
            logger.info(f"🧹 FAISS index for document {doc_id} removed from cache.")

    @staticmethod
    def get_document_text(doc_id: str) -> Optional[str]:
        """Retrieve cached document text."""
        return _document_text_cache.get(doc_id)

    @staticmethod
    def set_document_text(doc_id: str, text: str) -> None:
        """Cache document text."""
        _document_text_cache[doc_id] = text

    @staticmethod
    def get_summary(doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached summary metadata."""
        return _summary_cache.get(doc_id)

    @staticmethod
    def set_summary(doc_id: str, summary_data: Dict[str, Any]) -> None:
        """Cache summary metadata."""
        _summary_cache[doc_id] = summary_data

    @staticmethod
    def invalidate_summary(doc_id: str) -> None:
        """Evict a cached summary so it gets re-fetched from DB on next access."""
        if doc_id in _summary_cache:
            del _summary_cache[doc_id]
            logger.info(f"🧹 Summary cache evicted for document {doc_id}.")

    @staticmethod
    def get(key: str) -> Optional[Any]:
        """Get value from general cache with TTL expiration check."""
        if key in _general_cache:
            expiry = _general_cache_expiry.get(key, 0.0)
            if expiry == 0.0 or expiry > time.time():
                return _general_cache[key]
            else:
                # Expired
                del _general_cache[key]
                if key in _general_cache_expiry:
                    del _general_cache_expiry[key]
        return None

    @staticmethod
    def set(key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in general cache with optional TTL."""
        _general_cache[key] = value
        if ttl_seconds:
            _general_cache_expiry[key] = time.time() + ttl_seconds
        else:
            _general_cache_expiry[key] = 0.0

    @staticmethod
    def clear_all() -> None:
        """Clear all caches."""
        _faiss_index_cache.clear()
        _document_text_cache.clear()
        _summary_cache.clear()
        _general_cache.clear()
        _general_cache_expiry.clear()
        logger.info("🧹 All in-memory caches cleared.")
