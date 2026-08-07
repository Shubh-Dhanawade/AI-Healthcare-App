"""
Cache Manager Service
Disabled Cache Implementation — all in-memory caching has been removed to ensure
every RAG chat, document view, and analysis runs completely fresh and reads directly
from the primary databases.
"""

import hashlib
from typing import Any, Dict, Optional
from loguru import logger

class CacheManager:
    @staticmethod
    def get_cache_key(task: str, text: str) -> str:
        """Generate a stable key (unused for caching now)."""
        digest = hashlib.md5(text[:4000].encode("utf-8")).hexdigest()
        return f"{task}:{digest}"

    @staticmethod
    def get_rag_cache_key(query: str, policy_ids: list) -> str:
        """Generate a stable key (unused for caching now)."""
        id_str = ",".join(sorted(str(i) for i in policy_ids))
        digest = hashlib.md5(f"{query}:{id_str}".encode("utf-8")).hexdigest()
        return f"rag:{digest}"

    @staticmethod
    def get_faiss_index(doc_id: str) -> Optional[Any]:
        """Always return None to bypass cache."""
        return None

    @staticmethod
    def set_faiss_index(doc_id: str, index: Any) -> None:
        """No-op: caching disabled."""
        pass

    @staticmethod
    def clear_faiss_index(doc_id: str) -> None:
        """No-op: caching disabled."""
        pass

    @staticmethod
    def evict_faiss_index(doc_id: str) -> None:
        """No-op: caching disabled."""
        pass

    @staticmethod
    def get_document_text(doc_id: str) -> Optional[str]:
        """Always return None to bypass cache."""
        return None

    @staticmethod
    def set_document_text(doc_id: str, text: str) -> None:
        """No-op: caching disabled."""
        pass

    @staticmethod
    def get_summary(doc_id: str) -> Optional[Dict[str, Any]]:
        """Always return None to bypass cache."""
        return None

    @staticmethod
    def set_summary(doc_id: str, summary_data: Dict[str, Any]) -> None:
        """No-op: caching disabled."""
        pass

    @staticmethod
    def invalidate_summary(doc_id: str) -> None:
        """No-op: caching disabled."""
        pass

    @staticmethod
    def get(key: str) -> Optional[Any]:
        """Always return None to bypass cache."""
        return None

    @staticmethod
    def set(key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """No-op: caching disabled."""
        pass

    @staticmethod
    def clear_all() -> None:
        """No-op: caching disabled."""
        pass
