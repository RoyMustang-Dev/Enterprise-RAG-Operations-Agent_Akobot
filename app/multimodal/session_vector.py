"""
Ephemeral Session Vector Store

Creates per-session collections that persist for a TTL window (default 24 hours).
"""
import os
import time
import uuid
import logging
from typing import Optional, Tuple

from app.retrieval.vector_store import QdrantStore
from app.infra.database import (
    record_ephemeral_collection,
    list_expired_collections,
    delete_ephemeral_collection_record,
    upsert_session_collection,
    get_session_collection,
    delete_session_collection_by_collection,
)

logger = logging.getLogger(__name__)


class SessionVectorManager:
    def __init__(self, dimension: int = 1024, ttl_hours: Optional[float] = None):
        self.dimension = dimension
        self.ttl_hours = float(os.getenv("EPHEMERAL_TTL_HOURS", ttl_hours or 24))

    def _collection_name(self, session_id: str) -> str:
        return f"session_{session_id}"

    def get_or_create(self, session_id: Optional[str] = None) -> Tuple[str, QdrantStore]:
        if not session_id:
            session_id = str(uuid.uuid4())
        collection_name = self._collection_name(session_id)
        store = QdrantStore(collection_name=collection_name, dimension=self.dimension)
        record_ephemeral_collection(collection_name, time.time())
        upsert_session_collection(session_id, collection_name)
        return session_id, store

    def get_session_collection(self, session_id: str) -> Optional[str]:
        record = get_session_collection(session_id)
        if not record:
            return None
        return record.get("collection_name")

    def cleanup_expired(self):
        expired = list_expired_collections(self.ttl_hours)
        if not expired:
            return

        for collection_name in expired:
            try:
                store = QdrantStore(collection_name=collection_name, dimension=self.dimension)
                store.delete_collection(collection_name)
                logger.info(f"[SESSION VECTOR] Deleted expired collection: {collection_name}")
            except Exception as e:
                logger.warning(f"[SESSION VECTOR] Failed to delete {collection_name}: {e}")
            finally:
                delete_ephemeral_collection_record(collection_name)
                delete_session_collection_by_collection(collection_name)
