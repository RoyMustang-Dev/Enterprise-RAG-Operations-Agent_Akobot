"""
Cloud Vector Store (Qdrant) Wrapper

Provides a functional database wrapper around Qdrant vector database.
Replaces the legacy offline FAISS implementation.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)
from typing import List, Dict, Any, Optional
import os
import uuid
import threading
import logging

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Enterprise-grade vector store using Qdrant.
    Supports local disk mode and Qdrant Cloud mode (via env vars).
    """

    _instances = {}
    _instances_lock = threading.Lock()
    _shared_local_client = None
    _shared_local_path = None

    def __new__(cls, dimension: int = 1024, collection_name: str = None, path: str = "data/qdrant_storage", *args, **kwargs):
        actual_collection = collection_name or os.getenv("QDRANT_COLLECTION") or "enterprise_rag"
        with cls._instances_lock:
            if actual_collection not in cls._instances:
                instance = super(QdrantStore, cls).__new__(cls)
                instance._initialized = False
                instance._lock = threading.Lock()
                cls._instances[actual_collection] = instance
            return cls._instances[actual_collection]

    def __init__(self, dimension: int = 1024, collection_name: str = None, path: str = "data/qdrant_storage", tenant_id: str = None, *args, **kwargs):
        if self._initialized:
            return

        self.dimension = dimension
        actual_collection = collection_name or os.getenv("QDRANT_COLLECTION") or "enterprise_rag"
        self.collection_name = actual_collection
        self.path = path
        self.tenant_id = tenant_id
        self.multi_tenant = os.getenv("QDRANT_MULTI_TENANT", "false").lower() == "true"

        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.is_cloud = bool(self.qdrant_url and self.qdrant_api_key)
        if self.qdrant_url and not self.qdrant_api_key:
            logger.warning("[QDRANT] QDRANT_URL set but QDRANT_API_KEY missing. Falling back to local storage.")
        if self.qdrant_api_key and not self.qdrant_url:
            logger.warning("[QDRANT] QDRANT_API_KEY set but QDRANT_URL missing. Falling back to local storage.")

        if self.is_cloud:
            self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
            logger.info("[QDRANT] Connected to Qdrant Cloud deployment.")
        else:
            os.makedirs(self.path, exist_ok=True)
            # Reuse a shared local client to avoid file-lock conflicts across collections.
            if QdrantStore._shared_local_client is None or QdrantStore._shared_local_path != self.path:
                QdrantStore._shared_local_client = QdrantClient(path=self.path)
                QdrantStore._shared_local_path = self.path
            self.client = QdrantStore._shared_local_client
            logger.info("[QDRANT] Using local persistent storage.")

        self._ensure_collection()
        self._ensure_payload_indexes()
        self._initialized = True

    def _ensure_collection(self):
        """Creates the logical collection mapping if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
            logger.info(f"[QDRANT] Created new collection: {self.collection_name} with Cosine distance.")

    def _ensure_payload_indexes(self):
        """Ensure payload indexes exist for common filter fields."""
        fields = ["document_type", "source_domain", "author", "creation_year", "source", "tenant_id"]
        for field in fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                # Ignore if index already exists or backend doesn't support it
                pass

    def clear(self):
        """Hard-resets the active vector memory state."""
        with self._lock:
            if self.is_cloud:
                self.client.delete_collection(collection_name=self.collection_name)
                self._ensure_collection()
                logger.info("[QDRANT] Cleared cloud collection.")
                return

            import shutil

            self.client.close()
            if os.path.exists(self.path):
                shutil.rmtree(self.path, ignore_errors=True)
            os.makedirs(self.path, exist_ok=True)
            self.client = QdrantClient(path=self.path)
            self._ensure_collection()
            logger.info("[QDRANT] Local database cleared.")

    def delete_collection(self, collection_name: str = None):
        """Deletes a specific collection and clears the cached instance."""
        target = collection_name or self.collection_name
        with self._lock:
            self.client.delete_collection(collection_name=target)
        # Remove cached instance so future requests re-init cleanly
        if target in QdrantStore._instances:
            QdrantStore._instances.pop(target, None)

    def get_all_documents(self) -> List[str]:
        """Returns unique source names currently stored in vector payloads."""
        sources = set()
        offset = None
        try:
            while True:
                records, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    if record.payload and "source" in record.payload:
                        sources.add(record.payload["source"])
                if offset is None:
                    break
            return sorted(list(sources))
        except Exception:
            return []

    @property
    def ntotal(self) -> int:
        try:
            return self.client.count(collection_name=self.collection_name).count
        except Exception:
            return 0

    def stats(self) -> Dict[str, Any]:
        return {
            "collection": self.collection_name,
            "mode": "cloud" if self.is_cloud else "local",
            "total_vectors": self.ntotal,
            "documents": self.get_all_documents(),
        }

    def add_documents(self, chunks: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
        if not chunks or not embeddings:
            return

        points = []
        for i, chunk in enumerate(chunks):
            meta = metadatas[i] if i < len(metadatas) else {}
            if self.multi_tenant and self.tenant_id:
                meta["tenant_id"] = self.tenant_id
            meta["page_content"] = chunk

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i],
                    payload=meta,
                )
            )

        with self._lock:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"[QDRANT] Added {len(chunks)} documents. Total: {self.ntotal}")

    def _build_qdrant_filter(self, metadata_filters: Optional[Dict[str, Dict[str, Any]]]) -> Optional[Filter]:
        if not metadata_filters:
            metadata_filters = {}

        conditions = []
        if self.multi_tenant and self.tenant_id:
            conditions.append(FieldCondition(key="tenant_id", match=MatchValue(value=self.tenant_id)))
        for field, cfg in metadata_filters.items():
            if not isinstance(cfg, dict):
                continue
            op = cfg.get("op")
            value = cfg.get("value")
            if value is None:
                continue

            if op == "$eq":
                conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
            elif op == "$in" and isinstance(value, list) and value:
                conditions.append(FieldCondition(key=field, match=MatchAny(any=value)))

        if not conditions:
            return None
        return Filter(must=conditions)

    def search(self, query_embedding: List[float], k: int = 5, metadata_filters: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict]:
        """Executes vector search over current collection with optional metadata filters."""
        if not query_embedding:
            return []

        query_filter = self._build_qdrant_filter(metadata_filters)
        with self._lock:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=k,
            )

            results = []
            for hit in search_result.points:
                result = hit.payload.copy() if hit.payload else {}
                if "text" in result and "page_content" not in result:
                    result["page_content"] = result["text"]
                result["score"] = hit.score
                results.append(result)

            return results
