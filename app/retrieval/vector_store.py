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
from qdrant_client.http.exceptions import UnexpectedResponse
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

    def __init__(self, dimension: int = 1024, collection_name: str = None, path: str = "data/qdrant_storage", *args, **kwargs):
        if self._initialized:
            return

        self.dimension = dimension
        actual_collection = collection_name or os.getenv("QDRANT_COLLECTION") or "enterprise_rag"
        self.collection_name = actual_collection
        self.path = path
        self.multi_tenant = os.getenv("QDRANT_MULTI_TENANT", "false").lower() == "true"
        self._payload_index_cache = set()

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

    def _ensure_payload_indexes(self, fields: Optional[List[str]] = None):
        """Ensure payload indexes exist for common filter fields."""
        if fields is None:
            fields = ["document_type", "source_domain", "author", "creation_year", "source", "tenant_id"]
        for field in fields:
            if field in self._payload_index_cache:
                continue
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword",
                )
                self._payload_index_cache.add(field)
            except Exception as e:
                # Ignore if index already exists or backend doesn't support it
                logger.debug(f"[QDRANT] Payload index ensure failed for {field}: {e}")

    def _infer_payload_schema(self, value: Any) -> str:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "float"
        return "keyword"

    def _ensure_payload_indexes_for_filters(self, metadata_filters: Optional[Dict[str, Dict[str, Any]]]):
        if not metadata_filters:
            return
        for field, cfg in metadata_filters.items():
            if not isinstance(cfg, dict):
                continue
            if field in self._payload_index_cache:
                continue
            value = cfg.get("value")
            if value is None:
                continue
            sample = None
            if isinstance(value, list):
                for item in value:
                    if item is not None:
                        sample = item
                        break
            else:
                sample = value
            if sample is None:
                continue
            schema = self._infer_payload_schema(sample)
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )
                self._payload_index_cache.add(field)
            except Exception as e:
                logger.debug(f"[QDRANT] Payload index ensure failed for {field} ({schema}): {e}")

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

    def get_all_documents(self, tenant_id: str = None) -> List[str]:
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
                    if not record.payload:
                        continue
                    if tenant_id and self.multi_tenant:
                        if record.payload.get("tenant_id") != tenant_id:
                            continue
                    if "source" in record.payload:
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

    def _count_for_tenant(self, tenant_id: str) -> int:
        if not tenant_id or not self.multi_tenant:
            return self.ntotal
        try:
            query_filter = self._build_qdrant_filter({}, tenant_id=tenant_id)
            return self.client.count(collection_name=self.collection_name, count_filter=query_filter).count
        except Exception:
            # Fallback: count by scrolling when filtered count isn't supported.
            count = 0
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
                        if record.payload and record.payload.get("tenant_id") == tenant_id:
                            count += 1
                    if offset is None:
                        break
            except Exception:
                return 0
            return count

    def stats(self, tenant_id: str = None) -> Dict[str, Any]:
        if tenant_id and self.multi_tenant:
            return {
                "collection": tenant_id,
                "mode": "cloud" if self.is_cloud else "local",
                "total_vectors": self._count_for_tenant(tenant_id),
                "documents": self.get_all_documents(tenant_id=tenant_id),
            }
        return {
            "collection": self.collection_name,
            "mode": "cloud" if self.is_cloud else "local",
            "total_vectors": self.ntotal,
            "documents": self.get_all_documents(),
        }

    def add_documents(self, chunks: List[str], embeddings: List[List[float]], metadatas: List[Dict], tenant_id: str = None):
        if not chunks or not embeddings:
            return

        points = []
        for i, chunk in enumerate(chunks):
            meta = metadatas[i] if i < len(metadatas) else {}
            if self.multi_tenant and tenant_id:
                meta["tenant_id"] = tenant_id
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

    def _build_qdrant_filter(self, metadata_filters: Optional[Dict[str, Dict[str, Any]]], tenant_id: str = None) -> Optional[Filter]:
        if not metadata_filters:
            metadata_filters = {}

        conditions = []
        if self.multi_tenant and tenant_id:
            conditions.append(FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)))
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

    def search(self, query_embedding: List[float], k: int = 5, metadata_filters: Optional[Dict[str, Dict[str, Any]]] = None, tenant_id: str = None) -> List[Dict]:
        """Executes vector search over current collection with optional metadata filters."""
        if not query_embedding:
            return []

        # Auto-index any new metadata fields coming from the extractor.
        self._ensure_payload_indexes_for_filters(metadata_filters)
        query_filter = self._build_qdrant_filter(metadata_filters, tenant_id=tenant_id)
        with self._lock:
            try:
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=query_filter,
                    limit=k,
                )
            except UnexpectedResponse as e:
                # Safer path: if a payload index is missing, retry once without metadata filters.
                msg = str(e)
                if "Index required but not found" in msg:
                    if "tenant_id" in msg:
                        logger.warning("[QDRANT] tenant_id index missing; creating index and retrying once.")
                        self._ensure_payload_indexes(fields=["tenant_id"])
                        fallback_filter = query_filter
                    else:
                        # Drop dynamic metadata filters but keep tenant isolation if enabled.
                        logger.warning("[QDRANT] Missing payload index for metadata filter. Retrying without metadata filters.")
                        fallback_filter = self._build_qdrant_filter({}, tenant_id=tenant_id)
                    try:
                        search_result = self.client.query_points(
                            collection_name=self.collection_name,
                            query=query_embedding,
                            query_filter=fallback_filter,
                            limit=k,
                        )
                    except Exception:
                        raise
                else:
                    raise

            results = []
            for hit in search_result.points:
                result = hit.payload.copy() if hit.payload else {}
                if "text" in result and "page_content" not in result:
                    result["page_content"] = result["text"]
                result["score"] = hit.score
                results.append(result)

            return results
