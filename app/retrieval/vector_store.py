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


class QdrantStore:
    """
    Enterprise-grade vector store using Qdrant.
    Supports local disk mode and Qdrant Cloud mode (via env vars).
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QdrantStore, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance._lock = threading.Lock()
        return cls._instance

    def __init__(self, dimension: int = 1024, collection_name: str = "enterprise_rag", path: str = "data/qdrant_storage"):
        if self._initialized:
            return

        self.dimension = dimension
        self.collection_name = collection_name
        self.path = path

        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.is_cloud = bool(self.qdrant_url and self.qdrant_api_key)

        if self.is_cloud:
            self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
            print("[QDRANT] Connected to Qdrant Cloud deployment.")
        else:
            os.makedirs(self.path, exist_ok=True)
            self.client = QdrantClient(path=self.path)
            print("[QDRANT] Using local persistent storage.")

        self._ensure_collection()
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
            print(f"Created new Qdrant Collection: {self.collection_name} with Cosine distance.")

    def clear(self):
        """Hard-resets the active vector memory state."""
        with self._lock:
            if self.is_cloud:
                self.client.delete_collection(collection_name=self.collection_name)
                self._ensure_collection()
                print("Cleared Qdrant Cloud collection.")
                return

            import shutil

            self.client.close()
            if os.path.exists(self.path):
                shutil.rmtree(self.path, ignore_errors=True)
            os.makedirs(self.path, exist_ok=True)
            self.client = QdrantClient(path=self.path)
            self._ensure_collection()
            print("Explicit Qdrant local database cleared.")

    def get_all_documents(self) -> List[str]:
        """Returns unique source names currently stored in vector payloads."""
        sources = set()
        offset = None
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

    @property
    def ntotal(self) -> int:
        return self.client.count(collection_name=self.collection_name).count

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
            print(f"Added {len(chunks)} documents to Qdrant. Total: {self.ntotal}")

    def _build_qdrant_filter(self, metadata_filters: Optional[Dict[str, Dict[str, Any]]]) -> Optional[Filter]:
        if not metadata_filters:
            return None

        conditions = []
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
