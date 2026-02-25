"""
Cloud Vector Store (Qdrant) Wrapper

Provides a functional database wrapper around Qdrant vector database.
Replaces the legacy offline FAISS implementation.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List, Dict
import os
import uuid
import threading

class QdrantStore:
    """
    Enterprise-grade vector store using Qdrant.
    It operates in local disk mode by default for zero-cost scalability, but is ready for Qdrant Cloud API.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QdrantStore, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance._lock = threading.Lock()
        return cls._instance

    def __init__(self, dimension: int = 1024, collection_name: str = "enterprise_rag", path: str = "data/qdrant_storage"):
        """
        Initializes the DB mapping boundaries.
        Args:
            dimension (int): Must perfectly match the dimensions of the active embedding model (BAAI=1024).
            collection_name (str): The logical namespace for tenant separation.
            path (str): The local disk path to persist Qdrant data.
        """
        if self._initialized:
            return
            
        self.dimension = dimension
        self.collection_name = collection_name
        self.path = path
        
        # Ensure directory
        os.makedirs(self.path, exist_ok=True)
        
        # Initialize the client pointing to a local directory for zero-cost operation
        self.client = QdrantClient(path=self.path)
        
        # Ensure collection exists
        self._ensure_collection()
        self._initialized = True
            
    def _ensure_collection(self):
        """Creates the logical collection mapping if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
            )
            print(f"Created new Qdrant Collection: {self.collection_name} with Cosine L2 Space.")

    def clear(self):
        """
        Hard-resets the active vector memory state and forcefully deletes the persisted OS files.
        Typically only utilized explicitly when the user requests a "Reset Knowledge Base" trigger.
        """
        with self._lock:
            import shutil
            # 1. Release the active SQLite file handlers
            self.client.close()
            
            # 2. Physically wipe the local disk directory
            if os.path.exists(self.path):
                shutil.rmtree(self.path, ignore_errors=True)
            os.makedirs(self.path, exist_ok=True)
            
            # 3. Transparently Re-initialize the Client
            self.client = QdrantClient(path=self.path)
            self._ensure_collection()
            print("Explicit Qdrant Vector database cleared.")

    def get_all_documents(self) -> List[str]:
        """
        Iterates over the metadata list to output a consolidated set of unique source names 
        actively present in the retrievable DB mappings.
        """
        sources = set()
        offset = None
        while True:
            response = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            records, offset = response
            for record in records:
                if record.payload and 'source' in record.payload:
                    sources.add(record.payload['source'])
            if offset is None:
                break
        return sorted(list(sources))

    @property
    def ntotal(self) -> int:
        """Helper to seamlessly replace Faiss index.ntotal reads"""
        return self.client.count(collection_name=self.collection_name).count

    def add_documents(self, chunks: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
        """
        Ingests the massively generated textual array chunks mapping directly against their tensor embeddings.
        """
        if not chunks or not embeddings:
            return

        points = []
        for i, chunk in enumerate(chunks):
            meta = metadatas[i] if i < len(metadatas) else {}
            meta['page_content'] = chunk  # Attach textual snippet payload directly
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload=meta
            )
            points.append(point)
            
        with self._lock:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Added {len(chunks)} structural documents to Qdrant vector store layer. Memory Total: {self.ntotal}")

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Executes a high-speed L2 Distance boundary search over the entire massive index mappings array.
        """
        with self._lock:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=k
            )
            
            results = []
            for hit in search_result.points:
                result = hit.payload.copy() if hit.payload else {}
                # Backwards compatibility check
                if 'text' in result and 'page_content' not in result:
                    result['page_content'] = result['text']
                
                # Attach mathematical retrieval distance to the semantic chunk logic string payload
                result['score'] = hit.score
                results.append(result)
                    
            return results
