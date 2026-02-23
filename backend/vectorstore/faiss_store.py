"""
Local Vector Store (FAISS) Wrapper

Provides a functional, stateless database wrapper around the Meta FAISS internal library.
This module translates mathematical queries into highly optimized L2 distance array loops.
"""
import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Optional

class FAISSStore:
    """
    A lightweight, prototype wrapper around FAISS for persisting high-dimensional arrays locally.
    [ROADMAP DEPRECATION FLAG]: This local database will be completely replaced by the Qdrant/Pinecone SaaS 
    integration in Phase 8 to enable distributed Enterprise scalability.
    """
    def __init__(self, dimension: int = 1024, index_file: str = "data/vectorstore.faiss", meta_file: str = "data/metadata.json"):
        """
        Initializes the DB mapping boundaries.
        Args:
            dimension (int): Must perfectly match the dimensions of the active embedding model (BAAI=1024).
        """
        self.dimension = dimension
        self.index_path = index_file
        self.metadata_path = meta_file
        self._index = None
        self._metadata = None
        
    def _ensure_loaded(self):
        """
        Lazy-loads the flat byte files exclusively when semantic mapping is explicitly requested by an Agent.
        """
        if self._index is None:
            # Re-initialize the L2 distance base logic struct
            self._index = faiss.IndexFlatL2(self.dimension)
            self._metadata = []
            
            # Load the existing massive index file if the payload exists on the local OS
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            if os.path.exists(self.index_path) and os.path.getsize(self.index_path) > 0 and os.path.exists(self.metadata_path):
                self.load()
                
    @property
    def index(self):
        self._ensure_loaded()
        return self._index
        
    @property
    def metadata(self):
        self._ensure_loaded()
        return self._metadata

    def clear(self):
        """
        Hard-resets the active vector memory state and forcefully deletes the persisted OS files.
        Typically only utilized explicitly when the user requests a "Reset Knowledge Base" trigger.
        """
        self._index = faiss.IndexFlatL2(self.dimension)
        self._metadata = []
        
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        print("Explicit FAISS Vector database cleared.")

    def get_all_documents(self) -> List[str]:
        """
        Iterates over the JSON metadata list to output a consolidated set of unique source names 
        actively present in the retrievable DB mappings.
        """
        if not self.metadata:
            return []
        sources = set()
        for meta in self.metadata:
            if 'source' in meta:
                sources.add(meta['source'])
        return sorted(list(sources))

    def add_documents(self, chunks: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
        """
        Ingests the massively generated textual array chunks mapping directly against their tensor embeddings.
        """
        if not chunks or not embeddings:
            return

        # 1. Cast tensor standard float wrappers into ultra-optimized `float32` numpy representations.
        vectors = np.array(embeddings).astype('float32')
        
        # 2. Append vectors to massive C++ native FAISS array memory index
        self.index.add(vectors)
        
        # 3. Securely map corresponding string payloads to python dictionaries explicitly
        for i, chunk in enumerate(chunks):
            meta = metadatas[i] if i < len(metadatas) else {}
            meta['text'] = chunk  # Attach textual snippet payload directly metadata
            self.metadata.append(meta)
            
        print(f"Added {len(chunks)} structural documents to vector store layer. Memory Total: {self.index.ntotal}")
        self.save()

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Executes a high-speed L2 Distance boundary search over the entire massive index mappings array.
        """
        if self.index.ntotal == 0:
            return []

        # Convert target query standard tensor format float explicitly to FAISS structural bounds
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Dispatch native compiled `.search` algorithm
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                # Attach mathematical retrieval distance to the semantic chunk logic string payload
                result['score'] = float(distances[0][i])
                results.append(result)
                
        return results

    def save(self):
        """Serializes the highly optimized C++ FAISS index maps to local primitive binary byte files on disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Vector Database persistently saved OS file mappings to {self.index_path}")

    def load(self):
        """Loads and inherently deserializes local `.faiss` C++ binaries strictly back into programmatic active system memory."""
        try:
            if not os.path.exists(self.index_path) or os.path.getsize(self.index_path) == 0:
                print(f"Skipping flat DB FAISS extraction logic load: {self.index_path} not found or fully mapped empty.")
                return
            self._index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self._metadata = pickle.load(f)
            print(f"Loaded massive local Vector architecture data store indexing {self._index.ntotal} vector metric arrays.")
        except Exception as e:
            print(f"Catastrophic local OS File mapping error loading vector store files bounds: {e}")
            # Ensure safe failover fallback initialization
            self._index = faiss.IndexFlatL2(self.dimension)
            self._metadata = []
