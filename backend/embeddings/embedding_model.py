"""
Semantic Embedding Generation Layer

Initializes and proxies the massive BAAI/bge-large-en-v1.5 transformer model.
This file is the exact computational layer that translates human semantic strings 
into 1024-dimensional floating-point tensors capable of mathematical comparison.
"""
from sentence_transformers import SentenceTransformer
from typing import List, Union
from functools import lru_cache

class EmbeddingModel:
    """
    Singleton class to handle dense text embedding generations using the HuggingFace sentence-transformers library.
    
    [DESIGN ARCHITECTURE]: 
    Uses `BAAI/bge-large-en-v1.5` for high-quality enterprise retrieval. 
    It is currently instantiated locally in system memory (requires ~1.5GB RAM).
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self._model = None
        
    @property
    def model(self):
        """
        Lazy-loads the massive 1.5GB transformer model entirely into RAM exclusively on the first function call,
        drastically reducing the startup time for fast API routing sequences.
        """
        if self._model is None:
            print(f"Loading embedding model into Active Memory: {self.model_name}...")
            
            import os
            import logging
            
            # -------------------------------------------------------------------------
            # 1. Silence HuggingFace Authentication Warnings
            # -------------------------------------------------------------------------
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                try:
                    from huggingface_hub import login
                    login(token=hf_token)
                except Exception:
                    pass
                    
            # -------------------------------------------------------------------------
            # 2. Silence BAAI bge-large 'Unexpected Layer' Warnings
            # -------------------------------------------------------------------------
            # The BAAI model natively contains legacy layers that trigger massive console spam 
            # if the transformers logging module isn't explicitly clamped.
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            
            # -------------------------------------------------------------------------
            # 3. Model Instantiation Phase
            # -------------------------------------------------------------------------
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            print("Mathematical Model loaded successfully into Active Context.")
        return self._model

    @lru_cache(maxsize=1024)
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates a 1024-dimensional mathematical embedding for a single text string.
        Utilizes an LRU Cache to instantly return vectors for perfectly identical repetitive questions.
        
        Args:
            text (str): Input conversational query text.
            
        Returns:
            List[float]: The 1024-dimensional vector embedding tensor array.
        """
        if not text:
            return []
            
        # TODO Phase 8: Add `normalize_embeddings=True` to enforce perfect L2 Cosine Spheres.
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a massive List array of text strings (Ingestion batch processing).
        
        Args:
            texts (List[str]): List of parsed document chunks.
            
        Returns:
            List[List[float]]: Array of 1024-dimensional vector embedding arrays.
        """
        if not texts:
            return []
            
        # TODO Phase 8: Add `normalize_embeddings=True` across the batch tensor generations
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
