"""
Cross-Encoder Semantic Reranker

This module solves the 'Lost in the Middle' problem prevalent in LLMs.
Instead of sending 30 loosely-matched chunks to the expensive 70B model (causing hallucinations/distraction), 
we retrieve 30 chunks rapidly via Vector Math, and then execute a slow, precise Cross-Encoder model locally 
to strictly score the relevance from 0-1, explicitly isolating only the TOP 5 chunks.
"""
import logging
import threading
import os
from typing import List, Dict, Any
from app.infra.hardware import HardwareProbe

logger = logging.getLogger(__name__)

class SemanticReranker:
    """
    Applies BGE-Reranker matrices against candidate pairs.
    """
    
    _instances = {}
    _instances_lock = threading.Lock()

    def __init__(self, model_name: str = None):
        """
        Instantiates the PyTorch model onto the local GPU dynamically.
        """
        self.model_name = model_name or os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
        self._model = None
        self._gpu_lock = threading.Lock()
        profile = HardwareProbe.get_profile()
        self.device = profile.get("primary_device", "cpu")
        self.batch_size = profile.get("rerank_batch_size", 4)
        if self.device == "cpu":
            logger.warning("[RERANKER] Running on CPU. Expect higher latency. Install CUDA-enabled PyTorch for GPU acceleration.")
        
        # In a physical deployment, this checks `app.infra.hardware.HardwareProbe`
        # and binds to 'cuda' or 'mps' immediately upon instantiation.
        logger.info(f"[RERANKER] Scaffolded {self.model_name}. Model will lazy-load on first execution.")

    @classmethod
    def get_instance(cls, model_name: str = None):
        name = model_name or os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
        with cls._instances_lock:
            if name not in cls._instances:
                cls._instances[name] = cls(model_name=name)
            return cls._instances[name]
        
    @property
    def model(self):
        """Lazy-loads the huggingface Cross-Encoder model directly into active hardware limits."""
        if self._model is None:
            logger.info(f"[RERANKER] Physically allocating memory for Cross-Encoder: {self.model_name}")
            try:
                from sentence_transformers import CrossEncoder
                
                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info(f"[RERANKER] Successfully bound {self.model_name} to {self.device} tensor mapping. batch={self.batch_size}")
            except ImportError:
                logger.error("[RERANKER] sentence_transformers not installed. Reranking will universally fail.")
                raise
        return self._model
        
    def rerank(self, query: str, context_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Evaluates the linguistic overlap between the exact query and every individual candidate chunk.
        
        Args:
            query (str): The rewritten user prompt.
            context_chunks (List[dict]): The pre-filtered list of up to 30 chunks pulled from Qdrant.
            top_k (int): Exact integer limit for final output delivery to the 70B Synthesis Engine.
            
        Returns:
            List[dict]: The tightly sorted subset of contexts matching exact semantic relevance.
        """
        if not context_chunks:
            return []

        if os.getenv("RERANKER_ENABLED", "true").lower() != "true":
            logger.info("[RERANKER] Disabled via env. Returning top_k from dense retrieval.")
            return context_chunks[:top_k]
            
        # We must align the payloads as a 2D Array mapping [ [Query, Chunk1], [Query, Chunk2] ]
        pairs = [[query, chunk.get("page_content", "")] for chunk in context_chunks]
        
        try:
            # Predict produces an explicit array of numeric values corresponding accurately to input positions
            logger.info(f"[RERANKER] Requesting lock for model inference on {len(pairs)} candidate arrays...")
            with self._gpu_lock:
                scores = self.model.predict(pairs, batch_size=self.batch_size)
            
            # Reattach the raw scores directly to the primary dictionaries
            scored_chunks = []
            for i, chunk in enumerate(context_chunks):
                score = float(scores[i])
                chunk["rerank_score"] = score
                scored_chunks.append(chunk)
                
            # Filter mathematically mapping highest arrays to index 0
            sorted_chunks = sorted(scored_chunks, key=lambda x: x["rerank_score"], reverse=True)
            
            # Truncate
            final_chunks = sorted_chunks[:top_k]
            logger.info(f"[RERANKER] Narrowed search space from {len(context_chunks)} -> {len(final_chunks)} exact chunks.")
            
            if not final_chunks:
                logger.warning("[RERANKER] All candidate arrays failed Semantic Similarity Thresholds. Zero chunks returned.")
                
            return final_chunks
            
        except Exception as e:
            logger.error(f"[RERANKER] Cross-Encoder execution physically crashed: {e}")
            # Failsafe: Revert to the raw retrieved chunks (already sorted by Cosine Distance)
            logger.warning("[RERANKER] Bypassing re-rank mechanism due to error.")
            return context_chunks[:top_k]
