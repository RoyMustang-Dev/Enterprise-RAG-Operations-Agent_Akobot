"""
Semantic Embedding Generation Layer

Initializes and proxies the massive BAAI/bge-large-en-v1.5 transformer model.
This file is the exact computational layer that translates human semantic strings 
into 1024-dimensional floating-point tensors capable of mathematical comparison.
"""
import os
from dotenv import load_dotenv
load_dotenv(override=True)

import torch
import requests
import concurrent.futures
from sentence_transformers import SentenceTransformer
from typing import List, Union
from functools import lru_cache
import logging
from app.infra.hardware import HardwareProbe

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Singleton class to handle dense text embedding generations safely across varied client hardware architectures.
    
    [PHASE 13 & 14 ARCHITECTURE]: 
    1. It intercepts OPENAI_API_KEY to bypass massive RAM requirements via Cloud APIs.
    2. It detects CUDA/MPS to aggressively batch-encode via GPUs natively.
    3. It falls back to multi-threaded CPU Executor concurrency ensuring no single threading bottlenecks.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self._model = None
        
        # Phase 14: Cloud Fallback
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Phase 13: Hardware Probing
        if not self.openai_api_key:
            profile = HardwareProbe.get_profile()
            self.device = profile.get("primary_device", "cpu")
            self.num_cores = profile.get("cpu_cores", os.cpu_count() or 4)
            self.batch_size = profile.get("embedding_batch_size", 32)
            logger.info(f"[EMBEDDING ENGINE] Hardware Probe: device={self.device} cores={self.num_cores} batch={self.batch_size}")
            if self.device == "cpu":
                logger.warning("[EMBEDDING ENGINE] Running on CPU. Embedding throughput will be slower.")
        else:
            self.batch_size = 32
            logger.info("[EMBEDDING ENGINE] Cloud Override -> Routing to OpenAI API text-embedding-3-small.")
        
    @property
    def model(self):
        """
        Lazy-loads the massive 1.5GB transformer model entirely into RAM exclusively on the first function call,
        drastically reducing the startup time for fast API routing sequences.
        """
        if self.openai_api_key:
            return None # Prevent allocating 1.5GB of RAM by bypassing PyTorch altogether
            
        if self._model is None:
            logger.info(f"Loading local tensor mathematics model into Hardware Active Memory: {self.model_name}...")
            
            import os
            import logging
            
            # -------------------------------------------------------------------------
            # 1. Silence BAAI bge-large 'Unexpected Layer' Warnings
            # -------------------------------------------------------------------------
            # The BAAI model natively contains legacy layers that trigger massive console spam 
            # if the transformers logging module isn't explicitly clamped.
            import logging
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            
            # -------------------------------------------------------------------------
            # 3. Model Instantiation Phase
            # -------------------------------------------------------------------------
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Mathematical Model loaded successfully into {self.device} Hardware Context.")
        return self._model

    @lru_cache(maxsize=1000)
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
            
        if self.openai_api_key:
            # Route singular query directly to the cloud preserving 1024 dimensional bounds
            headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
            payload = {"input": [text], "model": "text-embedding-3-small", "dimensions": 1024}
            response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
            
        # TODO Phase 8: Add `normalize_embeddings=True` to enforce perfect L2 Cosine Spheres.
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
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
            
        if self.openai_api_key:
            # Sub-batch to avoid OpenAI API payload limits inherently
            all_embeddings = []
            headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
            
            # OpenAI strictly limits array sizes per REST call, usually safe up to 2000 passages
            batch_size = 500
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                payload = {"input": batch_texts, "model": "text-embedding-3-small", "dimensions": 1024}
                response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload)
                response.raise_for_status()
                
                # Align exact array positioning robustly
                data = sorted(response.json()["data"], key=lambda x: x["index"])
                all_embeddings.extend([item["embedding"] for item in data])
                
            return all_embeddings

        # Phase 13: Hardware Execution Sub-Routing
        if self.device in ["cuda", "mps"]:
            # Native GPU Tensor processing devours entire arrays aggressively
            embeddings = self.model.encode(
                texts, 
                batch_size=self.batch_size,
                convert_to_numpy=True, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embeddings.tolist()
        else:
            # Native CPU Single-Core bottleneck aversion
            # Slice sequential arrays across all explicit multicore system boundaries via ThreadPoolExecutor
            def embed_sub_batch(sub_batch):
                return self.model.encode(
                    sub_batch, 
                    convert_to_numpy=True, 
                    normalize_embeddings=True, 
                    show_progress_bar=False
                )
                
            num_threads = self.num_cores
            batch_size = max(1, len(texts) // num_threads)
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            
            all_embeddings = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                for result in executor.map(embed_sub_batch, batches):
                    all_embeddings.extend(result.tolist())
                    
            return all_embeddings
