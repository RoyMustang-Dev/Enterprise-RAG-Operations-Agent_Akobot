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
import time
from app.infra.hardware import HardwareProbe

logger = logging.getLogger(__name__)

# Reduce HuggingFace console noise
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

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
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_model = "gemini-embedding-001"
        # Keep Qdrant dimension stable (1024). Gemini supports outputDimensionality 128..3072.
        self.gemini_output_dim = 1024
        
        # Phase 13: Hardware Probing
        if not self.openai_api_key and not self.gemini_api_key:
            profile = HardwareProbe.get_profile()
            self.device = profile.get("primary_device", "cpu")
            self.num_cores = profile.get("cpu_cores", os.cpu_count() or 4)
            self.batch_size = profile.get("embedding_batch_size", 32)
            logger.info(f"[EMBEDDING ENGINE] Hardware Probe: device={self.device} cores={self.num_cores} batch={self.batch_size}")
            if self.device == "cpu":
                logger.warning("[EMBEDDING ENGINE] Running on CPU. Embedding throughput will be slower.")
        else:
            self.batch_size = 32
            if self.gemini_api_key:
                logger.info("[EMBEDDING ENGINE] Cloud Override -> Routing to Gemini embeddings.")
            else:
                logger.info("[EMBEDDING ENGINE] Cloud Override -> Routing to OpenAI API text-embedding-3-small.")
        
    @property
    def model(self):
        """
        Lazy-loads the massive 1.5GB transformer model entirely into RAM exclusively on the first function call,
        drastically reducing the startup time for fast API routing sequences.
        """
        if self.gemini_api_key:
            return None
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
            
        if self.gemini_api_key:
            return self._gemini_embed_text(text)
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
            
        if self.gemini_api_key:
            return self._gemini_embed_batch(texts)
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

        # Ensure model is initialized on the main thread before parallel work
        _ = self.model

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

    def _normalize(self, vec: List[float]) -> List[float]:
        # Avoid numpy dependency; use simple L2 normalization.
        import math
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _mask_key(self, text: str) -> str:
        if not text:
            return text
        key = self.gemini_api_key or ""
        if key and key in text:
            return text.replace(key, "[REDACTED]")
        return text

    def _raise_clean_http_error(self, err: Exception, context: str) -> None:
        msg = self._mask_key(str(err))
        raise RuntimeError(f"{context}: {msg}") from None

    def _gemini_embed_text(self, text: str) -> List[float]:
        if not text:
            return []
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:embedContent"
        payload = {
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.gemini_output_dim,
        }
        params = {"key": self.gemini_api_key}
        backoff = 1
        for attempt in range(5):
            try:
                response = requests.post(url, params=params, json=payload, timeout=30)
                if response.status_code == 429:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue
                response.raise_for_status()
                values = response.json().get("embedding", {}).get("values", [])
                return self._normalize(values)
            except Exception as e:
                if attempt == 4:
                    self._raise_clean_http_error(e, "Gemini embedContent failed")
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
        return []

    def _gemini_embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # Gemini batch endpoint requires a per-request model field.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:batchEmbedContents"
        params = {"key": self.gemini_api_key}
        batch_size = 32
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            backoff = 1
            for attempt in range(6):
                payload = {
                    "requests": [
                        {
                            "model": f"models/{self.gemini_model}",
                            "content": {"parts": [{"text": t}]},
                            "outputDimensionality": self.gemini_output_dim,
                        }
                        for t in batch
                    ],
                }
                try:
                    response = requests.post(url, params=params, json=payload, timeout=60)
                    if response.status_code == 429:
                        # throttle: backoff + shrink batch to reduce TPM spikes
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 16)
                        if batch_size > 8:
                            batch_size = max(8, batch_size // 2)
                        continue
                    response.raise_for_status()
                    embeddings = response.json().get("embeddings", [])
                    for emb in embeddings:
                        values = emb.get("values", [])
                        all_embeddings.append(self._normalize(values))
                    break
                except Exception as e:
                    if attempt == 5:
                        self._raise_clean_http_error(e, "Gemini batchEmbedContents failed")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
        return all_embeddings
