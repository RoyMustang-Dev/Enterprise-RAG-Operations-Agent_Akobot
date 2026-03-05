"""
Cross-Encoder Semantic Reranker -> Meta-Ranker (LLM-as-a-Judge)

Refactored to utilize a hyper-fast 8B LLM (e.g. `llama-3.1-8b-instant`) operating strictly via Groq LPU API.
This explicitly eliminates the PyTorch local CPU bottleneck (>5.0s latency) and leverages dynamic
semantic reasoning rather than rigid mathematical overlap, allowing us to drop irrelevant contexts seamlessly in <400ms.
"""
import logging
import os
import json
import asyncio
from app.infra.llm_client import achat_completion
from app.infra.model_registry import get_phase_model
from typing import List, Dict, Any
from app.prompt_engine.groq_prompts.config import get_compiled_prompt

logger = logging.getLogger(__name__)

class SemanticReranker:
    """
    Applies an LLM-as-a-Judge to evaluate semantic relevance over candidate pairs.
    """
    
    _instances = {}

    def __init__(self, model_name: str = None):
        """
        Instantiates the Meta-Ranker with an explicit model.
        """
        phase = get_phase_model("meta_ranker")
        self.provider = phase["provider"]
        self.model_name = model_name or phase["model"]
        self.temperature = phase.get("temperature", 0.0)
        self.max_tokens = phase.get("max_tokens", 1024)
        
        # We explicitly load the optimized prompt logic containing strict JSON mappings
        self.system_prompt = get_compiled_prompt("meta_ranker", self.model_name)
        
        logger.info(f"[META-RANKER] Scaffolded {self.model_name} for high-speed dynamic ranking.")

    @classmethod
    def get_instance(cls, model_name: str = None):
        phase = get_phase_model("meta_ranker")
        name = model_name or os.getenv("RERANKER_MODEL_NAME", "llama-3.1-8b-instant")
        # If ModelsLab is active, prefer the phase model to avoid invalid Groq-only ids.
        if phase.get("provider") == "modelslab":
            name = phase.get("model", name)
        if name not in cls._instances:
            cls._instances[name] = cls(model_name=name)
        return cls._instances[name]
        
    def rerank(self, query: str, context_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for the asynchronous `arerank` method, preserving interface compatibility.
        NOTE: In a fully async pipeline (like `rag.py`), you should ideally call this asynchronously. 
        Since this is currently called synchronously in `node_rerank_documents`, we execute via asyncio.run().
        """
        try:
            return asyncio.run(self.arerank(query, context_chunks, top_k))
        except Exception as e:
            logger.error(f"[META-RANKER] Sync Wrapper Exception: {e}")
            return context_chunks[:top_k]
            
    async def arerank(self, query: str, context_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Evaluates the linguistic overlap between the exact query and every individual candidate chunk using a fast LLM.
        """
        if not context_chunks:
            return []

        if os.getenv("RERANKER_ENABLED", "true").lower() != "true" or (not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY")):
            logger.info("[META-RANKER] Disabled via env or API key missing. Returning raw dense retrieval.")
            return context_chunks[:top_k]
            
        # Compress chunks into a numbered list payload for the LLM Judge
        payload_string = f"USER QUERY: {query}\n\nCANDIDATE CHUNKS:\n"
        for i, chunk in enumerate(context_chunks):
            # Truncate content slightly to ensure 30 chunks fits in the 8K context safely
            content = chunk.get("page_content", "")[:350].replace('\n', ' ')
            payload_string += f"[{i}] {content}\n"
            
        try:
            logger.info(f"[META-RANKER] Transmitting {len(context_chunks)} chunks to {self.model_name}...")
            data = await achat_completion(
                provider=self.provider,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": payload_string},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                timeout=10,
            )
            raw_content = data["choices"][0]["message"]["content"]
            result = json.loads(raw_content)
            ranked_list = result.get("ranked_chunks", [])
                    
            # We map the JSON array of {"chunk_id": X, "rerank_score": Y} back to our native memory arrays
            scored_chunks = []
            for rank_data in ranked_list:
                c_id = rank_data.get("chunk_id")
                if c_id is not None and 0 <= c_id < len(context_chunks):
                    chunk = context_chunks[int(c_id)]
                    chunk["rerank_score"] = float(rank_data.get("rerank_score", 0.0))
                    scored_chunks.append(chunk)

            # Only filter if the LLM successfully ranked at least some chunks.
            if scored_chunks:
                # Filter dynamically just like we did for the Cross-Encoder (> 0.10)
                sorted_chunks = sorted(scored_chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                sorted_chunks = [ch for ch in sorted_chunks if ch.get("rerank_score", 0.0) > 0.10]
                final_chunks = sorted_chunks[:top_k]
                
                logger.info(f"[META-RANKER] Dropped {len(context_chunks) - len(final_chunks)} irrelevant chunks dynamically.")
                return final_chunks
            else:
                raise ValueError("LLM returned an empty rankings array representation.")
                        
        except Exception as e:
             logger.error(f"[META-RANKER] LLM Judge execution crashed: {e}")
             logger.warning("[META-RANKER] Failsafe: Reverting to raw Vector Cosine Distance ordering.")
             return context_chunks[:top_k]
