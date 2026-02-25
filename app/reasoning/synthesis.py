"""
Core Synthesis Engine (Llama 70B & Groq Native Annotations)

This is the primary mathematical reasoning brain of the Enterprise RAG System.
It explicitly accepts the Top-5 strictly reranked Vector DB chunks and forces the
model to synthesize an answer bounded entirely by those chunks natively utilizing 
Groq's `documents` API array mapping.

It embeds exponential backoff protections and strictly budgets input tokens.
"""
import os
import json
import logging
import requests
import tiktoken
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SynthesisEngine:
    """
    Executes the dense grounding logic utilizing Groq's largest available reasoning model.
    """
    
    def __init__(self, model_override: str = "llama-3.3-70b-versatile"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_id = model_override
        
        # Tokenizer initialization for boundary limits
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.MAX_INPUT_TOKENS = 6500 # Strict ceiling below the 8K bound
        
        # The rigid System Prompt
        self.system_prompt = '''SYSTEM: You are the enterprise-grade reasoning brain. Your responsibility is to synthesize final answers using ONLY the securely uploaded context documents. Follow these rules EXACTLY.

1) INPUTS available: 
   - USER_PROMPT: "<user text>"

2) OUTPUT format (Output exactly one JSON object):
   - "answer": <concise human-readable result, <= 500 words>
   - "confidence": 0.00-1.00

3) RULES:
    - DO NOT make up answers.
    - When using information from a document, you MUST append `[Doc ID]` exactly where the fact is mentioned (e.g. "Aditya is a data scientist [Doc 1].").
    - If you cannot answer, respond: "I don't know based on the provided documents."
    - Do NOT include chain-of-thought.
'''

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _execute_api_call(self, headers: dict, payload: dict) -> requests.Response:
        """Executes the POST HTTP layer securely guarded by an exponential backoff decorator."""
        logger.info("[SYNTHESIS] Transmitting native generation inference request...")
        # 30 second timeout as 8K token context evaluation requires heavy vRAM latency on Groq's backend
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"[SYNTHESIS API ERROR] {response.text}")
            
        response.raise_for_status()
        return response

    def synthesize(self, user_prompt: str, context_chunks: List[Dict[str, Any]], 
                   override_model: str = None, override_temp: float = None) -> Dict[str, Any]:
        """
        Synthesizes the final conversational output while structurally balancing token economics.
        """
        if not self.api_key:
             return {"answer": "Error: Groq API Key missing.", "provenance": [], "confidence": 0.0}
             
        # Target Model Assignments via Routing Override
        active_model = override_model if override_model else self.model_id
        active_temp = override_temp if override_temp is not None else 0.3
             
        # Token Budgeting
        system_overhead = len(self.tokenizer.encode(self.system_prompt))
        user_prompt_overhead = len(self.tokenizer.encode(user_prompt))
        budget_remaining = self.MAX_INPUT_TOKENS - (system_overhead + user_prompt_overhead)
        
        context_string = ""
        cost_sum = 0
        provenance = []
        
        # Concat context into string natively since Groq 70B no longer supports the schema
        for i, chunk in enumerate(context_chunks):
            chunk_text = chunk.get("page_content", "")
            cost = len(self.tokenizer.encode(chunk_text))
            
            if cost_sum + cost > budget_remaining:
                logger.warning(f"[SYNTHESIS] Truncating remaining chunks to survive Token Budget. Cost hit {cost_sum}/{budget_remaining}")
                break
                
            native_id = str(chunk.get("source", f"Chunk-{i}"))
            context_string += f"[Doc {native_id}]:\n{chunk_text}\n---\n"
            cost_sum += cost
            provenance.append(
                {
                    "source": native_id,
                    "score": chunk.get("score", 0.0),
                    "text": chunk_text[:240],
                }
            )
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": active_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"CONTEXT DOCUMENTS:\n{context_string}\n\nUSER_PROMPT: {user_prompt}"}
            ],
            "temperature": active_temp, # Bounded creativity
            "max_completion_tokens": 4096,
            "response_format": {"type": "json_object"}
        }
        
        try:
            logger.info(f"[SYNTHESIS] Inference payload configured with raw context boundary.")
            response = self._execute_api_call(headers, payload)
            
            response_json = response.json()
            raw_content = response_json["choices"][0]["message"]["content"]
            parsed_result = json.loads(raw_content)
            
            return {
                "answer": parsed_result.get("answer", ""),
                "provenance": provenance,
                "confidence": parsed_result.get("confidence", 0.0)
            }
            
        except Exception as e:
             logger.error(f"[SYNTHESIS] Execution completely crashed: {e}")
             return {"answer": "Internal Generation Error validating tokens.", "provenance": provenance, "confidence": 0.0}
