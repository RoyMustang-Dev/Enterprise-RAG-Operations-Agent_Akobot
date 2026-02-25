import os
import json
import logging
from typing import Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class PromptRewriter:
    """
    MoE Controller: The 'Magic Prompt' Engine
    
    This module intercepts the raw user dictionary right after intent classification. 
    It actively utilizes `llama-3.3-70b-versatile` to distill the user's intent into 3 canonical 
    downstream prompts (concise_low, standard_med, deep_high) with precisely recommended runtime metadata.
    This enables the underlying pipelines to select the explicit contextual execution depth intelligently.
    """
    def __init__(self, model_id: str = "openai/gpt-oss-120b"):
        self.model_id = model_id
        
        # Enterprise-Grade Environment mapping
        self.api_key = os.getenv("GROQ_API_KEY") 
        if not self.api_key:
            logger.warning("[SECURITY] GROQ_API_KEY not found. Prompt Rewriter Offline. (WARN: Bypassing Logic)")
            
        self.system_prompt = """SYSTEM: You are an elite Prompt Engineer and Optimization Controller. 
Your explicit objective is to analyze the user's raw input and synthesize 3 robust, instruction-tuned downstream prompts.

Generate EXACTLY the following JSON schema:
{
 "original_user_prompt": "...",
 "prompts": {
   "concise_low": {
      "prompt":"...", 
      "recommended_model":"llama-3.1-8b-instant",
      "temperature":0.0,
      "expected_tokens":150,
      "purpose":"Fast, explicit, factual summary."
   },
   "standard_med": {
      "prompt":"...", 
      "recommended_model":"llama-3.3-70b-versatile",
      "temperature":0.1,
      "expected_tokens":500,
      "purpose":"Standard explanatory breakdown."
   },
   "deep_high": {
      "prompt":"...", 
      "recommended_model":"llama-3.3-70b-versatile",
      "temperature":0.2,
      "expected_tokens":1500,
      "purpose":"Exhaustive multi-step analytical reasoning."
   }
 }
}

CRITICAL RULES:
1. Do NOT hallucinate facts into the prompts. Simply clarify the user's existing linguistic intent.
2. If the user commands code, the prompts must explicitly command strict programmatic output formatting.
3. If the user asks a knowledge question, all 3 prompts MUST contain the explicit instruction: "Answer utilizing ONLY the provided provided CONTEXT_CHUNKS. Provide inline citations."
"""

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    async def rewrite(self, user_prompt: str, intent_classification: str = "rag_question") -> Dict[str, Any]:
        """
        Asynchronously invokes the MoE controller to synthesize optimized prompt variants.
        """
        if not self.api_key:
            # Dev-Fallback / Bypass
            return {
                "original_user_prompt": user_prompt,
                "prompts": {
                    "standard_med": {"prompt": user_prompt, "recommended_model": "llama-3.3-70b-versatile", "temperature": 0.1}
                }
            }

        logger.info(f"[MoE - REWRITER] Distilling raw query via {self.model_id}...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        user_payload = f"RAW QUERY: {user_prompt}\nDETECTED SYSTEM INTENT: {intent_classification}"
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_payload}
            ],
            "temperature": 0.0, # Deterministic JSON structure
            "max_completion_tokens": 1024,
            "response_format": {"type": "json_object"}
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=15
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    json_str = data["choices"][0]["message"]["content"]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        import re
                        match = re.search(r"```(?:json)?\n(.*?)\n```", json_str, re.DOTALL)
                        if match:
                            try:
                                return json.loads(match.group(1))
                            except Exception:
                                pass
                        logger.error("[MoE - REWRITER] Schema Failure: Non-JSON payload returned.")
                        return {"original_user_prompt": user_prompt, "prompts": {"standard_med": {"prompt": user_prompt}}}

        except Exception as e:
            logger.error(f"[MoE - REWRITER] Sub-Routine Fault: {e}")
            # Failsafe: Bypass rewriter natively rather than crashing the API Gateway pipeline
            return {"original_user_prompt": user_prompt, "prompts": {"standard_med": {"prompt": user_prompt}}}
