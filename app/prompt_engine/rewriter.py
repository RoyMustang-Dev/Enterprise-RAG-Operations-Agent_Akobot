import os
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from typing import Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt
import asyncio
from app.infra.llm_client import achat_completion
from app.infra.model_registry import get_phase_model

logger = logging.getLogger(__name__)

class PromptRewriter:
    """
    MoE Controller: The 'Magic Prompt' Engine
    
    This module intercepts the raw user dictionary right after intent classification. 
    It actively utilizes `llama-3.3-70b-versatile` to distill the user's intent into 3 canonical 
    downstream prompts (concise_low, standard_med, deep_high) with precisely recommended runtime metadata.
    This enables the underlying pipelines to select the explicit contextual execution depth intelligently.
    """
    def __init__(self, model_id: str = None):
        phase = get_phase_model("query_rewriter")
        self.provider = phase["provider"]
        self.model_id = model_id or phase["model"]
        self.temperature = phase.get("temperature", 0.0)
        self.max_tokens = phase.get("max_tokens", 1024)
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
            logger.warning("[SECURITY] No API key found. Prompt Rewriter Offline. (WARN: Bypassing Logic)")
            
        from app.prompt_engine.groq_prompts.config import get_compiled_prompt
        self.system_prompt = get_compiled_prompt("query_rewriter", self.model_id)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    async def rewrite(self, user_prompt: str, intent_classification: str = "rag_question") -> Dict[str, Any]:
        """
        Asynchronously invokes the MoE controller to synthesize optimized prompt variants.
        """
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
            # Dev-Fallback / Bypass
            return {
                "original_user_prompt": user_prompt,
                "prompts": {
                    "standard_med": {"prompt": user_prompt, "recommended_model": "llama-3.3-70b-versatile", "temperature": 0.1}
                }
            }

        logger.info(f"[MoE - REWRITER] Distilling raw query via {self.model_id}...")
        
        user_payload = f"RAW QUERY: {user_prompt}\nDETECTED SYSTEM INTENT: {intent_classification}"
        max_input_chars = int(os.getenv("REWRITER_MAX_INPUT_CHARS", "8000"))
        if len(self.system_prompt) + len(user_payload) > max_input_chars:
            logger.warning("[MoE - REWRITER] Payload too large for safe Groq call. Bypassing rewriter.")
            return {
                "original_user_prompt": user_prompt,
                "prompts": {
                    "standard_med": {"prompt": user_prompt, "recommended_model": "llama-3.3-70b-versatile", "temperature": 0.1}
                }
            }
        
        try:
            data = await achat_completion(
                provider=self.provider,
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                timeout=15,
            )
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
