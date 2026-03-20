import os
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from typing import Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt
import asyncio
import re
from app.infra.llm_client import achat_completion, extract_message_content
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

    @staticmethod
    def _normalize_greeting(text: str) -> str:
        if not text:
            return text
        lowered = text.lower().strip()
        # Collapse repeated characters: "hiiiiii" -> "hii"
        lowered = re.sub(r"(.)\1{2,}", r"\1\1", lowered)
        # Basic typo normalization
        lowered = lowered.replace("helo", "hello").replace("helloo", "hello").replace("hh", "h")
        lowered = lowered.replace("whow", "how").replace("cna", "can").replace("elph", "help").replace("mee", "me")
        tokens = re.sub(r"[^a-z\s]", " ", lowered)
        tokens = re.sub(r"\s+", " ", tokens).strip()
        if not tokens:
            return "Hi! How can you help me?"
        if "help me" in tokens or "can you help" in tokens or "how can you help" in tokens:
            return "Hi! How can you help me?"
        if tokens in {"hi", "hii", "hello", "hey", "hey there"} or tokens.startswith(("hi ", "hello ", "hey ")):
            return "Hi! How can you help me?"
        return "Hi! How can you help me?"

    @staticmethod
    def _looks_like_greeting(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        lowered = re.sub(r"(.)\1{2,}", r"\1\1", lowered)
        lowered = lowered.replace("whow", "how").replace("cna", "can").replace("elph", "help").replace("mee", "me")
        tokens = re.sub(r"[^a-z\s]", " ", lowered)
        tokens = re.sub(r"\s+", " ", tokens).strip()
        if not tokens:
            return False
        greeting_terms = {"hi", "hii", "hello", "hey", "heyy", "hiya", "yo"}
        token_list = tokens.split()
        if any(t in greeting_terms for t in token_list):
            return True
        if "how can you help me" in tokens or "what can you do" in tokens or "help me" in tokens:
            return True
        return False

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    async def rewrite(self, user_prompt: str, intent_classification: str = "rag_question") -> Dict[str, Any]:
        """
        Asynchronously invokes the MoE controller to synthesize optimized prompt variants.
        """
        if self._looks_like_greeting(user_prompt):
            normalized = self._normalize_greeting(user_prompt)
            return {
                "original_user_prompt": user_prompt,
                "prompts": {
                    "concise_low": {"prompt": normalized, "recommended_model": "llama-3.1-8b-instant", "temperature": 0.0},
                    "standard_med": {"prompt": normalized, "recommended_model": "llama-3.1-8b-instant", "temperature": 0.0},
                    "deep_high": {"prompt": normalized, "recommended_model": "llama-3.1-8b-instant", "temperature": 0.0},
                }
            }
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
            json_str = extract_message_content(data)
            if not json_str:
                raise ValueError("empty_response")
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
                # Fallback attempt: stricter JSON-only prompt and Groq if available
                try:
                    fallback_provider = "groq" if os.getenv("GROQ_API_KEY") else self.provider
                    fallback_model = os.getenv("REWRITER_GROQ_MODEL", "llama-3.1-8b-instant")
                    data_fb = await achat_completion(
                        provider=fallback_provider,
                        model=fallback_model if fallback_provider == "groq" else self.model_id,
                        messages=[
                            {"role": "system", "content": (self.system_prompt or "") + "\nReturn ONLY valid JSON."},
                            {"role": "user", "content": user_payload},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"},
                        timeout=15,
                    )
                    json_fb = extract_message_content(data_fb)
                    if json_fb:
                        return json.loads(json_fb)
                except Exception:
                    pass
                logger.error("[MoE - REWRITER] Schema Failure: Non-JSON payload returned.")
                return {"original_user_prompt": user_prompt, "prompts": {"standard_med": {"prompt": user_prompt}}}

        except Exception as e:
            logger.error(f"[MoE - REWRITER] Sub-Routine Fault: {e}")
            # Failsafe: Bypass rewriter natively rather than crashing the API Gateway pipeline
            return {"original_user_prompt": user_prompt, "prompts": {"standard_med": {"prompt": user_prompt}}}
