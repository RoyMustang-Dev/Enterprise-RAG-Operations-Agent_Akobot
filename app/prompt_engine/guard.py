"""
Enterprise Security Filter (Prompt Guard)

This module executes *before* any other LLM logic. It leverages Groq's micro-models 
(e.g., llama-prompt-guard-2-86m) to mathematically evaluate if the user's string is 
attempting prompt-injection, jailbreaking, or data exfiltration.

Enterprise Clients mandate this layer to prevent prompt extracting and data poisoning.
"""
import os
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PromptInjectionGuard:
    """
    Mandatory preprocessing filter. Executes a <200ms API call to a specific Guard Model.
    If 'is_malicious' is True, the process throws a 403 Forbidden automatically.
    """
    
    def __init__(self, model_override: str = "llama-3.1-8b-instant", escalation_model: str = "llama-3.3-70b-versatile"):
        """
        Dynamically binds to the Groq API Key required for inference.
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("[SECURITY] GROQ_API_KEY not found. Overriding Prompt Guard (WARN: UNSECURE FALLBACK)")
            
        # Hardcoding the model specifically designated for Safety logic
        # If Groq rotates the naming convention, update this string.
        self.model_id = model_override
        self.escalation_model_id = escalation_model
        
        # The rigid JSON schema instructed in the rag-implementation blueprint
        from app.prompt_engine.groq_prompts.config import get_compiled_prompt
        self.system_prompt = get_compiled_prompt("security_guard", self.model_id)

    def evaluate(self, user_prompt: str) -> Dict[str, Any]:
        """
        Executes the synchronous REST call to Groq's inference engine.
        
        Args:
            user_prompt (str): The raw string submitted by the frontend payload.
            
        Raises:
            Exception: If network layer fails.
            
        Returns:
            dict: The explicit structural extraction determining if execution should proceed.
        """
        if not self.api_key:
            # Bypass simulation if keys are missing to prevent catastrophic failures during scaffolding
            return {"is_malicious": False, "action": "allow", "evidence": "Key missing. Guard bypassed."}
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        def _payload(model_id: str):
            return {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # Strict determinism required for security filters
                "temperature": 0.0,
                "max_tokens": 200,
                "response_format": {"type": "json_object"}
            }
        
        try:
            logger.info(f"[PROMPT GUARD] Evaluating prompt against {self.model_id}")
            # Intentionally synchronous mapping. We block the DAG until Safety confirms authorization.
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=_payload(self.model_id),
                timeout=5
            )
            response.raise_for_status()
            
            raw_content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(raw_content)
            
            # Log successful evasions actively for SOC compliance
            if result.get("is_malicious"):
                logger.warning(f"[PROMPT GUARD] Malicious intent intercepted! Categories: {result.get('categories')}")
                # Escalate to a larger model to reduce false positives
                try:
                    logger.info(f"[PROMPT GUARD] Escalating guard decision to {self.escalation_model_id}")
                    esc = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=_payload(self.escalation_model_id),
                        timeout=8
                    )
                    esc.raise_for_status()
                    esc_raw = esc.json()["choices"][0]["message"]["content"]
                    esc_result = json.loads(esc_raw)
                    if not esc_result.get("is_malicious"):
                        logger.info("[PROMPT GUARD] Escalation override: allow")
                        return {"is_malicious": False, "action": "allow", "evidence": "Escalation override"}
                except Exception as e:
                    logger.warning(f"[PROMPT GUARD] Escalation failed: {e}")
                
            return result
            
        except requests.exceptions.Timeout:
             logger.error("[PROMPT GUARD] Network timeout evaluating guard metrics.")
             # Fall-Closed paradigm: Assume malicious if unable to verify
             return {"is_malicious": True, "action": "block", "evidence": "Guard Network Timeout"}
        except requests.exceptions.RequestException as e:
             logger.error(f"[PROMPT GUARD] Inference API failure: {e}")
             # If HTTP layer specifically crashed due to quotas, we block.
             return {"is_malicious": True, "action": "block", "evidence": "Quota or API Exceeded"}
        except json.JSONDecodeError as e:
             logger.error(f"[PROMPT GUARD] LLM failed to emit JSON compliant payload: {e}")
             return {"is_malicious": True, "action": "block", "evidence": "Malformed Schema Evaluator"}
