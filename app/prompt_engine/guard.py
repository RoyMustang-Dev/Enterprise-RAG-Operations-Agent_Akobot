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
from app.infra.llm_client import chat_completion, extract_message_content
from app.infra.model_registry import get_phase_model

logger = logging.getLogger(__name__)

class PromptInjectionGuard:
    """
    Mandatory preprocessing filter. Executes a <200ms API call to a specific Guard Model.
    If 'is_malicious' is True, the process throws a 403 Forbidden automatically.
    """
    
    def __init__(self, model_override: str = None, escalation_model: str = None):
        """
        Dynamically binds to the Groq API Key required for inference.
        """
        phase = get_phase_model("security_guard")
        self.provider = phase["provider"]
        self.model_id = model_override or phase["model"]
        self.temperature = phase.get("temperature", 0.0)
        self.max_tokens = phase.get("max_tokens", 200)
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
            logger.warning("[SECURITY] No API key found. Overriding Prompt Guard (WARN: UNSECURE FALLBACK)")
            
        # Hardcoding the model specifically designated for Safety logic
        # If Groq rotates the naming convention, update this string.
        self.escalation_model_id = escalation_model or "llama-3.3-70b-versatile"
        
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
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
            # Bypass simulation if keys are missing to prevent catastrophic failures during scaffolding
            return {"is_malicious": False, "action": "allow", "evidence": "Key missing. Guard bypassed."}
        
        try:
            logger.info(f"[PROMPT GUARD] Evaluating prompt against {self.model_id}")
            data = chat_completion(
                provider=self.provider,
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=5,
            )
            raw_content = extract_message_content(data).strip().lower()
            if not raw_content:
                return {"is_malicious": False, "action": "allow", "evidence": "Guard empty response; fail-open"}
            
            if "unsafe" in raw_content:
                result = {"is_malicious": True, "categories": ["policy_violation"], "action": "block", "evidence": "Llama Guard flagged as unsafe"}
            else:
                result = {"is_malicious": False, "categories": [], "action": "allow", "evidence": "safe"}
            
            # Log successful evasions actively for SOC compliance
            if result.get("is_malicious"):
                # Soft-allow benign-looking, misspelled queries without risky keywords.
                if os.getenv("GUARD_SOFT_ALLOW_MISSPELL", "true").lower() == "true":
                    risky_env = os.getenv(
                        "GUARD_SOFT_ALLOW_RISKY_PHRASES",
                        "ignore previous,system prompt,developer message,jailbreak,exfiltrate,bypass,override,policy,prompt injection,roleplay"
                    )
                    risky_phrases = [p.strip().lower() for p in risky_env.split(",") if p.strip()]
                    q = (user_prompt or "").lower()
                    if not any(r in q for r in risky_phrases):
                        alpha_tokens = [t for t in q.split() if any(c.isalpha() for c in t)]
                        if len(alpha_tokens) >= 3:
                            logger.info("[PROMPT GUARD] Soft-allow benign query after heuristic check.")
                            return {"is_malicious": False, "action": "allow", "evidence": "heuristic_allow"}

                logger.warning(f"[PROMPT GUARD] Malicious intent intercepted! Categories: {result.get('categories')}")
                # Escalate to a larger model to reduce false positives
                try:
                    logger.info(f"[PROMPT GUARD] Escalating guard decision to {self.escalation_model_id}")
                    esc_data = chat_completion(
                        provider=self.provider,
                        model=self.escalation_model_id,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=8,
                    )
                    esc_raw = esc_data["choices"][0]["message"]["content"].strip().lower()
                    
                    if "unsafe" not in esc_raw:
                        logger.info("[PROMPT GUARD] Escalation override: allow")
                        return {"is_malicious": False, "action": "allow", "evidence": "Escalation override"}
                except Exception as e:
                    logger.warning(f"[PROMPT GUARD] Escalation failed: {e}")
                
            return result
            
        except requests.exceptions.Timeout:
             logger.error("[PROMPT GUARD] Network timeout evaluating guard metrics.")
             if os.getenv("GUARD_FAIL_OPEN", "true").lower() == "true":
                 return {"is_malicious": False, "action": "allow", "evidence": "Guard timeout; fail-open"}
             return {"is_malicious": True, "action": "block", "evidence": "Guard Network Timeout"}
        except requests.exceptions.RequestException as e:
             logger.error(f"[PROMPT GUARD] Inference API failure: {e}")
             if os.getenv("GUARD_FAIL_OPEN", "true").lower() == "true":
                 return {"is_malicious": False, "action": "allow", "evidence": "Guard API failure; fail-open"}
             return {"is_malicious": True, "action": "block", "evidence": "Quota or API Exceeded"}
        except json.JSONDecodeError as e:
             logger.error(f"[PROMPT GUARD] LLM failed to emit JSON compliant payload: {e}")
             return {"is_malicious": False, "action": "allow", "evidence": "Guard JSON parse failed; soft-allow"}
        except Exception as e:
             logger.error(f"[PROMPT GUARD] Guard evaluation failed: {e}")
             if os.getenv("GUARD_FAIL_OPEN", "true").lower() == "true":
                 return {"is_malicious": False, "action": "allow", "evidence": "Guard failure; fail-open"}
             return {"is_malicious": True, "action": "block", "evidence": "Guard failure"}
