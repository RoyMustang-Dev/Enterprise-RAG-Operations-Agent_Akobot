"""
ReAct Supervisor Intent Classifier

This module acts as the "Traffic Cop" for the entire architecture.
Before we waste computing power executing a dense 70B RAG pipeline, we pass 
the user's string to a hyper-fast 8B model (`llama-3.1-8b-instant`).

By forcing the 8B model to return strict JSON, we can programmatically fork 
the execution graph (e.g., Short-circuit "Hello" directly to a fast response, 
saving massive compute costs).
"""
import os
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from typing import Dict, Any
import re
from app.infra.llm_client import achat_completion
from app.infra.model_registry import get_phase_model

logger = logging.getLogger(__name__)

class IntentClassifier:
    """
    Executes a <400ms inference call establishing the conversational trajectory.
    """
    
    def __init__(self, model_override: str = None, escalation_model: str = None):
        phase = get_phase_model("intent_classifier")
        esc = get_phase_model("intent_classifier_escalation")
        self.provider = phase["provider"]
        self.fallback_provider = phase.get("fallback_provider")
        self.model_id = model_override or phase["model"]
        self.escalation_model_id = escalation_model or esc["model"]
        self.escalation_provider = esc.get("provider")
        self.escalation_fallback_provider = esc.get("fallback_provider")
        self.temperature = phase.get("temperature", 0.1)
        self.max_tokens = phase.get("max_tokens", 120)
        
        from app.prompt_engine.groq_prompts.config import get_compiled_prompt
        self.system_prompt = get_compiled_prompt("intent_classifier", self.model_id)

    async def classify(self, user_prompt: str) -> Dict[str, Any]:
        """
        Calculates the explicit logical intent of the user.
        Dynamically escalates to the 70B model if confidence drops < 0.60.
        
        Args:
            user_prompt (str): The raw string extracted from the HTTP payload.
            
        Returns:
            dict: The dictionary containing the exact routing path to follow.
        """
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
             # Scaffolding fallback preventing 500 crashes during Phase 2/3 migration
             logger.warning("[SUPERVISOR] GROQ_API_KEY missing. Defaulting to 'rag_question'.")
             return {
                 "intent": "rag_question", 
                 "confidence": 0.0, 
                 "route": "rag_agent", 
                 "multi_intent": False,
                 "notes": "API Key missing. Forced fallback."
             }
             
        # Execute Initial Fast Routing using 8B Model
        result = await self._execute_inference(self.provider, self.model_id, user_prompt)
        
        # Enterprise Confidence Escalation Loop
        confidence = result.get("confidence", 0.0)
        intents = result.get("intents") or ([result.get("intent")] if result.get("intent") else [])
        multi_intent = result.get("multi_intent", False) or (len(intents) > 1)
        
        if confidence < 0.60 or multi_intent:
            logger.warning(f"[SUPERVISOR] Escalation Triggered! Confidence: {confidence} | Multi-Intent: {multi_intent}. Rerouting to {self.escalation_model_id}...")
            # Automatically invoke the 70B core brain to verify the user intent
            result = await self._execute_inference(self.escalation_provider, self.escalation_model_id, user_prompt)
            logger.info(f"[SUPERVISOR] Escalation Resolved Intent -> {result.get('intent')} (Conf: {result.get('confidence')})")
        else:
            logger.info(f"[SUPERVISOR] Detected Intent -> {result.get('intent')} (Conf: {confidence})")
        # Normalize payload
        intent = result.get("intent") or (intents[0] if intents else "rag_question")
        intents = result.get("intents") or ([intent] if intent else [])
        # Guardrail: if an email address is present, ensure email intent is included
        if re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}", user_prompt, re.IGNORECASE):
            if "email_request" not in intents:
                intents.append("email_request")
        return {
            "intent": intent,
            "intents": intents,
            "multi_intent": result.get("multi_intent", False) or (len(intents) > 1),
            "confidence": result.get("confidence", confidence),
        }

    async def _execute_inference(self, provider: str, target_model: str, user_prompt: str) -> Dict[str, Any]:
        """Executes the exact prompt context via fully non-blocking asynchronous I/O."""
        try:
            logger.info(f"[SUPERVISOR] Executing Async Intent Classification via {target_model}...")
            data = await achat_completion(
                provider=provider,
                model=target_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                timeout=8,
            )
            raw_content = data["choices"][0]["message"]["content"]
            return json.loads(raw_content)
                    
        except Exception as e:
            logger.error(f"[SUPERVISOR] Critical LLM async execution failure on {target_model}: {e}")
            # Try fallback provider/model before failing open
            try:
                fallback_provider = self.fallback_provider or self.escalation_fallback_provider or "gemini"
                fallback_model = os.getenv("INTENT_FALLBACK_MODEL", "gemini-2.5-flash")
                if fallback_provider != provider and os.getenv("GEMINI_API_KEY"):
                    data = await achat_completion(
                        provider=fallback_provider,
                        model=fallback_model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"},
                        timeout=8,
                    )
                    raw_content = data["choices"][0]["message"]["content"]
                    return json.loads(raw_content)
            except Exception as fallback_error:
                logger.error(f"[SUPERVISOR] Fallback intent classification failed: {fallback_error}")
            # Fall-Open strictly to the RAG processing block
            return {"intent": "rag_question", "intents": ["rag_question"], "confidence": 0.1, "multi_intent": False}
