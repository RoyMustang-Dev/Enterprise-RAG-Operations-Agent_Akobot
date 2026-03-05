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
        self.model_id = model_override or phase["model"]
        self.escalation_model_id = escalation_model or esc["model"]
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
        result = await self._execute_inference(self.model_id, user_prompt)
        
        # Enterprise Confidence Escalation Loop
        confidence = result.get("confidence", 0.0)
        multi_intent = result.get("multi_intent", False)
        
        if confidence < 0.60 or multi_intent:
            logger.warning(f"[SUPERVISOR] Escalation Triggered! Confidence: {confidence} | Multi-Intent: {multi_intent}. Rerouting to {self.escalation_model_id}...")
            # Automatically invoke the 70B core brain to verify the user intent
            result = await self._execute_inference(self.escalation_model_id, user_prompt)
            logger.info(f"[SUPERVISOR] Escalation Resolved Intent -> {result.get('intent')} (Conf: {result.get('confidence')})")
        else:
            logger.info(f"[SUPERVISOR] Detected Intent -> {result.get('intent')} (Conf: {confidence})")
            
        return result

    async def _execute_inference(self, target_model: str, user_prompt: str) -> Dict[str, Any]:
        """Executes the exact prompt context via fully non-blocking asynchronous I/O."""
        try:
            logger.info(f"[SUPERVISOR] Executing Async Intent Classification via {target_model}...")
            data = await achat_completion(
                provider=self.provider,
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
             # Fall-Open strictly to the RAG processing block
             return {"intent": "rag_question", "route": "rag_agent", "confidence": 0.1, "multi_intent": False}
