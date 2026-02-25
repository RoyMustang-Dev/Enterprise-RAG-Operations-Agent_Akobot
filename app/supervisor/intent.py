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
import json
import logging
import aiohttp
from typing import Dict, Any

logger = logging.getLogger(__name__)

class IntentClassifier:
    """
    Executes a <400ms inference call establishing the conversational trajectory.
    """
    
    def __init__(self, model_override: str = "llama-3.1-8b-instant", escalation_model: str = "llama-3.3-70b-versatile"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_id = model_override
        self.escalation_model_id = escalation_model
        
        # The rigid JSON schema instructed in the rag-implementation blueprint
        self.system_prompt = '''SYSTEM: You are a high-precision intent classifier. Given the user message, classify into one of: ["greeting","smalltalk","out_of_scope","rag_question","code_request","analytics_request","multimodal_audio","other"]. Output exactly one compact JSON:

{"intent": "<one of the labels above>", "confidence": 0.00-1.00, "route": "<target_agent_name>", "multi_intent": <true/false>, "notes": "<optional 1-sentence signal words>"}

Rules:
- If user asks for anything illegal, set intent="out_of_scope".
- If user references files, docs, data, or asks about specific people/entities (e.g. HR reviews, employee data) -> prefer "rag_question".
- If user asks to write/execute code -> "code_request".
- If user asks two highly conflicting/separate questions simultaneously, set "multi_intent": true.
- If not confident (<0.5), choose "other" and include notes.
'''

    async def classify(self, user_prompt: str) -> Dict[str, Any]:
        """
        Calculates the explicit logical intent of the user.
        Dynamically escalates to the 70B model if confidence drops < 0.60.
        
        Args:
            user_prompt (str): The raw string extracted from the HTTP payload.
            
        Returns:
            dict: The dictionary containing the exact routing path to follow.
        """
        if not self.api_key:
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # Near determinism. We don't want the router 'hallucinating' paths.
            "temperature": 0.1,
            "max_completion_tokens": 120,
            "response_format": {"type": "json_object"}
        }
        
        try:
            logger.info(f"[SUPERVISOR] Executing Async Intent Classification via {target_model}...")
            async with aiohttp.ClientSession() as session:
                async with session.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=8) as response:
                    response.raise_for_status()
                    
                    data = await response.json()
                    raw_content = data["choices"][0]["message"]["content"]
                    return json.loads(raw_content)
                    
        except Exception as e:
             logger.error(f"[SUPERVISOR] Critical LLM async execution failure on {target_model}: {e}")
             # Fall-Open strictly to the RAG processing block
             return {"intent": "rag_question", "route": "rag_agent", "confidence": 0.1, "multi_intent": False}
