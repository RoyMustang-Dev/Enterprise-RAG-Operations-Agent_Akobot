"""
Source Scope Classifier

Determines whether a query should be answered using:
- knowledge base only (kb_only)
- session uploads only (session_only)
- both (both)
"""
import os
import json
import logging
import aiohttp
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SourceScopeClassifier:
    def __init__(self, model_override: str = "llama-3.1-8b-instant"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_id = model_override
        from app.prompt_engine.groq_prompts.config import get_compiled_prompt
        self.system_prompt = get_compiled_prompt("source_scope_classifier", self.model_id)

    async def classify(self, user_prompt: str, has_session_files: bool) -> Dict[str, Any]:
        if not self.api_key:
            logger.warning("[SUPERVISOR] GROQ_API_KEY missing. Defaulting scope to 'both'.")
            return {"scope": "both", "confidence": 0.0}

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_query": user_prompt,
                            "has_session_files": bool(has_session_files),
                        }
                    ),
                },
            ],
            "temperature": 0.0,
            "max_completion_tokens": 60,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            logger.info("[SUPERVISOR] Executing Source Scope Classification...")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=8,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    raw_content = data["choices"][0]["message"]["content"]
                    result = json.loads(raw_content)
                    logger.info(f"[SUPERVISOR] Source Scope -> {result.get('scope')} (Conf: {result.get('confidence')})")
                    return result
        except Exception as e:
            logger.error(f"[SUPERVISOR] Source scope classification failed: {e}")
            return {"scope": "both", "confidence": 0.0}
