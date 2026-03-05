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
from typing import Dict, Any
from app.infra.llm_client import achat_completion
from app.infra.model_registry import get_phase_model

logger = logging.getLogger(__name__)


class SourceScopeClassifier:
    def __init__(self, model_override: str = None):
        phase = get_phase_model("source_scope_classifier")
        self.provider = phase["provider"]
        self.model_id = model_override or phase["model"]
        self.temperature = phase.get("temperature", 0.0)
        self.max_tokens = phase.get("max_tokens", 60)
        from app.prompt_engine.groq_prompts.config import get_compiled_prompt
        self.system_prompt = get_compiled_prompt("source_scope_classifier", self.model_id)

    async def classify(self, user_prompt: str, has_session_files: bool) -> Dict[str, Any]:
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
            logger.warning("[SUPERVISOR] GROQ_API_KEY missing. Defaulting scope to 'both'.")
            return {"scope": "both", "confidence": 0.0}

        try:
            logger.info("[SUPERVISOR] Executing Source Scope Classification...")
            data = await achat_completion(
                provider=self.provider,
                model=self.model_id,
                messages=[
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
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                timeout=8,
            )
            raw_content = data["choices"][0]["message"]["content"]
            result = json.loads(raw_content)
            logger.info(f"[SUPERVISOR] Source Scope -> {result.get('scope')} (Conf: {result.get('confidence')})")
            return result
        except Exception as e:
            logger.error(f"[SUPERVISOR] Source scope classification failed: {e}")
            return {"scope": "both", "confidence": 0.0}
