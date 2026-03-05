"""
Query Complexity Classifier

Uses a low-latency 8B model to score query complexity from 0.0 to 1.0.
Falls back to lightweight heuristics if API keys are missing.
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


class ComplexityClassifier:
    """
    Returns a strict float score in [0.0, 1.0] representing query complexity.
    """

    def __init__(self, model_override: str = None):
        phase = get_phase_model("complexity_scorer")
        self.provider = phase["provider"]
        self.model_id = model_override or phase["model"]
        self.temperature = phase.get("temperature", 0.0)
        self.max_tokens = phase.get("max_tokens", 120)

        from app.prompt_engine.groq_prompts.config import get_compiled_prompt
        self.system_prompt = get_compiled_prompt("complexity_scorer", self.model_id)

    async def score(self, query: str) -> float:
        """
        Returns a strict float score between 0.0 and 1.0.
        """
        if not query:
            return 0.0

        # Fallback heuristic if API key is missing
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
            return self._heuristic_score(query)

        try:
            data = await achat_completion(
                provider=self.provider,
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                timeout=8,
            )
            raw = data["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
            return self._clamp_score(parsed.get("score", 0.0))
        except Exception as e:
            logger.warning(f"[COMPLEXITY] Scoring failed, using heuristic fallback: {e}")
            return self._heuristic_score(query)

    def _heuristic_score(self, query: str) -> float:
        """Simple heuristic fallback based on length and multi-hop cues."""
        q = query.lower()
        word_count = len(q.split())
        multi_hop_flags = ["compare", "contrast", "difference", "analyze", "resolve", "tradeoff", "versus"]
        has_multi_hop = any(flag in q for flag in multi_hop_flags)

        score = 0.2
        if word_count > 25:
            score += 0.2
        if word_count > 50:
            score += 0.2
        if has_multi_hop:
            score += 0.3
        return self._clamp_score(score)

    @staticmethod
    def _clamp_score(value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            score = 0.0
        return max(0.0, min(1.0, score))
