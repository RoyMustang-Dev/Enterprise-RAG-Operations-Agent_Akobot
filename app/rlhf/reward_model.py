import os
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from typing import Dict, Any, List
from tenacity import retry, wait_exponential, stop_after_attempt
import asyncio
from app.infra.llm_client import achat_completion
from app.infra.model_registry import get_phase_model

logger = logging.getLogger(__name__)

class OnlineRewardModel:
    """
    RLAIF (Reinforcement Learning from AI Feedback) Scoring Judge
    
    This module implements an active, synchronous A/B testing inference loop without requiring a 
    physically retrained 70B variant. It evaluates multiple candidate responses (e.g. standard vs deep reasoning) 
    and mathematically selects the superior output based strictly on Enterprise grounding metrics.
    """
    def __init__(self, model_id: str = None):
        phase = get_phase_model("reward_scorer")
        self.provider = phase["provider"]
        self.model_id = model_id or phase["model"]
        self.temperature = phase.get("temperature", 0.0)
        self.max_tokens = phase.get("max_tokens", 200)
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
            logger.warning("[SECURITY] No API key found. RLAIF Online Reward offline.")
            
        from app.prompt_engine.groq_prompts.config import get_compiled_prompt
        self.system_prompt = get_compiled_prompt("reward_scorer", self.model_id)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=6), stop=stop_after_attempt(3))
    async def select_best_candidate(
        self, 
        query: str, 
        context: List[Dict[str, Any]], 
        candidate_a: str, 
        candidate_b: str
    ) -> str:
        """
        Asynchronously invokes the Reward Model to evaluate two distinct LLM syntagm chains.
        Returns the raw string of the winning candidate natively.
        """
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
            return candidate_a # Failsafe defaults to Standard execution

        logger.info(f"[RLAIF] Evaluating A/B Candidate Responses natively via {self.model_id}...")
        
        context_str = "\n".join([c.get("page_content", "") for c in context[:3]])
        
        user_payload = f"""
USER RAW QUERY: {query}

PHYSICAL CONTEXT: 
{context_str}

=== CANDIDATE A ===
{candidate_a}

=== CANDIDATE B ===
{candidate_b}
"""
        
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
                timeout=10,
            )
            json_str = data["choices"][0]["message"]["content"]
            try:
                result = json.loads(json_str)
                if result.get("winner") == "B":
                    logger.info(f"[RLAIF] Selected Candidate B (Score: {result.get('candidate_b_score')}) over A.")
                    return candidate_b
                else:
                    return candidate_a
            except json.JSONDecodeError:
                return candidate_a

        except Exception as e:
            logger.error(f"[RLAIF] Evaluation Fault: {e}")
            return candidate_a

    async def score_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        response: str
    ) -> float:
        """
        Returns a weighted reward score:
        Grounding (0.4), Actionability (0.3), Conciseness (0.2), Coherence (0.1)
        """
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
            return 0.0

        system_prompt = """SYSTEM: You are a deterministic response grader.
Return EXACTLY one JSON object:
{
  "grounding": 0.0-1.0,
  "actionability": 0.0-1.0,
  "conciseness": 0.0-1.0,
  "coherence": 0.0-1.0
}

Only use the provided context for grounding judgment."""

        context_str = "\n".join([c.get("page_content", "") for c in context[:5]])
        user_payload = f"""USER QUERY: {query}

CONTEXT:
{context_str}

RESPONSE:
{response}
"""

        try:
            data = await achat_completion(
                provider=self.provider,
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                timeout=10,
            )
            json_str = data["choices"][0]["message"]["content"]
            result = json.loads(json_str)

            grounding = float(result.get("grounding", 0.0))
            actionability = float(result.get("actionability", 0.0))
            conciseness = float(result.get("conciseness", 0.0))
            coherence = float(result.get("coherence", 0.0))

            weighted = (
                grounding * 0.4 +
                actionability * 0.3 +
                conciseness * 0.2 +
                coherence * 0.1
            )
            return max(0.0, min(1.0, weighted))
        except Exception as e:
            logger.error(f"[RLAIF] Scoring fault: {e}")
            return 0.0
