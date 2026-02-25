import os
import json
import logging
from typing import Dict, Any, List
from tenacity import retry, wait_exponential, stop_after_attempt
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class OnlineRewardModel:
    """
    RLAIF (Reinforcement Learning from AI Feedback) Scoring Judge
    
    This module implements an active, synchronous A/B testing inference loop without requiring a 
    physically retrained 70B variant. It evaluates multiple candidate responses (e.g. standard vs deep reasoning) 
    and mathematically selects the superior output based strictly on Enterprise grounding metrics.
    """
    def __init__(self, model_id: str = "llama-3.1-8b-instant"):
        self.model_id = model_id
        
        self.api_key = os.getenv("GROQ_API_KEY") 
        if not self.api_key:
            logger.warning("[SECURITY] GROQ_API_KEY not found. RLAIF Online Reward offline.")
            
        self.system_prompt = """SYSTEM: You are an elite, deterministic Evidence Scorer. 
Your explicit objective is to evaluate two candidate responses against the user's raw query and physical Vector context.

Generate EXACTLY the following JSON schema:
{
 "candidate_a_score": 0.0,
 "candidate_b_score": 0.0,
 "winner": "A" or "B",
 "rationale": "One brief sentence explaining why."
}

SCORING CRITERIA (0.0 to 10.0):
1. Grounding: Does the candidate exclusively use the provided Context? (Penalty if hallucinated).
2. Conciseness: Is the candidate free of verbose padding and conversational fluff?
3. Actionability: Does it directly address the user's explicit intent?
"""

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
        if not self.api_key:
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
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_payload}
            ],
            "temperature": 0.0, 
            "response_format": {"type": "json_object"}
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=10
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
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
