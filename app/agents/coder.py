import os
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import re
import logging
from typing import Dict, Any, List
from tenacity import retry, wait_exponential, stop_after_attempt
import asyncio
from app.infra.llm_client import achat_completion
from app.infra.model_registry import get_phase_model

logger = logging.getLogger(__name__)

class CoderAgent:
    """
    Dedicated Mixture-of-Experts Node: The Code Execution & Analytics Synthesizer
    
    This agent strictly utilizes the `qwen/qwen3-32b` engine to parse mathematical, programmatic, 
    and systemic analytical user intents. By isolating coding requests from the standard RAG pipeline, 
    we prevent the 70B language model from generating hallucinated code syntax and significantly reduce costs.
    """
    def __init__(self, model_id: str = None):
        phase = get_phase_model("coder_agent")
        self.provider = phase["provider"]
        self.model_id = model_id or phase["model"]
        self.temperature = phase.get("temperature", 0.1)
        self.max_tokens = phase.get("max_tokens", 4096)
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
            logger.warning("[SECURITY] No API key found. Coder Agent Offline.")
            
        from app.prompt_engine.groq_prompts.config import get_compiled_prompt
        self.system_prompt = get_compiled_prompt("coder_agent", self.model_id)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously parses the AgentState, extracts the user's programmatic intent, 
        and invokes the Qwen Coder model to synthesize structural output.
        """
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
            state["answer"] = "Coder Agent Offline: Missing API Key."
            state["confidence"] = 0.0
            return state

        query = state.get("query", "")
        context_chunks = state.get("context_chunks", [])
        
        # We append retrieved chunks as context if the user's coding query involves corporate data
        user_payload = f"USER QUERY: {query}\n\n"
        if context_chunks:
            user_payload += "PROVIDED SYSTEM CONTEXT:\n"
            coder_context_max_chars = int(os.getenv("CODER_CONTEXT_MAX_CHARS", "8000"))
            current_chars = 0
            
            for chunk in context_chunks:
                chunk_text = f"- {chunk.get('page_content', '')}\n"
                if current_chars + len(chunk_text) > coder_context_max_chars:
                    logger.info(f"[MoE - CODER AGENT] Context truncated at {current_chars} chars to prevent payload overflow.")
                    break
                user_payload += chunk_text
                current_chars += len(chunk_text)

        logger.info(f"[MoE - CODER AGENT] Invoking {self.model_id} for code synthesis...")
        max_total_chars = int(os.getenv("CODER_MAX_INPUT_CHARS", "12000"))
        if len(self.system_prompt) + len(user_payload) > max_total_chars:
            # Aggressively trim context to stay within safe bounds.
            overflow = (len(self.system_prompt) + len(user_payload)) - max_total_chars
            if overflow > 0 and "PROVIDED SYSTEM CONTEXT" in user_payload:
                user_payload = user_payload[:-overflow]
                user_payload = user_payload.rsplit("\n", 1)[0] + "\n[Context truncated for safety]\n"
        
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
                timeout=30,
            )
            answer_text = data["choices"][0]["message"]["content"]
            # Strip any accidental chain-of-thought blocks if present
            answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL)
            answer_text = re.sub(r"<think>.*", "", answer_text, flags=re.DOTALL).strip()
            
            # Update State
            state["answer"] = answer_text
            # Coder models are generally highly confident in deterministic syntax
            state["confidence"] = 0.95 
            state["routed_agent"] = self.model_id
            return state

        except Exception as e:
            logger.error(f"[MoE - CODER AGENT] Code Synthesis failed: {e}")
            raise Exception("Coder Synthesis Pipeline Fault")
