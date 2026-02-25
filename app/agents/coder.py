import os
import json
import logging
from typing import Dict, Any, List
from tenacity import retry, wait_exponential, stop_after_attempt
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class CoderAgent:
    """
    Dedicated Mixture-of-Experts Node: The Code Execution & Analytics Synthesizer
    
    This agent strictly utilizes the `qwen/qwen3-32b` engine to parse mathematical, programmatic, 
    and systemic analytical user intents. By isolating coding requests from the standard RAG pipeline, 
    we prevent the 70B language model from generating hallucinated code syntax and significantly reduce costs.
    """
    def __init__(self, model_id: str = "qwen/qwen3-32b"):
        self.model_id = model_id
        
        # We actively detect the key per the overarching Enterprise RAG standards.
        self.api_key = os.getenv("GROQ_API_KEY") 
        if not self.api_key:
            logger.warning("[SECURITY] GROQ_API_KEY not found. Coder Agent Offline.")
            
        self.system_prompt = """SYSTEM: You are an elite, deterministic Enterprise Software Engineer. 
Your objective is to generate syntactically perfect, highly structured Python, JavaScript, or Shell code.
Do not provide excessive conversational padding. 

OUTPUT FORMAT:
Provide a concise explanation of the logic, followed immediately by the runnable code block. 
All code blocks MUST specify the language syntax (e.g., ```python). 

If the user provides context chunks or structural data, you MUST utilize that context exclusively to ground your programmatic variables.
"""

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously parses the AgentState, extracts the user's programmatic intent, 
        and invokes the Qwen Coder model to synthesize structural output.
        """
        if not self.api_key:
            state["answer"] = "Coder Agent Offline: Missing API Key."
            state["confidence"] = 0.0
            return state

        query = state.get("query", "")
        context_chunks = state.get("context", [])
        
        # We append retrieved chunks as context if the user's coding query involves corporate data
        user_payload = f"USER QUERY: {query}\n\n"
        if context_chunks:
            user_payload += "PROVIDED SYSTEM CONTEXT:\n"
            for chunk in context_chunks:
                user_payload += f"- {chunk.get('page_content', '')}\n"

        logger.info(f"[MoE - CODER AGENT] Invoking {self.model_id} for code synthesis...")
        
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
            "temperature": 0.1,  # highly deterministic for code
            "max_completion_tokens": 4096
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    answer_text = data["choices"][0]["message"]["content"]
                    
                    # Update State
                    state["answer"] = answer_text
                    # Coder models are generally highly confident in deterministic syntax
                    state["confidence"] = 0.95 
                    state["routed_agent"] = "qwen2.5-coder-32b"
                    return state

        except Exception as e:
            logger.error(f"[MoE - CODER AGENT] Code Synthesis failed: {e}")
            raise Exception("Coder Synthesis Pipeline Fault")
