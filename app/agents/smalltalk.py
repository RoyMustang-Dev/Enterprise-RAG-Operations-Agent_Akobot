"""
Smalltalk Agent (Llama 8B)

This node catches general chatter, intent=greeting, and explicit questions 
about the AI's identity. It strictly queries the LLM combined with the 
Global Agent Bootstrapper Persona so it correctly identifies itself according
to user specs, instead of hardcoded dev strings.
"""
import logging
from typing import Dict, Any
import re

from app.prompt_engine.groq_prompts.config import get_compiled_prompt, PersonaCacheManager
from app.infra.model_registry import get_phase_model
from app.infra.llm_client import run_chat_completion

logger = logging.getLogger(__name__)

class SmalltalkAgent:
    
    def __init__(self):
        cfg = get_phase_model("smalltalk")
        self.provider = cfg["provider"]
        self.model_id = cfg["model"]
        self.temperature = cfg.get("temperature", 0.5)
        self.max_tokens = cfg.get("max_tokens", 512)
        
    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the LLM specifically for persona-driven smalltalk and greeting workflows.
        """
        query = state.get("query", "")
        chat_history = state.get("chat_history", [])
        
        # Pull Persona details to add to state
        cache = PersonaCacheManager()
        persona = cache.get_persona()
        bot_name = persona.get("bot_name", "Enterprise RAG System") if persona else "Enterprise RAG System"
        state["active_persona"] = bot_name
        
        # Compile System Prompt explicitly using the multimodal/conversational template 
        # (This injects the ReAct persona + strict bounds)
        system_prompt = get_compiled_prompt("smalltalk_agent", self.model_id)
        
        # Build messages payload including history for context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history if available, limit to last 4 to keep context clean
        for turn in chat_history[-4:]:
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
            
        messages.append({"role": "user", "content": query})
        
        state["optimizations"]["agent_routed"] = "smalltalk"
        
        try:
            logger.info(f"[SMALLTALK] Sending conversation payload to {self.provider}/{self.model_id}")
            data = run_chat_completion(
                provider=self.provider,
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=20,
            )
            answer = data.get("choices", [{}])[0].get("message", {}).get("content")
            # Guardrail: avoid "analysis of the user's message" for greetings.
            if query and isinstance(answer, str):
                lowered_q = re.sub(r"(.)\1{2,}", r"\1\1", query.lower())
                tokens = re.sub(r"[^a-z\s]", " ", lowered_q)
                tokens = re.sub(r"\s+", " ", tokens).strip()
                greeting_terms = {"hi", "hii", "hello", "hey", "heyy", "hiya", "yo"}
                is_greeting = any(t in greeting_terms for t in tokens.split()) or "help me" in tokens or "how can you help" in tokens
                if is_greeting:
                    answer = f"Hi! I'm {bot_name}. How can I help you today?"
            state["answer"] = answer
            state["confidence"] = 0.99
            
        except Exception as e:
            logger.error(f"[SMALLTALK] LLM Execution failed: {e}")
            state["answer"] = f"Hello! I am {bot_name}. I'm currently experiencing a network interruption."
            state["confidence"] = 0.5
            
        return state
