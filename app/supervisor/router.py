"""
Execution Graph Orchestrator (ReAct Router)

This module builds the Directed Acyclic Graph (DAG) for the multi-agent system.
It acts as the single entrypoint for the `/chat` route. It physically passes the State 
dictionary down the pipe, running the Guard -> Intent -> RAG or Smalltalk sequence.
"""
import logging
import os
import asyncio
from typing import Dict, Any
import re

from app.core.types import AgentState
from app.infra.database import get_chat_history, save_chat_turn
from app.prompt_engine.guard import PromptInjectionGuard
from app.prompt_engine.rewriter import PromptRewriter
from app.supervisor.intent import IntentClassifier
from app.supervisor.planner import AdaptivePlanner
from app.agents.rag import RAGAgent
from app.agents.coder import CoderAgent
from app.reasoning.complexity import ComplexityClassifier
from app.infra.provider_router import ProviderRouter

logger = logging.getLogger(__name__)

class ExecutionGraph:
    """
    Constructs the operational state machine ensuring that user input is systematically
    cleansed, categorized, and then mathematically routed to the correct Execution Brain.
    """
    
    def __init__(self):
        """
        Instantiates the immutable middleware layers that execute on every single request.
        """
        self.guard = PromptInjectionGuard()
        self.rewriter = PromptRewriter()
        self.classifier = IntentClassifier()
        self.complexity = ComplexityClassifier()
        self.planner = AdaptivePlanner()
        self.rag_agent = RAGAgent()
        self.coder_agent = CoderAgent()
        self.provider_router = ProviderRouter()

    @staticmethod
    def _strip_think(text: str) -> str:
        if not text:
            return text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip()
        
    async def invoke(self, query: str, chat_history: list = None, session_id: str = "", tenant_id: str = None, model_provider: str = "groq", extra_collections: list = None, reranker_profile: str = "auto", reranker_model_name: str = None) -> AgentState:
        """
        The formal entrypoint called by `app.api.routes`.
        
        Args:
            query (str): The raw string sent by the user.
            chat_history: Previously serialized human/AI turns.
            
        Returns:
            AgentState: The fully completed dictionary containing the final grounded answer.
        """
        # Initialize the global TypedDict scope
        # Pull recent chat history from SQLite and merge with provided history
        persisted_history = []
        if session_id:
            try:
                persisted_history = get_chat_history(session_id)
            except Exception:
                persisted_history = []

        safe_history = persisted_history + (chat_history or [])
        safe_history.append({"role": "user", "content": query})
        
        state: AgentState = {
            "session_id": session_id,
            "tenant_id": tenant_id,
            "query": query,
            "chat_history": safe_history,
            "streaming_callback": None,
            "intent": None,
            "search_query": None,
            "context_chunks": [],
            "context_text": "",
            "extra_collections": extra_collections or [],
            "confidence": 0.0,
            "verifier_verdict": "PENDING",
            "is_hallucinated": False,
            "verification_claims": [],
            "optimizations": {},
            "optimized_prompts": {},
            "answer": "",
            "sources": [],
            "reasoning_effort": "low",
            "latency_optimizations": {}
        }
        selected_provider = self.provider_router.select_provider(model_provider)
        state["optimizations"]["model_provider"] = selected_provider
        state["optimizations"]["reranker_profile"] = reranker_profile
        if reranker_model_name:
            state["optimizations"]["reranker_model_name"] = reranker_model_name
        
        # Persist the user turn immediately
        if session_id:
            try:
                save_chat_turn(session_id, "user", query)
            except Exception:
                pass

        # -------------------------------------------------------------
        # STEP 1: SAFETY (Prompt Injection Guard)
        # -------------------------------------------------------------
        # Guard uses blocking HTTP; offload to thread so API event loop stays responsive.
        safety_report = await asyncio.to_thread(self.guard.evaluate, query)
        if safety_report.get("is_malicious", False):
            guard_strict = os.getenv("GUARD_STRICT", "false").lower() == "true"
            if not guard_strict:
                safe_phrases = [
                    "summarize", "summary", "read the text", "extract text", "ocr",
                    "describe the image", "what is in the image", "list the text"
                ]
                risky_phrases = [
                    "ignore previous", "system prompt", "developer message", "jailbreak",
                    "exfiltrate", "bypass", "override", "policy"
                ]
                q = (query or "").lower()
                if any(s in q for s in safe_phrases) and not any(r in q for r in risky_phrases):
                    logger.warning("[ROUTER] Guard flagged safe-looking prompt; soft-allow enabled.")
                else:
                    logger.warning(f"[ROUTER] Guard intercepted payload. Flagged as: {safety_report.get('categories')}")
                    state["answer"] = "Security Exception: Your request violates Enterprise parameters."
                    state["confidence"] = 1.0
                    state["chat_history"].append({"role": "assistant", "content": state["answer"]})
                    if session_id:
                        try:
                            save_chat_turn(session_id, "assistant", state["answer"])
                        except Exception:
                            pass
                    return state
            else:
                logger.warning(f"[ROUTER] Guard intercepted payload. Flagged as: {safety_report.get('categories')}")
                state["answer"] = "Security Exception: Your request violates Enterprise parameters."
                state["confidence"] = 1.0
                state["chat_history"].append({"role": "assistant", "content": state["answer"]})
                if session_id:
                    try:
                        save_chat_turn(session_id, "assistant", state["answer"])
                    except Exception:
                        pass
                return state
            
        # -------------------------------------------------------------
        # STEP 2: REASONING (Intent + Complexity in Parallel)
        # -------------------------------------------------------------
        intent_task = self.classifier.classify(query)
        complexity_task = self.complexity.score(query)
        intent_report, complexity_score = await asyncio.gather(intent_task, complexity_task)

        state["intent"] = intent_report.get("intent", "rag_question")
        state["optimizations"]["complexity_score"] = complexity_score
        
        # -------------------------------------------------------------
        # STEP 3: ADAPTIVE PLANNING (Dynamic Routing)
        # -------------------------------------------------------------
        plan = self.planner.generate_plan(state)
        state["reasoning_effort"] = plan.get("reasoning_effort", state["reasoning_effort"])
        if plan.get("override_model"):
            state["optimizations"]["override_model"] = plan["override_model"]
        if state.get("extra_collections") and plan.get("route") == "out_of_scope":
            logger.info("[ROUTER] File context present; overriding out_of_scope to rag_agent.")
            plan["route"] = "rag_agent"

        # -------------------------------------------------------------
        # STEP 4: PROMPT OPTIMIZATION (MoE Rewriter)
        # -------------------------------------------------------------
        if plan.get("run_rewriter", True):
            # Generate 3 canonical prompts with explicit Temp/Token Metadata
            rewrite_data = await self.rewriter.rewrite(query, state["intent"])
            state["optimized_prompts"] = rewrite_data.get("prompts", {})
        
        # -------------------------------------------------------------
        # STEP 5: EXECUTION FORK (MoE / ReAct Routing)
        # -------------------------------------------------------------
        if plan.get("route") == "out_of_scope":
            logger.info("[ROUTER] Out_of_scope Intent Bypass executing.")
            state["optimizations"]["agent_routed"] = "out_of_scope_bypass"
            state["answer"] = "I am unable to answer this question based on the provided enterprise context."
            state["confidence"] = 1.0
            state["answer"] = self._strip_think(state.get("answer", ""))
            state["chat_history"].append({"role": "assistant", "content": state["answer"]})
            if session_id:
                try:
                    save_chat_turn(session_id, "assistant", state["answer"])
                except Exception:
                    pass
            return state
             
        elif plan.get("route") == "smalltalk":
             # Bypass the expensive 70B logic completely.
             logger.info("[ROUTER] Smalltalk Bypass executing via Llama-8B.")
             state["optimizations"]["agent_routed"] = "smalltalk_bypass"
             state["answer"] = "Hello! I am the Enterprise RAG System. How can I help you today?"
             state["confidence"] = 0.99
             state["answer"] = self._strip_think(state.get("answer", ""))
             state["chat_history"].append({"role": "assistant", "content": state["answer"]})
             if session_id:
                 try:
                     save_chat_turn(session_id, "assistant", state["answer"])
                 except Exception:
                     pass
             return state
             
        elif plan.get("route") == "coder_agent":
             logger.info(f"[ROUTER] Dispatching payload to the {state['intent']} Coder Agent.")
             state["optimizations"]["agent_routed"] = "coder_agent"
             # We execute solely against the Coder MoE ignoring dense 70B RAG chains
             final_state = await self.coder_agent.ainvoke(state)
             final_state["answer"] = self._strip_think(final_state.get("answer", ""))
             final_state["chat_history"].append({"role": "assistant", "content": final_state.get("answer", "")})
             if session_id:
                 try:
                     save_chat_turn(session_id, "assistant", final_state.get("answer", ""))
                 except Exception:
                     pass
             return final_state
             
        elif plan.get("route") == "rag_agent":
             # Dispatch into the heavy machinery
             logger.info("[ROUTER] Dispatching payload to the dense RAG agent.")
             state["optimizations"]["agent_routed"] = "rag_agent"
             
             # Execute the end-to-end RAG pipeline
             final_state = await self.rag_agent.ainvoke(state)
             final_state["answer"] = self._strip_think(final_state.get("answer", ""))
             final_state["chat_history"].append({"role": "assistant", "content": final_state.get("answer", "")})
             if session_id:
                 try:
                     save_chat_turn(session_id, "assistant", final_state.get("answer", ""))
                 except Exception:
                     pass
             return final_state
        
        # Failsafe Catch-all
        state["optimizations"]["agent_routed"] = "fallback"
        state["answer"] = "I am unsure how to route this specific query format."
        state["answer"] = self._strip_think(state.get("answer", ""))
        state["chat_history"].append({"role": "assistant", "content": state["answer"]})
        if session_id:
            try:
                save_chat_turn(session_id, "assistant", state["answer"])
            except Exception:
                pass
        return state
