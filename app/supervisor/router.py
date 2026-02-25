"""
Execution Graph Orchestrator (ReAct Router)

This module builds the Directed Acyclic Graph (DAG) for the multi-agent system.
It acts as the single entrypoint for the `/chat` route. It physically passes the State 
dictionary down the pipe, running the Guard -> Intent -> RAG or Smalltalk sequence.
"""
import logging
import asyncio
from typing import Dict, Any

from app.core.types import AgentState
from app.prompt_engine.guard import PromptInjectionGuard
from app.prompt_engine.rewriter import PromptRewriter
from app.supervisor.intent import IntentClassifier
from app.agents.rag import RAGAgent
from app.agents.coder import CoderAgent

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
        self.rag_agent = RAGAgent()
        self.coder_agent = CoderAgent()
        
    async def invoke(self, query: str, chat_history: list = None, session_id: str = "", model_provider: str = "groq") -> AgentState:
        """
        The formal entrypoint called by `app.api.routes`.
        
        Args:
            query (str): The raw string sent by the user.
            chat_history: Previously serialized human/AI turns.
            
        Returns:
            AgentState: The fully completed dictionary containing the final grounded answer.
        """
        # Initialize the global TypedDict scope
        safe_history = chat_history or []
        safe_history.append({"role": "user", "content": query})
        
        state: AgentState = {
            "session_id": session_id,
            "query": query,
            "chat_history": safe_history,
            "streaming_callback": None,
            "intent": None,
            "search_query": None,
            "context_chunks": [],
            "context_text": "",
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
        state["optimizations"]["model_provider"] = model_provider
        
        # -------------------------------------------------------------
        # STEP 1: SAFETY (Prompt Injection Guard)
        # -------------------------------------------------------------
        # Guard uses blocking HTTP; offload to thread so API event loop stays responsive.
        safety_report = await asyncio.to_thread(self.guard.evaluate, query)
        if safety_report.get("is_malicious", False):
            logger.warning(f"[ROUTER] Guard intercepted payload. Flagged as: {safety_report.get('categories')}")
            state["answer"] = "Security Exception: Your request violates Enterprise parameters."
            state["confidence"] = 1.0
            state["chat_history"].append({"role": "assistant", "content": state["answer"]})
            return state
            
        # -------------------------------------------------------------
        # STEP 2: REASONING (Intent Classification)
        # -------------------------------------------------------------
        intent_report = await self.classifier.classify(query)
        state["intent"] = intent_report.get("intent", "rag_question")
        
        # -------------------------------------------------------------
        # STEP 3: PROMPT OPTIMIZATION (MoE Rewriter)
        # -------------------------------------------------------------
        # Generate 3 canonical prompts with explicit Temp/Token Metadata
        rewrite_data = await self.rewriter.rewrite(query, state["intent"])
        state["optimized_prompts"] = rewrite_data.get("prompts", {})
        
        # -------------------------------------------------------------
        # STEP 4: EXECUTION FORK (MoE / ReAct Routing)
        # -------------------------------------------------------------
        if state["intent"] == "out_of_scope":
             logger.info("[ROUTER] Out_of_scope Intent Bypass executing.")
             state["optimizations"]["agent_routed"] = "out_of_scope_bypass"
             state["answer"] = "I am unable to answer this question based on the provided enterprise context."
             state["confidence"] = 1.0
             state["chat_history"].append({"role": "assistant", "content": state["answer"]})
             return state
             
        elif state["intent"] in ["greeting", "smalltalk"]:
             # Bypass the expensive 70B logic completely.
             logger.info("[ROUTER] Smalltalk Bypass executing via Llama-8B.")
             state["optimizations"]["agent_routed"] = "smalltalk_bypass"
             state["answer"] = "Hello! I am the Enterprise RAG System. How can I help you today?"
             state["confidence"] = 0.99
             state["chat_history"].append({"role": "assistant", "content": state["answer"]})
             return state
             
        elif state["intent"] in ["code_request", "analytics_request"]:
             logger.info(f"[ROUTER] Dispatching payload to the {state['intent']} Coder Agent.")
             state["optimizations"]["agent_routed"] = "coder_agent"
             # We execute solely against the Coder MoE ignoring dense 70B RAG chains
             final_state = await self.coder_agent.ainvoke(state)
             final_state["chat_history"].append({"role": "assistant", "content": final_state.get("answer", "")})
             return final_state
             
        elif state["intent"] == "rag_question":
             # Dispatch into the heavy machinery
             logger.info("[ROUTER] Dispatching payload to the dense RAG agent.")
             state["optimizations"]["agent_routed"] = "rag_agent"
             
             # Execute the end-to-end RAG pipeline
             final_state = await self.rag_agent.ainvoke(state)
             final_state["chat_history"].append({"role": "assistant", "content": final_state.get("answer", "")})
             return final_state
        
        # Failsafe Catch-all
        state["optimizations"]["agent_routed"] = "fallback"
        state["answer"] = "I am unsure how to route this specific query format."
        state["chat_history"].append({"role": "assistant", "content": state["answer"]})
        return state
