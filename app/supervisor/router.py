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
from app.supervisor.source_scope import SourceScopeClassifier
from app.infra.history_budget import trim_history_by_token_budget
from app.infra.logging_utils import stage_info
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
        self.source_scope = SourceScopeClassifier()
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

    @staticmethod
    def _sanitize_rewrite(original: str, rewritten: str) -> str:
        if not rewritten:
            return original
        lowered = rewritten.strip().lower()
        if not lowered:
            return original
        # Avoid meta-instructions from the rewriter; use original in that case.
        meta_markers = [
            "interpret the user",
            "interpret the user's input",
            "rewrite the user",
            "rephrase the user",
            "optimize the prompt",
            "improve the prompt",
            "your task is to",
            "you are asked to",
        ]
        if any(m in lowered for m in meta_markers):
            return original
        return rewritten

    @staticmethod
    def _looks_like_greeting(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        lowered = re.sub(r"(.)\1{2,}", r"\1\1", lowered)
        lowered = lowered.replace("whow", "how").replace("cna", "can").replace("elph", "help").replace("mee", "me")
        tokens = re.sub(r"[^a-z\s]", " ", lowered)
        tokens = re.sub(r"\s+", " ", tokens).strip()
        if not tokens:
            return False
        token_list = tokens.split()
        greeting_terms = {"hi", "hii", "hello", "hey", "heyy", "hiya", "yo"}
        if any(t in greeting_terms for t in token_list):
            return True
        if "how can you help me" in tokens or "what can you do" in tokens or "help me" in tokens:
            return True
        return False

    @staticmethod
    def _normalize_greeting(text: str) -> str:
        if not text:
            return "Hi! How can you help me?"
        lowered = text.lower().strip()
        lowered = re.sub(r"(.)\1{2,}", r"\1\1", lowered)
        lowered = lowered.replace("whow", "how").replace("cna", "can").replace("elph", "help").replace("mee", "me")
        tokens = re.sub(r"[^a-z\s]", " ", lowered)
        tokens = re.sub(r"\s+", " ", tokens).strip()
        if "help me" in tokens or "can you help" in tokens or "how can you help" in tokens:
            return "Hi! How can you help me?"
        if tokens in {"hi", "hii", "hello", "hey", "hey there"} or tokens.startswith(("hi ", "hello ", "hey ")):
            return "Hi! How can you help me?"
        return "Hi! How can you help me?"
        
    async def invoke(self, query: str, chat_history: list = None, session_id: str = "", tenant_id: str = None, model_provider: str = "auto", extra_collections: list = None, reranker_profile: str = "auto", reranker_model_name: str = None, force_session_context: bool = False, streaming_callback=None) -> AgentState:
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
                persisted_history = get_chat_history(session_id, tenant_id=tenant_id)
            except Exception:
                persisted_history = []

        safe_history = persisted_history + (chat_history or [])
        safe_history.append({"role": "user", "content": query})
        # Token-aware trimming (keeps as much history as fits in the budget)
        safe_history = trim_history_by_token_budget(safe_history)
        
        state: AgentState = {
            "session_id": session_id,
            "tenant_id": tenant_id,
            "query": query,
            "chat_history": safe_history,
            "streaming_callback": streaming_callback,
            "intent": None,
            "search_query": None,
            "context_chunks": [],
            "context_text": "",
            "extra_collections": extra_collections or [],
            "force_session_context": bool(force_session_context),
            "confidence": 0.0,
            "verifier_verdict": "PENDING",
            "is_hallucinated": False,
            "verification_claims": [],
            "optimizations": {},
            "optimized_prompts": {},
            "answer": "",
            "reasoning_effort": "low",
            "latency_optimizations": {},
            "active_persona": None
        }
        selected_provider = self.provider_router.select_provider(model_provider)
        state["optimizations"]["model_provider"] = selected_provider
        state["optimizations"]["reranker_profile"] = reranker_profile
        if reranker_model_name:
            state["optimizations"]["reranker_model_name"] = reranker_model_name
        config: Dict[str, Any] = {}
        
        # Persist the user turn immediately
        if session_id:
            try:
                save_chat_turn(session_id, "user", query, tenant_id=tenant_id)
            except Exception:
                pass

        # -------------------------------------------------------------
        # STEP 1: PROMPT OPTIMIZATION (MoE Rewriter) - Run FIRST
        # -------------------------------------------------------------
        original_query = query
        rewrite_data = await self.rewriter.rewrite(query, "unknown")
        state["optimized_prompts"] = rewrite_data.get("prompts", {})
        rewritten = (
            state["optimized_prompts"].get("standard_med", {}) or {}
        ).get("prompt") or query
        rewritten = self._sanitize_rewrite(query, rewritten)
        state["optimizations"]["original_query"] = query
        state["optimizations"]["rewritten_query"] = rewritten
        query = rewritten
        state["query"] = query
        stage_info(logger, "ROUTER:REWRITE", "rewriter_completed=true")

        # -------------------------------------------------------------
        # STEP 2: SAFETY (Prompt Injection Guard)
        # -------------------------------------------------------------
        # Guard uses blocking HTTP; offload to thread so API event loop stays responsive.
        safety_report = await asyncio.to_thread(self.guard.evaluate, query)
        if safety_report.get("is_malicious", False):
            original_query = state["optimizations"].get("original_query") or query
            if original_query != query:
                original_report = await asyncio.to_thread(self.guard.evaluate, original_query)
                if not original_report.get("is_malicious", False):
                    stage_info(logger, "ROUTER:GUARD", "action=soft_allow_original")
                    query = original_query
                    state["query"] = query
                    safety_report = original_report
                else:
                    safety_report = original_report
            guard_strict = os.getenv("GUARD_STRICT", "false").lower() == "true"
            if not guard_strict:
                soft_allow = os.getenv("GUARD_SOFT_ALLOW", "true").lower() == "true"
                safe_phrases_env = os.getenv(
                    "GUARD_SOFT_ALLOW_PHRASES",
                    "summarize,summary,read the text,extract text,ocr,describe the image,what is in the image,list the text"
                )
                risky_phrases_env = os.getenv(
                    "GUARD_SOFT_ALLOW_RISKY_PHRASES",
                    "ignore previous,system prompt,developer message,jailbreak,exfiltrate,bypass,override,policy"
                )
                safe_phrases = [p.strip().lower() for p in safe_phrases_env.split(",") if p.strip()]
                risky_phrases = [p.strip().lower() for p in risky_phrases_env.split(",") if p.strip()]
                q = (query or "").lower()
                if soft_allow and any(s in q for s in safe_phrases) and not any(r in q for r in risky_phrases):
                    stage_info(logger, "ROUTER:GUARD", "action=soft_allow")
                    logger.warning("[ROUTER] Guard flagged safe-looking prompt; soft-allow enabled.")
                else:
                    stage_info(logger, "ROUTER:GUARD", "action=block")
                    logger.warning(f"[ROUTER] Guard intercepted payload. Flagged as: {safety_report.get('categories')}")
                    state["answer"] = "Security Exception: Your request violates Enterprise parameters."
                    state["confidence"] = 1.0
                    state["chat_history"].append({"role": "assistant", "content": state["answer"]})
                    if session_id:
                        try:
                            save_chat_turn(session_id, "assistant", state["answer"], tenant_id=tenant_id)
                        except Exception:
                            pass
                    return state
            else:
                stage_info(logger, "ROUTER:GUARD", "action=block")
                logger.warning(f"[ROUTER] Guard intercepted payload. Flagged as: {safety_report.get('categories')}")
                state["answer"] = "Security Exception: Your request violates Enterprise parameters."
                state["confidence"] = 1.0
                state["chat_history"].append({"role": "assistant", "content": state["answer"]})
                if session_id:
                    try:
                        save_chat_turn(session_id, "assistant", state["answer"], tenant_id=tenant_id)
                    except Exception:
                        pass
                return state
            
        # -------------------------------------------------------------
        # STEP 3: REASONING (Intent + Complexity in Parallel)
        # -------------------------------------------------------------
        intent_task = self.classifier.classify(query)
        complexity_task = self.complexity.score(query)
        intent_report, complexity_score = await asyncio.gather(intent_task, complexity_task)

        state["intent"] = intent_report.get("intent", "rag_question")
        state["optimizations"]["complexity_score"] = complexity_score
        stage_info(logger, "ROUTER:INTENT", f"intent={state['intent']} conf={intent_report.get('confidence')}")

        # Override intent for greeting-like messages to avoid misclassification.
        if self._looks_like_greeting(original_query):
            state["intent"] = "smalltalk"
            normalized = self._normalize_greeting(original_query)
            state["query"] = normalized
            state["optimizations"]["rewritten_query"] = normalized
            stage_info(logger, "ROUTER:INTENT", "intent_override=smalltalk_greeting")
        
        # -------------------------------------------------------------
        # STEP 4: ADAPTIVE PLANNING (Dynamic Routing)
        # -------------------------------------------------------------
        plan = self.planner.generate_plan(state)
        state["reasoning_effort"] = plan.get("reasoning_effort", state["reasoning_effort"])
        if plan.get("override_model"):
            state["optimizations"]["override_model"] = plan["override_model"]
        stage_info(logger, "ROUTER:PLAN", f"route={plan.get('route')} effort={state['reasoning_effort']}")
        if plan.get("route") == "out_of_scope":
            # If we have any file/session context, always favor RAG.
            if state.get("extra_collections") or state.get("tenant_id"):
                logger.info("[ROUTER] Tenant/session context present; overriding out_of_scope to rag_agent.")
                plan["route"] = "rag_agent"

        # Decide retrieval scope using LLM classifier (no hardcoded keyword rules)
        try:
            has_session_files = bool(state.get("extra_collections"))
            scope_report = await self.source_scope.classify(query, has_session_files=has_session_files)
            state["retrieval_scope"] = scope_report.get("scope", "both")
            state["optimizations"]["retrieval_scope"] = state["retrieval_scope"]
            stage_info(logger, "ROUTER:SCOPE", f"scope={state['retrieval_scope']} conf={scope_report.get('confidence')}")
        except Exception as e:
            logger.warning(f"[ROUTER] Source scope classifier failed; defaulting to 'both': {e}")
            state["retrieval_scope"] = "both"

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
                    save_chat_turn(session_id, "assistant", state["answer"], tenant_id=tenant_id)
                except Exception:
                    pass
            return state
             
        elif plan.get("route") == "smalltalk":
             logger.info("[ROUTER] Smalltalk Routed. Invoking dynamically...")
             from app.agents.smalltalk import SmalltalkAgent
             smalltalk_agent = SmalltalkAgent()
             final_state = await smalltalk_agent.ainvoke(state)
             final_state["chat_history"].append({"role": "assistant", "content": final_state.get("answer", "")})
             if session_id:
                 try:
                     save_chat_turn(session_id, "assistant", final_state.get("answer", ""), tenant_id=tenant_id)
                 except Exception:
                     pass
             return final_state
             
        elif plan.get("route") == "coder_agent":
             logger.info(f"[ROUTER] Dispatching payload to the {state['intent']} Coder Agent.")
             state["optimizations"]["agent_routed"] = "coder_agent"
             # We execute solely against the Coder MoE ignoring dense 70B RAG chains
             try:
                 final_state = await self.coder_agent.ainvoke(state, config=config)
             except Exception as e:
                 logger.error(f"[ROUTER] Coder agent failed: {e}")
                 final_state = dict(state)
                 final_state["answer"] = "The code assistant is temporarily unavailable. Please try again shortly."
                 final_state["confidence"] = 0.2
             if not final_state.get("verifier_verdict"):
                 final_state["verifier_verdict"] = "N/A"
             if "sources" not in final_state:
                 final_state["sources"] = []
             final_state["answer"] = self._strip_think(final_state.get("answer", ""))
             final_state["chat_history"].append({"role": "assistant", "content": final_state.get("answer", "")})
             if session_id:
                 try:
                     save_chat_turn(session_id, "assistant", final_state.get("answer", ""), tenant_id=tenant_id)
                 except Exception:
                     pass
             return final_state
             
        elif plan.get("route") == "rag_agent":
             # Dispatch into the heavy machinery
             logger.info("[ROUTER] Dispatching payload to the dense RAG agent.")
             state["optimizations"]["agent_routed"] = "rag_agent"
             
             # Execute the end-to-end RAG pipeline
             try:
                 final_state = await self.rag_agent.ainvoke(state, config=config)
             except Exception as e:
                 logger.error(f"[ROUTER] RAG agent failed: {e}")
                 final_state = dict(state)
                 final_state["answer"] = "The retrieval system is temporarily unavailable. Please try again shortly."
                 final_state["confidence"] = 0.2
                 final_state.setdefault("sources", [])
             final_state["answer"] = self._strip_think(final_state.get("answer", ""))
             final_state["chat_history"].append({"role": "assistant", "content": final_state.get("answer", "")})
             if session_id:
                 try:
                     save_chat_turn(session_id, "assistant", final_state.get("answer", ""), tenant_id=tenant_id)
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
                save_chat_turn(session_id, "assistant", state["answer"], tenant_id=tenant_id)
            except Exception:
                pass
        return state
