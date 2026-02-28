"""
RAG Agent Orchestrator (LangGraph)

This module constructs the directed acyclic graph (DAG) specifically for Retrieval-Augmented Generation.
It physically hooks together Phase 4 (Retrieval) and Phase 5 (Reasoning) into a strict, fault-tolerant pipeline.
"""
import logging
import os
import time
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from app.core.types import AgentState

# Phase 4 Imports
from app.retrieval.metadata_extractor import MetadataExtractor
from app.retrieval.vector_store import QdrantStore
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.reranker import SemanticReranker
from app.retrieval.hybrid_search import HybridReranker

# Phase 5 Imports
from app.reasoning.synthesis import SynthesisEngine
from app.reasoning.verifier import HallucinationVerifier
from app.reasoning.formatter import ResponseFormatter

# Phase 13 Imports
from app.rlhf.reward_model import OnlineRewardModel
import asyncio

logger = logging.getLogger(__name__)

class RAGAgent:
    """
    Constructs the `rag_question` execution path using LangGraph.
    Data flows strictly: Metadata -> Retrive -> Rerank -> Synthesize -> Verify -> Format.
    """
    
    def __init__(self):
        """
        Instantiates all immutable heavy models required for the pipeline.
        These are loaded once to prevent RAM/VRAM exhaustion on every request.
        """
        logger.info("[RAG AGENT] Initializing Core Architecture Models...")
        self.metadata_extractor = MetadataExtractor()
        self.embedding_model = EmbeddingModel()
        self.vector_store = QdrantStore()
        self.reranker = SemanticReranker()
        self.hybrid_reranker = HybridReranker()
        self.synthesis_engine = SynthesisEngine()
        self.verifier = HallucinationVerifier()
        self.reward_model = OnlineRewardModel()
        
        # Build the physical state computer
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """
        Defines the sequential nodes and edges of the LLM computational graph.
        """
        workflow = StateGraph(AgentState)
        
        # Node Definitions
        workflow.add_node("extract_metadata", self.node_extract_metadata)
        workflow.add_node("retrieve_documents", self.node_retrieve_documents)
        workflow.add_node("rerank_documents", self.node_rerank_documents)
        workflow.add_node("synthesize_answer", self.node_synthesize_answer)
        workflow.add_node("verify_answer", self.node_verify_answer)
        workflow.add_node("format_output", self.node_format_output)
        
        # Edge Definitions (Strict Sequential Pipeline)
        workflow.set_entry_point("extract_metadata")
        workflow.add_edge("extract_metadata", "retrieve_documents")
        
        # Conditional Edge: If retrieval finds nothing, short-circuit to formatting
        workflow.add_conditional_edges(
            "retrieve_documents",
            lambda state: "rerank_documents" if len(state.get("context_chunks", [])) > 0 else "format_output"
        )
        
        workflow.add_edge("rerank_documents", "synthesize_answer")
        workflow.add_edge("synthesize_answer", "verify_answer")
        workflow.add_edge("verify_answer", "format_output")
        workflow.add_edge("format_output", END)
        
        return workflow.compile()
        
    def invoke(self, state: AgentState) -> AgentState:
        """
        Executes the compiled LangGraph pipeline.
        """
        logger.info(f"[RAG AGENT] Commencing Execution DAG for query: {state['query'][:50]}...")
        # Deep copy to ensure immutable updates structurally
        result = self.graph.invoke(state)
        return result
        
    async def ainvoke(self, state: AgentState) -> AgentState:
        """
        Executes the compiled LangGraph pipeline non-blockingly inside the asynchronous threadpool bounds.
        """
        logger.info(f"[RAG AGENT] Commencing Execution DAG (Async) for query: {state['query'][:50]}...")
        result = await self.graph.ainvoke(state)
        return result

    # -------------------------------------------------------------------------
    # Graph Node Definitions
    # -------------------------------------------------------------------------

    async def node_extract_metadata(self, state: AgentState) -> AgentState:
        """Dynamically extracts strict Qdrant bounds from natural language."""
        query = state["query"]
        extraction = await self.metadata_extractor.extract_filters(query)
        
        # We store the applied filters in optimizations for audit tracking
        if "optimizations" not in state:
            state["optimizations"] = {}
        state["optimizations"]["metadata_filters"] = extraction.get("filters", {})
        
        # Normally you'd assign a rewritten query here, but we pass the raw intent down
        state["search_query"] = query 
        return state

    def node_retrieve_documents(self, state: AgentState) -> AgentState:
        """Executes the high-speed L2 distance fetch against the Vector DB with smart fallback."""
        t0 = time.perf_counter()
        query = state["search_query"] or state["query"]
        filters = state.get("optimizations", {}).get("metadata_filters", {})
        # Apply tenant scoping if enabled
        if os.getenv("QDRANT_MULTI_TENANT", "false").lower() == "true" and state.get("tenant_id"):
            self.vector_store.tenant_id = state.get("tenant_id")
        
        # 1. Generate Query Embedding
        query_tensor = self.embedding_model.generate_embedding(query)
        
        # 2. Primary Search (Filtered)
        # We attempt retrieval using the dynamically extracted metadata filters (e.g. author, doc_type)
        chunks = self.vector_store.search(query_tensor, k=30, metadata_filters=filters)

        # Optional: merge in retrieval results from extra ephemeral collections (uploaded files)
        extra_collections = state.get("extra_collections", []) or []
        if extra_collections:
            for collection_name in extra_collections:
                try:
                    extra_store = QdrantStore(collection_name=collection_name, dimension=self.vector_store.dimension)
                    # Avoid applying KB metadata filters to uploaded files (they may not contain those fields).
                    extra_results = extra_store.search(query_tensor, k=30, metadata_filters=None)
                    for item in extra_results:
                        item["_collection"] = collection_name
                    chunks.extend(extra_results)
                except Exception as e:
                    logger.warning(f"[RAG AGENT] Failed to query extra collection {collection_name}: {e}")
        
        # 3. Smart Fallback: If no results and filters were applied, retry unfiltered
        # This prevents over-restrictive metadata from causing 0-result failures.
        if not chunks and filters:
            logger.info(f"[RAG AGENT] Filtered search returned 0 chunks (Filters: {filters}). Executing smart fallback to unfiltered semantic search.")
            chunks = self.vector_store.search(query_tensor, k=30, metadata_filters=None)
            state["optimizations"]["fallback_triggered"] = True
            state["optimizations"]["original_filters"] = filters
        
        # Optional hybrid lexical rerank on dense results
        use_hybrid = os.getenv("HYBRID_SEARCH", "false").lower() == "true"
        if use_hybrid:
            chunks = self.hybrid_reranker.rerank(query, chunks)

        state["context_chunks"] = chunks
        
        if not chunks:
            logger.warning("[RAG AGENT] Retrieval yielded 0 chunks even after fallback. Bypassing synthesis.")
            state["answer"] = "DATA_NOT_FOUND"
            state["confidence"] = 1.0
            
        t1 = time.perf_counter()
        state.setdefault("latency_optimizations", {})
        state["latency_optimizations"]["retrieval_time_ms"] = round((t1 - t0) * 1000, 3)
        return state

    def node_rerank_documents(self, state: AgentState) -> AgentState:
        """Executes the GPU semantic cross-encoder to isolate the Top 5 chunks."""
        t0 = time.perf_counter()
        query = state["search_query"] or state["query"]
        raw_chunks = state["context_chunks"]
        
        # Compress 30 -> 5
        try:
            top_k = int(os.getenv("RERANK_TOP_K", 5))
        except Exception:
            top_k = 5

        reranker_profile = state.get("optimizations", {}).get("reranker_profile", "auto")
        model_override = state.get("optimizations", {}).get("reranker_model_name")
        reranker_enabled = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

        if reranker_profile == "off":
            reranker_enabled = False
        if not reranker_enabled:
            refined_chunks = raw_chunks[:top_k]
        else:
            if reranker_profile == "fast":
                model_name = "BAAI/bge-reranker-base"
            elif reranker_profile == "accurate":
                model_name = "BAAI/bge-reranker-large"
            else:
                model_name = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")

            if model_override:
                model_name = model_override

            reranker = SemanticReranker.get_instance(model_name=model_name)
            refined_chunks = reranker.rerank(query, raw_chunks, top_k=top_k)
        state["context_chunks"] = refined_chunks
        t1 = time.perf_counter()
        state.setdefault("latency_optimizations", {})
        state["latency_optimizations"]["rerank_time_ms"] = round((t1 - t0) * 1000, 3)
        return state

    async def node_synthesize_answer(self, state: AgentState) -> AgentState:
        """Executes the dynamic generation bounded by the strictly retrieved chunks."""
        query = state["query"]
        chunks = state["context_chunks"]
        
        # -----------------------------------------------------------------
        # PART 2: THE COMPLEXITY ANALYZER (Enterprise Dynamic Routing)
        # -----------------------------------------------------------------
        override_model = state.get("optimizations", {}).get("override_model")
        override_effort = state.get("reasoning_effort")

        if override_model:
            reasoning_effort = override_effort or "high"
            target_model = override_model
            target_temp = 0.2
        else:
            word_count = len(query.split())
            chunk_count = len(chunks)
            
            multi_hop_flags = ["compare", "contrast", "difference", "analyze", "resolve"]
            has_multi_hop = any(flag in query.lower() for flag in multi_hop_flags)
            
            if word_count > 40 or has_multi_hop:
                reasoning_effort = "high"
                target_model = "openai/gpt-oss-120b"
                target_temp = 0.2
            elif chunk_count > 4:
                reasoning_effort = "medium"
                target_model = "llama-3.3-70b-versatile"
                target_temp = 0.2
            else:
                reasoning_effort = "low"
                target_model = "llama-3.3-70b-versatile"
                target_temp = 0.0
            
        logger.info(f"[RAG AGENT] Complexity Analyzer computed -> Effort: {reasoning_effort.upper()} | Selected Engine: {target_model}")
        
        state["reasoning_effort"] = reasoning_effort
        state.setdefault("latency_optimizations", {})
        state["latency_optimizations"].update({
            "reasoning_effort": reasoning_effort,
            "model_selected": target_model,
            "short_circuited": False
        })
        
        # Pull Prompt Rewriter outputs if they successfully propagated
        optimized = state.get("optimized_prompts", {})
        
        prompt_b = optimized.get("deep_high", {}).get("prompt", query)
        
        # Execute concurrent Candidate Syntheses via threadpool wrapping to prevent Event Loop deadlocks
        logger.info("[RAG AGENT] Spawning Concurrent A/B MoE Sythesis threads...")
        
        llm_t0 = time.perf_counter()
        # Offload the blocking requests.post network calls to worker threads
        result_a_task = asyncio.to_thread(self.synthesis_engine.synthesize, query, chunks, target_model, target_temp)
        result_b_task = asyncio.to_thread(self.synthesis_engine.synthesize, prompt_b, chunks, target_model, target_temp)
        
        # Await both LLM generations concurrently 
        result_a, result_b = await asyncio.gather(result_a_task, result_b_task)
        llm_t1 = time.perf_counter()
        state.setdefault("latency_optimizations", {})
        state["latency_optimizations"]["llm_time_ms"] = round((llm_t1 - llm_t0) * 1000, 3)
        
        # Execute RLAIF Selection asynchronously
        reward_t0 = time.perf_counter()
        winning_answer = await self.reward_model.select_best_candidate(
            query=query,
            context=chunks,
            candidate_a=result_a.get("answer", ""),
            candidate_b=result_b.get("answer", "")
        )
        reward_t1 = time.perf_counter()
        state["latency_optimizations"]["llm_time_ms"] += round((reward_t1 - reward_t0) * 1000, 3)
        
        # Re-attach provenance/confidence matching the Standard result mapping globally (Mocked merge for Phase 13)
        state["answer"] = winning_answer
        if winning_answer == result_b.get("answer", ""):
            state["sources"] = result_b.get("provenance", [])
            state["confidence"] = result_b.get("confidence", 0.95)
            state["optimizations"]["tokens_input"] = result_b.get("tokens_input", 0)
            state["optimizations"]["tokens_output"] = result_b.get("tokens_output", 0)
            state["optimizations"]["temperature_used"] = result_b.get("temperature_used", 0.0)
        else:
            state["sources"] = result_a.get("provenance", [])
            state["confidence"] = result_a.get("confidence", 0.95)
            state["optimizations"]["tokens_input"] = result_a.get("tokens_input", 0)
            state["optimizations"]["tokens_output"] = result_a.get("tokens_output", 0)
            state["optimizations"]["temperature_used"] = result_a.get("temperature_used", 0.0)
        
        return state

    async def node_verify_answer(self, state: AgentState) -> AgentState:
        """Audits the 70B output utilizing the independent Sarvam Verifier."""
        draft_answer = state["answer"]
        chunks = state["context_chunks"]
        
        # Offload the blocking requests.post Sarvam network call to a worker thread
        t0 = time.perf_counter()
        verification = await asyncio.to_thread(self.verifier.verify, draft_answer, chunks)
        t1 = time.perf_counter()
        state.setdefault("latency_optimizations", {})
        state["latency_optimizations"]["llm_time_ms"] = state["latency_optimizations"].get("llm_time_ms", 0.0) + round((t1 - t0) * 1000, 3)

        state["verification_claims"] = verification.get("claims", [])
        state["verifier_verdict"] = verification.get("overall_verdict", "UNVERIFIED")
        state["is_hallucinated"] = verification.get("is_hallucinated", False)
        
        # Adjust confidence mathematically if Sarvam detects a logic flaw
        if state["is_hallucinated"]:
             state["confidence"] = state["confidence"] * 0.5

        # Correction loop if hallucination detected
        state = await self._run_correction_loop(state)

        # Reward scoring (grounding/actionability/conciseness/coherence)
        try:
            t0 = time.perf_counter()
            reward = await self.reward_model.score_response(
                query=state.get("query", ""),
                context=state.get("context_chunks", []),
                response=state.get("answer", "")
            )
            t1 = time.perf_counter()
            state["latency_optimizations"]["llm_time_ms"] = state["latency_optimizations"].get("llm_time_ms", 0.0) + round((t1 - t0) * 1000, 3)
            state["optimizations"]["reward_score"] = reward
        except Exception:
            state["optimizations"]["reward_score"] = 0.0
             
        return state

    def node_format_output(self, state: AgentState) -> AgentState:
        """Injects Markdown citational formatting ensuring Streamlit UI compatibility."""
        # The formatter returns the complete output dictionary mapping
        formatted_state = ResponseFormatter.construct_final_response(state)
        
        # Mutate the mutable LangGraph state safely
        state["answer"] = formatted_state["answer"]
        state["sources"] = formatted_state["sources"]
        state["confidence"] = formatted_state["confidence"]
        state["verifier_verdict"] = formatted_state["verifier_verdict"]
        state["is_hallucinated"] = formatted_state["is_hallucinated"]
        state["optimizations"] = formatted_state["optimizations"]
        state["latency_optimizations"] = formatted_state.get("latency_optimizations", {})
        
        logger.info("[RAG AGENT] Execution DAG Completed.")
        return state

    async def _run_correction_loop(self, state: AgentState) -> AgentState:
        """
        If verifier flags hallucination, re-run synthesis with stronger grounding constraint.
        """
        if not state.get("is_hallucinated"):
            return state

        logger.warning("[RAG AGENT] Hallucination flagged. Triggering correction loop.")
        query = state["query"]
        chunks = state.get("context_chunks", [])
        if not chunks:
            return state

        # Force a strict grounding prompt
        correction_prompt = (
            "Answer using ONLY the provided CONTEXT_CHUNKS. "
            "If the answer is not in context, respond with 'I don't know based on the provided documents.' "
            f"\n\nUSER_QUERY: {query}"
        )

        override_model = state.get("optimizations", {}).get("override_model")
        target_model = override_model or "llama-3.3-70b-versatile"

        result = await asyncio.to_thread(
            self.synthesis_engine.synthesize,
            correction_prompt,
            chunks,
            target_model,
            0.0,
        )
        state["answer"] = result.get("answer", state["answer"])
        state["sources"] = result.get("provenance", state.get("sources", []))
        state["confidence"] = float(result.get("confidence", state.get("confidence", 0.0))) * 0.7
        return state
