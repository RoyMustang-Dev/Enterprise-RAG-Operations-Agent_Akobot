"""
RAG Service / UI Adapter Module

This module bridges the gap between the stateless FastAPI endpoints (or Streamlit UI) 
and the stateful LangGraph orchestrator. It acts as the Controller, 
initializing graph dependencies, intercepting graph outputs, and silently firing 
off asynchronous threads for Enterprise Audit Logging.
"""
from typing import List, Dict, Tuple
from backend.embeddings.embedding_model import EmbeddingModel
from backend.vectorstore.faiss_store import FAISSStore
import os
import threading
import json
from datetime import datetime
from backend.orchestrator.graph import GalactusOrchestrator

class RAGService:
    """
    Adapter wrapper that instantiates the Agentic Orchestrator and maintains strict UI JSON payload compatibility.
    """

    def __init__(self, **kwargs):
        """
        Initializes the heavy dependencies exactly once per service lifecycle 
        to prevent redundant 1GB model reloads in memory.
        """
        self.embedding_model = EmbeddingModel()
        self.vector_store = FAISSStore()
        
        # Determine strict AI provider rules from the instantiation payload
        self.model_provider = kwargs.get("model_provider", "sarvam").lower()
        self.model_name = kwargs.get("model_name", "sarvam-m")
        
        from dotenv import load_dotenv
        load_dotenv()
        api_key = kwargs.get("api_key") or os.getenv("SARVAM_API_KEY")

        print(f"RAGService Adapter: Initializing via LLMRouter for provider: {self.model_provider}...")
        try:
            from backend.generation.llm_router import LLMRouter
            # Request the multiplexer generator to build the correct Client API Wrapper object instance
            self.llm_client = LLMRouter.get_best_available_llm(target_provider=self.model_provider, api_key=api_key)
        except Exception as e:
            print(f"Failed to route LLM: {e}")
            raise ValueError(f"CRITICAL: Router failed to instantiate explicitly targeted model {self.model_provider}")
            
        # -------------------------------------------------------------------------
        # 1. Graph Instantiation
        # -------------------------------------------------------------------------
        # Construct the immutable nodal graph DAG execution paths, injecting the actively initialized database and API connections
        self.orchestrator = GalactusOrchestrator(
            llm_client=self.llm_client,
            faiss_store=self.vector_store,
            embedding_model=self.embedding_model
        )

    def answer_query(self, query: str, initial_k: int = 20, final_k: int = 5, status_callback=None, streaming_callback=None) -> Dict:
        """
        Executes the Agentic Graph and maps its raw outputs back to the strict dictionary UI contract.
        
        Args:
            query (str): The raw text/multimodal user query string.
            initial_k (int): Legacy retrieval limit (moved internally to RetrieverTool logic module).
            final_k (int): Legacy slicing limit (moved internally to RetrieverTool logic module).
            status_callback (callable, optional): UI hook updating "Analyzing..." loading text spinners.
            streaming_callback (callable, optional): Explicit UI hook enabling typewriter character text generation pipes.
            
        Returns:
            Dict: The final structured response payload mapped perfectly to the FastAPI Pydantic schema schemas.
        """
        if status_callback:
            status_callback("Supervisor routing...")
            
        chat_history = []

        # -------------------------------------------------------------------------
        # 2. Synchronous Graph Execution Lock
        # -------------------------------------------------------------------------
        result_state = self.orchestrator.execute(
            query=query,
            streaming_callback=streaming_callback,
            chat_history=chat_history
        )
        
        # -------------------------------------------------------------------------
        # 3. Enterprise Audit Logging (Telemetry Generation Trace)
        # -------------------------------------------------------------------------
        # We spawn a discrete background parallel thread so the UI/FastAPI response is instantly 
        # returned to the client user, while the IO-blocking JSONL file system write happens invisibly.
        self._audit_log({
            "query": query,
            "intent": result_state.get("intent"), # Indicates logical routing mapping: 'rag', 'analytical', 'smalltalk'
            "model_provider": self.model_provider,
            "latency_optimizations": result_state.get("optimizations", {}),
            "confidence": result_state.get("confidence", 0.0), # Synthesized relevance threshold 
            "verifier_verdict": result_state.get("verifier_verdict", "UNKNOWN"),
            "is_hallucinated": result_state.get("is_hallucinated", False),
            "agent_routed": result_state.get("optimizations", {}).get("agent_routed", result_state.get("intent")) # Actual worker node path
        })
        
        # -------------------------------------------------------------------------
        # 4. Strict Contract Output Formatting Dictionary Mapping
        # -------------------------------------------------------------------------
        return {
            "answer": result_state.get("answer", "No answer generated."),
            "sources": result_state.get("sources", []),
            "confidence": result_state.get("confidence", 0.0),
            "verifier_verdict": result_state.get("verifier_verdict", "UNKNOWN"),
            "is_hallucinated": result_state.get("is_hallucinated", False),
            "attributions": [], # Legacy UI payload compatibility mapping 
            "search_query": result_state.get("search_query", query),
            "optimizations": result_state.get("optimizations", {})
        }

    def _audit_log(self, payload: dict):
        """
        Asynchronously log user interaction traces and routing paths for strict enterprise observability tracking.
        """
        def write_log():
            # Build an OS-agnostic path explicitly locating the master backend tracking directory layout
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'audit')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'audit_logs.jsonl')
            payload["timestamp"] = datetime.utcnow().isoformat()
            try:
                # Open in explicitly continuous Append mode so historical tracking queries are never wiped
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
            except Exception as e:
                # Catch errors silently; an offline audit log failure should NEVER crash the active chat user runtime
                print(f"Audit Log File Write Failed: {e}")
                
        # Detach the file-write execution entirely from the main thread stack
        threading.Thread(target=write_log, daemon=True).start()
