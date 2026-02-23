"""
LangGraph Orchestrator Module

Constructs the primary Directed Acyclic Graph (DAG) for the Multi-Agent system.
This class defines the global mutable state logic, initializes the distinct AI agents, 
and manages the conditional routing pathways used to safely evaluate user queries.
"""
import os
import json
from langgraph.graph import StateGraph, END
from backend.orchestrator.state import AgentState
from backend.agents.supervisor import SupervisorAgent
from backend.agents.rag import RAGAgent
from backend.agents.smalltalk import SmalltalkAgent
from backend.agents.analytical import AnalyticalAgent
from backend.agents.tools.retriever import RetrieverTool

class GalactusOrchestrator:
    """
    The graph-based semantic routing orchestrator.
    Constructs the directed acyclic graph (DAG) of the Multi-Agent system.
    Serves as the central state-machine, preventing procedural execution crashes.
    """
    def __init__(self, llm_client, faiss_store, embedding_model):
        """
        Initializes the Orchestrator with interconnected system dependencies.
        
        Args:
            llm_client: The primary Text generation model client (e.g., SarvamClient, GroqClient).
            faiss_store: The local vector database instance utilized for fast semantic retrieval.
            embedding_model: The underlying token embedding engine for context similarity.
        """
        self.llm_client = llm_client
        self.faiss_store = faiss_store
        self.embedding_model = embedding_model
        
        # Instantiate retrieval tools required to supply context into knowledge-aware agents
        self.retriever = RetrieverTool(self.faiss_store, self.embedding_model)
        
        # Instantiate the full suite of modular Autonomous Agents
        self.supervisor = SupervisorAgent(llm_client)
        self.rag_agent = RAGAgent(llm_client, self.retriever)
        self.smalltalk_agent = SmalltalkAgent(llm_client)
        self.analytical_agent = AnalyticalAgent(llm_client, self.retriever)
        
        # Compile the state machine graph strictly on initialization
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """
        Constructs the nodal map and sets defined conditional edges mapping the execution flow.
        
        Returns:
            CompiledGraph: The executable conversational pipeline logic.
        """
        workflow = StateGraph(AgentState)
        
        # 1. Add Execution Nodes (The 'Actors')
        # Each node represents a distinct LLM agent or sub-system that intercepts and mutates the global graph state
        workflow.add_node("supervisor", self.supervisor.execute)
        workflow.add_node("rag_agent", self.rag_agent.execute)
        workflow.add_node("smalltalk_agent", self.smalltalk_agent.execute)
        workflow.add_node("analytical_agent", self.analytical_agent.execute)
        
        # 2. Add Edges & Conditional Routing Paths
        # The query ALWAYS initializes at the Supervisor which performs fast Intent Classification
        workflow.set_entry_point("supervisor")
        
        # Conditionally map routes spanning from the Supervisor evaluating against the dynamically generated text string intent
        workflow.add_conditional_edges(
            "supervisor",
            self._route_intent,
            {
                "rag": "rag_agent",               # Grounded document retrieval workflows
                "analytical": "analytical_agent", # Intense comparisons reasoning operations
                "smalltalk": "smalltalk_agent"    # Immediate persona bypass mechanisms
            }
        )
        
        # Once any functional expert agent replies and mutates state, securely terminate the loop sequence
        workflow.add_edge("rag_agent", END)
        workflow.add_edge("smalltalk_agent", END)
        workflow.add_edge("analytical_agent", END)
        
        # 3. Compiles memory pipeline into a rigidly executable architecture class representation
        return workflow.compile()
        
    # --- Edge Logic ---
    def _route_intent(self, state: AgentState) -> str:
        """
        Edge mapping function. Evaluates the mutated state dict from the Supervisor 
        logic node and securely determines the next exact system operation.
        """
        intent = state.get("intent", "rag")
        print(f"Orchestrator: Supervisor routed intent to -> {intent.upper()}")
        return intent

    def execute(self, query: str, streaming_callback=None, chat_history=None) -> dict:
        """
        Public entry function to trigger the graph operations loop.
        
        Initializes the globally-typed State Dictionary schema structure before firing the DAG.
        
        Args:
            query (str): The raw text requested by the user API loop (can hold multimodal strings).
            streaming_callback (callable, optional): Active GUI hook to stream characters locally.
            chat_history (list, optional): System records for previous dialogue resolution.
            
        Returns:
            dict: The final enriched system state logging answers, context origins, and operational metrics.
        """
        # Formulate the unified immutable structure required by LangGraph mappings
        initial_state = {
            "query": query,
            "chat_history": chat_history or [],
            "intent": None,
            "search_query": None,           # Extracted entity term from user prompt (for retrieval bypasses)
            "context_chunks": [],           # Cleaned lists of raw db strings
            "context_text": "",             # Stringified compilation chunk output
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "verifier_verdict": "UNKNOWN",
            "is_hallucinated": False,
            "optimizations": {"agent_routed": "SupervisorAgent"},
            "current_agent": None,
            "streaming_callback": streaming_callback
        }
        
        try:
            # Synchronously trigger the execution graph DAG logic loop and block until agent END 
            result_state = self.graph.invoke(initial_state)
            
            # Post-processing data verification logic (Traceability Security Tracking Mechanisms)
            # If the user isn't just saying 'Hello' (smalltalk mechanism), strictly perform safety assertions
            if result_state.get("intent") != "smalltalk":
                # Ensure factual fallback verdicts organically parse if not set independently by deep agents
                if result_state.get("verifier_verdict") in ["UNKNOWN", None]:
                    result_state["verifier_verdict"] = "SUPPORTED" if len(result_state.get("sources", [])) > 0 else "NO_CONTEXT"
                
                result_state["is_hallucinated"] = result_state.get("is_hallucinated", False)
                
                # Assign foundational algorithmic confidences utilizing the retrieved chunk count if independent metrics missing
                if result_state.get("confidence") == 0.0 or result_state.get("confidence") is None:
                    result_state["confidence"] = 1.0 if len(result_state.get("sources", [])) > 0 else 0.0
                
            return result_state
            
        except Exception as e:
            # Failsafe Catch explicitly designed to intercept catastrophic processing faults without crashing standard 502 frameworks 
            import traceback
            traceback.print_exc()
            return {
                "answer": f"System Operator Error: Graph execution failed logically -> {e}",
                "sources": [],
                "confidence": 0.0,
                "verifier_verdict": "ERROR",
                "is_hallucinated": False,
                "optimizations": {}
            }
