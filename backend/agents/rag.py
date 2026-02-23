"""
Grounded RAG (Retrieval-Augmented Generation) Agent

This is the core informational actor of the LangGraph DAG.
It intercepts state, triggers the VectorDB Retriever to fetch strictly relevant 
context chunks, and executes a grounded generation prompt using the assigned LLM.
"""
from backend.agents.base import BaseAgent
from backend.orchestrator.state import AgentState
import logging

class RAGAgent(BaseAgent):
    """
    The core Document Retrieval & Q&A Agent.
    Executes grounded RAG generation using retrieved semantic context.
    """
    
    def __init__(self, llm_client=None, retriever=None):
        """
        Initializes the RAG Agent with Generation and Retrieval dependencies.
        
        Args:
            llm_client: The model wrapper allocated by the Orchestrator.
            retriever: The active Vector Store (FAISS/Qdrant) search tool.
        """
        super().__init__(llm_client)
        self.retriever = retriever
        
    def execute(self, state: AgentState) -> dict:
        """
        Executes the RAG pipeline on the current state graph.
        
        Args:
            state (AgentState): The global system dict holding the user query.
            
        Returns:
            dict: The mutated state containing the retrieved chunks and synthesized answer.
        """
        # Prioritize an AI-Rewritten Search Query if available, otherwise fallback to the raw query
        query = state.get("search_query") or state.get("query", "")
        context_chunks = state.get("context_chunks", [])
        context_text = state.get("context_text", "")
        
        # -------------------------------------------------------------------------
        # 1. Semantic Retrieval (If not pre-fetched by another node)
        # -------------------------------------------------------------------------
        if not context_chunks and self.retriever:
            print(f"RAG Agent: Retrieving context for '{query}'...")
            # Execute embedding generation and high-dimensional cosine similarity search
            context_chunks = self.retriever.search(query)
            
            # Format raw dictionaries into a structured text string for the LLM Prompt Sequence
            if context_chunks:
                context_text = "\n\n".join(
                    [f"FILE: {c.get('source', 'Unknown')}\n{c.get('text', '')}" for c in context_chunks]
                )
            else:
                context_text = ""
                
        # -------------------------------------------------------------------------
        # 2. Safety Fallback (Context Missing)
        # -------------------------------------------------------------------------
        # If the database returns completely empty results, immediately halt generation.
        # This prevents the LLM from hallucinating an answer outside the knowledge base context.
        if not context_text:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                "sources": [],
                "context_chunks": [],
                "context_text": ""
            }
            
        # -------------------------------------------------------------------------
        # 3. Grounded Prompt Engineering
        # -------------------------------------------------------------------------
        system_prompt = """You are Galactus — a grounded enterprise knowledge assistant.

Your mandate is to answer the user's Question using strictly and ONLY the provided Context chunks.
The Context contains multiple FILE sections representing raw text extracted from uploaded enterprise documents.

RULES:
1. Read the Context carefully.
2. If the user's question cannot be answered using the Context, explicitly state: "I don't know based on the provided context." DO NOT improvise or use outside knowledge.
3. If the answer is in the Context, provide a clear, concise, and direct response.
4. When you provide facts from the context, naturally weave the filename into your sentence (e.g., "According to 'Annual_Report.pdf', the revenue was..."). DO NOT output raw formatting like `FILE: ...`. Treat each FILE independently.

Accuracy is more important than completeness."""

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        
        print("RAG Agent: Synthesizing Answer...")
        
        # Determine strict generation heuristics
        temperature = 0.4
        reasoning = "low"
        
        # Failsafe telemetry boundary marker: If analytical routing accidentally fell through to RAG, elevate compute thresholds
        if state.get("intent") == "analytical":
            temperature = 0.7
            reasoning = "high"
            
        # -------------------------------------------------------------------------
        # 4. LLM Generation Pipeline
        # -------------------------------------------------------------------------
        answer = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning,
            stream=state.get("streaming_callback") is not None
        )
        
        # Unpack execution generator if active streaming UI pipeline is attached to pipeline loop
        if hasattr(answer, '__iter__') and not isinstance(answer, str):
            final_ans = ""
            for chunk in answer:
                if state.get("streaming_callback"):
                    state["streaming_callback"](chunk)
                final_ans += chunk
            answer = final_ans
            
        # UI NOTE: We deliberately do not strip `<think>` syntax markers here. 
        # The Traceability Streamlit UI explicitly extracts them for the expanding thought block.
        
        return {
            "answer": answer,
            "context_chunks": context_chunks,
            "context_text": context_text,
            "sources": context_chunks,
            "optimizations": {
                "temperature": temperature,
                "reasoning_effort": reasoning,
                "agent_routed": "RAGAgent"
            }
        }
