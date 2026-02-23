"""
Smalltalk / Greeting Persona Bypass Agent

Designed as a high-speed circuit-breaker. When the Supervisor detects simple 
pleasantries (e.g., "Hi", "Thanks"), this agent answers instantly, bypassing 
latency-heavy Database Retrieval and Token Embedding phases completely.
"""
from backend.agents.base import BaseAgent
from backend.orchestrator.state import AgentState

class SmalltalkAgent(BaseAgent):
    """
    The Smalltalk/Greeting Agent.
    Bypasses deep retrieval pipelines to offer an instant persona response.
    Saves thousands of context tokens per trivial interacting query.
    """
    
    def execute(self, state: AgentState) -> dict:
        """
        Instantly outputs a hardcoded (or lightweight LLM) template greeting
        and positively evaluates interaction metrics without vector fetching.
        """
        print("Smalltalk Agent: Executing instant bypass...")
        
        # In a fully dynamic architecture, this could be a rapid Llama 3 API call.
        # But for maximum latency reduction and cost-saving, we deploy a programmatic persona template.
        answer = "Hello — I’m Galactus. I help answer questions using your uploaded documents and connected knowledge sources.\nYou can ask me about your files, compare documents, extract information, or reason across sources."
        
        # -------------------------------------------------------------------------
        # Programmatic Streaming UI Emulation
        # -------------------------------------------------------------------------
        # If a streaming callback exists (Streamlit UI attached state), we mimic typewriter 
        # generation cadence so the user logically processes the standard latency response timings.
        streaming_callback = state.get("streaming_callback")
        if streaming_callback:
            import time
            for word in answer.split():
                streaming_callback(word + " ")
                time.sleep(0.01)
                
        # Return state with 100% Traceability Metrics generated synthetically 
        return {
            "answer": answer,
            "sources": [],
            "context_chunks": [],
            "context_text": "",
            "confidence": 1.0,           # 100% confident it correctly said "Hello"
            "verifier_verdict": "SUPPORTED",
            "is_hallucinated": False,
            "optimizations": {
                "short_circuited": True, # Flags the graph that massive DB work was explicitly skipped
                "temperature": 0.2,      # Tracked purely for telemetry accuracy mappings
                "reasoning_effort": "low",
                "agent_routed": "SmalltalkAgent"
            }
        }
