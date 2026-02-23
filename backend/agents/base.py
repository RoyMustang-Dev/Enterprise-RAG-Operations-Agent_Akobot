"""
Agent Base Interface Module

Defines the abstract foundation for all independent AI agents operating within the LangGraph orchestrator.
Enforces strict execution contracts to ensure state mutability remains consistent across the pipeline.
"""
from abc import ABC, abstractmethod
from backend.orchestrator.state import AgentState

class BaseAgent(ABC):
    """
    Abstract interface for all specialized agents in the Graph.
    
    By inheriting from this class, all downstream agents (RAG, Analytical, Smalltalk) 
    are guaranteed to expose the exact same API signature (`execute`) required 
    by the LangGraph execution nodes.
    """
    
    def __init__(self, llm_client=None):
        """
        Initializes the agent with an optional LLM client dependency.
        
        Args:
            llm_client: The initialized language model wrapper (e.g., SarvamClient).
                        Abstracting this allows for easy unit testing and mock injections.
        """
        self.llm_client = llm_client
        
    @abstractmethod
    def execute(self, state: AgentState) -> dict:
        """
        Executes the agent's core capability on the current shared graph state.
        
        Args:
            state (AgentState): The immutable copy of the current global dictionary graph state.
            
        Returns:
            dict: A mapping of keys to values that LangGraph will merge into the master state.
                  (e.g., returning {"intent": "rag"} updates state["intent"]).
        """
        pass
