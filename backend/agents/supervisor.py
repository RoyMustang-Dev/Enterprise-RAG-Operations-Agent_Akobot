"""
Supervisor Agent (Intent Classifier) Module

Acts as the Chief Routing Officer of the LangGraph DAG.
Instead of answering questions directly, it uses a rigid LLM classification prompt 
to parse the user's intent and dynamically route the query to the correct expert agent.
"""
from backend.agents.base import BaseAgent
from backend.orchestrator.state import AgentState

class SupervisorAgent(BaseAgent):
    """
    The Orchestrator / Semantic Router.
    Analyzes the user's raw query and routes them to the correct sub-agent node 
    (Smalltalk, RAG, or Analytical) to optimize token costs and retrieval accuracy.
    """
    
    def execute(self, state: AgentState) -> dict:
        """
        Executes the zero-shot intent classification logic.
        
        Args:
            state (AgentState): The current graph state containing the user query.
            
        Returns:
            dict: The state update dictionary containing the resolved 'intent' string.
        """
        query = state.get("query", "")
        
        # -------------------------------------------------------------------------
        # Classification Prompt Construction
        # -------------------------------------------------------------------------
        # We use a strict prompt to force categorical routing without conversational fluff.
        # This acts as a logical function call rather than a chatbot response.
        supervisor_prompt = """You are the Semantic Router for an Enterprise Knowledge Assistant.
Analyze the user's query and categorize their exact intent into exactly ONE of the following three categories.

CATEGORIES:
1. "smalltalk" : The user is saying hello, greeting, asking how you are, thanking you, or engaging in general chit-chat unrelated to documents. (e.g. "hi", "how are you", "thanks", "hhi", "hellow")
2. "analytical" : The user is asking to compare data, calculate something, extract a specific deeply nested fact, or requesting you to "reason" across the documents. (e.g. "Compare X and Y", "Is Aditya eligible based on criteria X")
3. "rag" : The user is asking a standard informational question about the documents, requesting summaries, explanations, or facts. (e.g. "What is the policy on X?", "Explain document Y")

OUTPUT INSTRUCTIONS:
Return ONLY the exact category name ("smalltalk", "analytical", or "rag"). Do not include any other text or punctuation.
"""
        try:
            # -------------------------------------------------------------------------
            # LLM Execution
            # -------------------------------------------------------------------------
            # We enforce temperature=0.0 to ensure the LLM behaves deterministically like 
            # a reliable router rather than creatively hallucinating category names.
            intent = self.llm_client.generate(
                prompt=query,
                system_prompt=supervisor_prompt,
                temperature=0.0
            ).strip().lower()
            
            # -------------------------------------------------------------------------
            # Output Sanitization & Heuristic Correction
            # -------------------------------------------------------------------------
            # Clean up potential LLM hallucination or surrounding conversational text 
            # if the model ignores the strict output instructions.
            if "smalltalk" in intent or "greet" in intent or "hello" in intent:
                intent = "smalltalk"
            elif "analytical" in intent or "compare" in intent or "reason" in intent:
                intent = "analytical"
            else:
                # Default to RAG if the intent is ambiguous, ensuring the system safely 
                # attempts to search the Vector Database rather than crashing.
                intent = "rag"
                
            return {"intent": intent}
        except Exception as e:
            # Safe logical fallback if the LLM API router goes down or connection drops
            print(f"Supervisor Error: {e}")
            return {"intent": "rag"}
