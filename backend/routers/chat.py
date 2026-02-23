"""
Chat API Router Module

This file handles the primary `/api/v1/chat/` endpoint. It processes multimodal 
inputs (text, audio, images), initializes the RAG service globally to prevent load times,
and routes payloads into the LangGraph orchestrator.
"""
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from typing import Optional, List
from pydantic import BaseModel
import logging
from backend.generation.rag_service import RAGService

# Initialize the router with a specific prefix to namespace the endpoints
router = APIRouter(
    prefix="/api/v1/chat",
    tags=["Chat & Agents"],
)

# Set up logging for this specific module strictly to monitor chat behavior
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global Service Initialization
# -----------------------------------------------------------------------------
# We use a global singleton pattern for the RAGService implementation.
# In production, this might be replaced with standard FastAPI Dependency Injection (`Depends()`).
# This prevents reloading massive LLM contexts and embedding models for every single chat request.
_rag_service = None

def get_rag_service(api_key: str = None) -> RAGService:
    """
    Retrieves or establishes the global RAGService singleton instance.
    
    If the requested API key differs from the currently bound API key, it will 
    force a re-initialization of the service to ensure the payload utilizes the new key.
    
    Args:
        api_key (str, optional): An explicit LLM key provided via API payload.
        
    Returns:
        RAGService: An initialized instance of the LangGraph-powered generation engine.
    """
    global _rag_service
    import os
    
    # Read the current environment API key to check for drift
    current_key = os.getenv("SARVAM_API_KEY", "")
    
    # If the service isn't loaded, or the user passes a *new* valid API key, initialize it.
    if _rag_service is None or (api_key and api_key != current_key):
        logger.info(f"Initializing RAG Service for Chat API (API Key override: {bool(api_key)})...")
        # Overwrite the environment permanently for this worker thread if a key is provided
        if api_key:
            os.environ["SARVAM_API_KEY"] = api_key
            
        # Instantiate the heavy LangGraph orchestrator exactly once
        _rag_service = RAGService()
        
    return _rag_service

# -----------------------------------------------------------------------------
# API Request / Response Schemas
# -----------------------------------------------------------------------------
class ChatResponse(BaseModel):
    """
    Pydantic Response Model defining the strict outgoing JSON contract for the Chat API.
    
    This schema guarantees the frontend receives exactly the right fields, 
    especially the crucial LLM Traceability metrics like the actual internal agent used, 
    the hallucination flag, and retrieval confidence statistics.
    """
    session_id: str                      # Unique identifier for tracking conversation turns
    reply: str                           # The final text synthesized by the LLM
    agent_used: str                      # The explicitly determined routing path (e.g., 'RAGAgent', 'Smalltalk')
    confidence_score: float              # The retrieval accuracy / relevance score
    sources: List[dict]                  # The exact contextual chunks retrieved from VectorDB
    verifier_verdict: Optional[str] = None # Output of the independent Fact-Checker (PASS/HALLUCINATION)
    is_hallucinated: Optional[bool] = False # Explicit boolean flag signalling semantic drift
    search_query: Optional[str] = None   # The actual query embedded (e.g., after pronoun resolution)
    optimizations: Optional[dict] = {}   # Tracing context containing temperatures and hidden heuristics

# -----------------------------------------------------------------------------
# Chat Endpoint Definition
# -----------------------------------------------------------------------------
@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    # Use Form data for text fields so the endpoint can simultaneously accept multi-part file payloads
    message: str = Form(...),
    session_id: str = Form(None),
    sarvam_api_key: str = Form(None, description="The Sarvam API Key"),
    
    # Multimodal binary attachments are natively hooked as UploadFile instances
    audio_file: UploadFile = File(None),
    image_file: UploadFile = File(None),
    document_file: UploadFile = File(None)
):
    """
    **Multimodal Agentic Chat Endpoint**
    
    Accepts standard text messages alongside optional audio voice notes (.wav), 
    images (.jpg/.png), and temporary context files (.pdf).
    
    The API payload is compiled, pre-processed for transcriptions/vision extraction, 
    and ultimately routed through the Galactus Supervisor Agent StateGraph.
    """
    try:
        # 1. Pre-processing Multimodal attachments
        transcribed_text = ""
        visual_context = ""
        
        # Audio Block: Intercepts Voice Notes for offline transcription Phase 8 integration
        if audio_file:
            logger.info(f"Received audio file: {audio_file.filename}")
            # Placeholder for faster-whisper implementation
            transcribed_text = "[Simulated Audio Transcription: User Issue]"
            
        # Image Block: Intercepts Images for Gemini Vision extraction Phase 8 integration
        if image_file:
            logger.info(f"Received image file: {image_file.filename}")
            # Placeholder for Gemini vision proxy
            visual_context = "[Simulated Vision Extraction: Error Code 500 Screen]"
            
        # Combine all parsed context dynamically into the final structured query sent to the LLM
        final_query = message
        if transcribed_text:
            final_query += f" | Voice Note Transcription: {transcribed_text}"
        if visual_context:
            final_query += f" | Image Context: {visual_context}"
            
        # -----------------------------------------------------------------------------
        # 2. Agentic Execution 
        # -----------------------------------------------------------------------------
        logger.info(f"Executing Agent Pipeline for query: {final_query}")
        
        # Instantiate or fetch the graph architect
        rag_service = get_rag_service(api_key=sarvam_api_key)
        
        # Execute the primary multi-agent graph with the synthesized prompt
        # This will process RAG, analyze intents, check hallucinations, and return a dictionary
        result = rag_service.answer_query(query=final_query)
        
        # Extract the exact internal agent mechanism from the telemetry (e.g., 'AnalyticalAgent', 'Supervisor')
        routing_path = result.get("optimizations", {}).get("agent_routed", result.get("intent", "Supervisor")).title()
        
        # -----------------------------------------------------------------------------
        # 3. Payload Construction
        # -----------------------------------------------------------------------------
        # Map the internal dictionary dict contract back to the explicit Pydantic response schema
        return ChatResponse(
            session_id=session_id or "session_default_01",
            reply=result.get("answer", "I could not generate an answer at this time."),
            agent_used=routing_path,
            confidence_score=result.get("confidence", 0.0),
            sources=result.get("sources", []),
            verifier_verdict=result.get("verifier_verdict", "UNKNOWN"),
            is_hallucinated=result.get("is_hallucinated", False),
            search_query=result.get("search_query", final_query),
            optimizations=result.get("optimizations", {})
        )
        
    except Exception as e:
        # Catch unexpected fatal pipeline failures, log properly, and shield the trace from users
        logger.error(f"Chat API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Agent Error")
