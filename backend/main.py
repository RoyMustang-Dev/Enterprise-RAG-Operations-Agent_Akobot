"""
Main entry point for the AKO AI Support Agent FastAPI Backend.

This module initializes the FastAPI application, configures cross-origin 
resource sharing (CORS), sets up necessary event loop policies for asynchronous 
execution on Windows, and mounts all modular API routers (chat, settings, ingestion).
"""
import os
import sys
import asyncio

# CRITICAL for Windows: Playwright subprocesses require the Proactor event loop
# to run concurrently without raising 'NotImplementedError' during execution.
# This allows async libraries to manage subprocesses flawlessly.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import modular API routers that handle specific domain logic
from backend.routers import chat, settings, ingestion

# -----------------------------------------------------------------------------
# FastAPI Application Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="AKO AI Support Agent API",
    description="Multimodal SaaS Intelligence Backend with dynamic LLM routing. "
                "Serves as the core orchestrator for document parsing, embeddings, and chat.",
    version="1.0.0",
)

# Configure CORS (Cross-Origin Resource Sharing) for Frontend Integrations.
# This allows external web clients (like our Streamlit UI or external SaaS dashboards)
# to securely make HTTP requests to this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # In production, this MUST be restricted to specific allowed domain URLs
    allow_credentials=True, # Allow cookies and authorization headers
    allow_methods=["*"],    # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],    # Allow all HTTP headers
)

# -----------------------------------------------------------------------------
# Router Mounting
# -----------------------------------------------------------------------------
# Include the modular routers for cleaner code organization and separation of concerns.
app.include_router(chat.router)       # Handles all chat execution and agent routing
app.include_router(settings.router)   # Handles bot configuration and system prompts
app.include_router(ingestion.router)  # Handles file uploads, parsing, and web crawling

# -----------------------------------------------------------------------------
# Root & Health Endpoints
# -----------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint serving as a basic pulse check for the API.
    
    Returns:
        dict: A simple JSON structure indicating the service is online and responsive.
    """
    return {
        "status": "online",
        "service": "AKO AI Support Agent API",
        "version": "1.0.0"
    }

@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    """
    Advanced health check endpoint that verifies the core systems 
    and dynamically detects which LLM API keys are currently loaded in the environment.
    This is useful for admin dashboards to verify which models are ready for routing.
    
    Returns:
        dict: A JSON object detailing the server status and the list of active 
              Language Model providers (e.g., OPENAI, SARVAM) available for routing.
    """
    # Create a dictionary mapping provider names to a boolean evaluating if their key exists
    detected_keys = {
        "OPENAI": bool(os.getenv("OPENAI_API_KEY")),
        "ANTHROPIC": bool(os.getenv("ANTHROPIC_API_KEY")),
        "GEMINI": bool(os.getenv("GEMINI_API_KEY")),
        "SARVAM": bool(os.getenv("SARVAM_API_KEY")),
    }
    
    return {
        "status": "healthy",
        # Use a list comprehension to only return the keys of providers that evaluate True
        "active_llm_providers": [k for k, v in detected_keys.items() if v]
    }

# -----------------------------------------------------------------------------
# Application Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Launch the multi-threaded ASGI server using uvicorn.
    # host="0.0.0.0" binds the server to all network interfaces.
    # reload=True automatically restarts the server when code changes are detected.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
