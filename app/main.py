"""
Main entry point for the Re-Architected Enterprise RAG Backend.

This module initializes the FastAPI application, wires up CORS, and mounts the 
single, highly-cohesive API Router built in `app.api.routes`.
"""
import os
import sys
import asyncio

# CRITICAL for Windows: Playwright subprocesses require the Proactor event loop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # Optional dependency in constrained environments; service can still run with injected env vars.
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the new Vertical Slice router
from app.api.routes import router as api_router

# -----------------------------------------------------------------------------
# FastAPI Application Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Enterprise Agentic RAG API",
    description="Vertical Slice implementation of the ReAct + MoE Architecture.",
    version="2.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Router Mounting
# -----------------------------------------------------------------------------
# Mount all endpoints under the /api/v1 namespace for versioning compliance
app.include_router(api_router, prefix="/api/v1")

# -----------------------------------------------------------------------------
# Root & Health Endpoints
# -----------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "architecture": "Vertical Slice (v2)",
        "service": "Enterprise Agentic RAG API"
    }

@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    # Detect hardware probe targets actively during health checks
    from app.infra.hardware import HardwareProbe
    hw_config = HardwareProbe.detect_environment()
    
    return {
        "status": "healthy",
        "hardware_profile": hw_config,
        "active_models": {
            "synthesis": "llama-3.3-70b-versatile",
            "metadata_extraction": "llama-3.1-8b-instant",
            "verifier": "Sarvam M",
            "reranker": "bge-reranker-large",
            "embeddings": "BAAI/bge-large-en-v1.5"
        }
    }


@app.get("/health", tags=["Health"], include_in_schema=False)
async def legacy_health_alias():
    """Backward-compatible alias for health checks."""
    return await health_check()

# -----------------------------------------------------------------------------
# Application Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
