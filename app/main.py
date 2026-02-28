"""
Main entry point for the Re-Architected Enterprise RAG Backend.

This module initializes the FastAPI application, wires up CORS, and mounts the 
single, highly-cohesive API Router built in `app.api.routes`.
"""
import os
import sys
import asyncio
import logging
import time
import threading
import multiprocessing as mp

# CRITICAL SECURITY/STABILITY PATCH:
# Force Windows to load the pristine cu121 Native DLLs into the Python 
# executable's global namespace BEFORE third-party libraries (like PaddleOCR)
# can blindly inject older cu11 DLL dependencies and trigger WinError 127.
try:
    import torch
except Exception:
    pass

# CRITICAL for Windows: Playwright subprocesses require the Proactor event loop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        mp.set_executable(sys.executable)
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    # Optional dependency in constrained environments; service can still run with injected env vars.
    pass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Patch qdrant-client grpc annotations before any qdrant imports
from app.infra.qdrant_patch import ensure_qdrant_grpc_compat
ensure_qdrant_grpc_compat()

# Import the new Vertical Slice router
from app.api.routes import router as api_router
from app.infra.otel import init_otel
from app.infra.hardware import HardwareProbe
from app.infra.model_bootstrap import configure_model_cache, preload_models

# -----------------------------------------------------------------------------
# FastAPI Application Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Enterprise Agentic RAG API",
    description="Vertical Slice implementation of the ReAct + MoE Architecture.",
    version="2.0.0",
)

# Configure model cache dir before any heavy model loads
configure_model_cache()

# Optional paddle runtime auto-install (controlled by env)


# Log hardware profile on startup
HardwareProbe.get_profile()

# Logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

# Configure CORS
cors_origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
if allow_credentials and not cors_origins:
    # If credentials are enabled, explicit origins are required
    cors_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins or allow_credentials else ["*"],
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Router Mounting
# -----------------------------------------------------------------------------
# Mount all endpoints under the /api/v1 namespace for versioning compliance
app.include_router(api_router, prefix="/api/v1")

# Optional OpenTelemetry bootstrap
init_otel(app)

# -----------------------------------------------------------------------------
# Ephemeral Collection Cleanup (Background Timer)
# -----------------------------------------------------------------------------
def _start_ephemeral_cleanup_daemon():
    interval_min = int(os.getenv("EPHEMERAL_CLEANUP_INTERVAL_MINUTES", "60"))
    from app.multimodal.session_vector import SessionVectorManager
    manager = SessionVectorManager()

    def _loop():
        logger.info(f"[CLEANUP] Ephemeral collection cleanup started. Interval={interval_min}m")
        while True:
            try:
                manager.cleanup_expired()
            except Exception as e:
                logger.warning(f"[CLEANUP] Ephemeral cleanup failed: {e}")
            time.sleep(interval_min * 60)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()

_start_ephemeral_cleanup_daemon()

# Optional model preload (controlled by PRELOAD_MODELS=true)
preload_models()

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 3)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed} ms)")
    return response

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
