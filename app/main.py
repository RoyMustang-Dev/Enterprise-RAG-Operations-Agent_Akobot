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

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Patch qdrant-client grpc annotations before any qdrant imports
from app.infra.qdrant_patch import ensure_qdrant_grpc_compat
ensure_qdrant_grpc_compat()

# Import the new Vertical Slice router
from app.api.routes import router as api_router
from app.infra.otel import init_otel
from app.infra.hardware import HardwareProbe
from app.infra.model_bootstrap import configure_model_cache, preload_models
from app.infra.database import cleanup_expired_tenant_dbs, set_current_tenant

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

# Reduce noisy library logs for readability
for noisy_name in [
    "httpx",
    "urllib3",
    "asyncio",
    "sentence_transformers",
    "qdrant_client",
    "uvicorn.access",
]:
    logging.getLogger(noisy_name).setLevel(logging.WARNING)

# Log provider availability summary once on boot
logger.info(
    "[BOOT] Providers: modelslab=%s gemini=%s groq=%s openai=%s anthropic=%s",
    bool(os.getenv("MODELSLAB_API_KEY")),
    bool(os.getenv("GEMINI_API_KEY")),
    bool(os.getenv("GROQ_API_KEY")),
    bool(os.getenv("OPENAI_API_KEY")),
    bool(os.getenv("ANTHROPIC_API_KEY")),
)

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
# Metrics (Prometheus)
# -----------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
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

# -----------------------------------------------------------------------------
# Tenant DB Cleanup (Background Timer)
# -----------------------------------------------------------------------------
def _start_tenant_db_cleanup_daemon():
    interval_hours = int(os.getenv("TENANT_DB_CLEANUP_INTERVAL_HOURS", "24"))
    ttl_days = int(os.getenv("TENANT_DB_TTL_DAYS", "30"))

    def _loop():
        logger.info(f"[CLEANUP] Tenant DB cleanup started. Interval={interval_hours}h TTL={ttl_days}d")
        while True:
            try:
                removed = cleanup_expired_tenant_dbs(ttl_days=ttl_days)
                if removed:
                    logger.info(f"[CLEANUP] Removed {removed} expired tenant DB folders.")
            except Exception as e:
                logger.warning(f"[CLEANUP] Tenant DB cleanup failed: {e}")
            time.sleep(interval_hours * 3600)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()

_start_tenant_db_cleanup_daemon()

# Optional model preload (controlled by PRELOAD_MODELS=true)
preload_models()

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    # Bind tenant context for DB routing/logging
    tenant_id = request.headers.get("x-tenant-id")
    set_current_tenant(tenant_id)
    try:
        response = await call_next(request)
    finally:
        set_current_tenant(None)
    elapsed = round((time.perf_counter() - start) * 1000, 3)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed} ms)")
    try:
        elapsed_sec = elapsed / 1000.0
        REQUEST_COUNT.labels(
            method=request.method,
            path=request.url.path,
            status=str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            path=request.url.path,
        ).observe(elapsed_sec)
    except Exception:
        pass
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
    from app.infra.model_registry import PHASE_MODELS
    from app.infra.provider_router import ProviderRouter
    hw_config = HardwareProbe.detect_environment()

    router = ProviderRouter()
    # Surface high-level model map (phase -> model) without secrets
    phase_models = {k: v.get("model") for k, v in PHASE_MODELS.items()}

    return {
        "status": "healthy",
        "hardware_profile": hw_config,
        "providers": {
            "modelslab": bool(os.getenv("MODELSLAB_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "preferred_order": router.preferred_order,
        },
        "active_models": phase_models,
        "embeddings_provider": "gemini-embedding-001" if os.getenv("GEMINI_API_KEY") else "BAAI/bge-large-en-v1.5",
        "stt_model": os.getenv("STT_MODEL_NAME", "openai/whisper-small"),
        "tts_model": os.getenv("TTS_MODEL_NAME", "tts_models/en/ljspeech/tacotron2-DDC"),
    }

@app.get("/metrics", tags=["Observability"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)



@app.get("/health", tags=["Health"], include_in_schema=False)
async def legacy_health_alias():
    """Backward-compatible alias for health checks."""
    return await health_check()

# -----------------------------------------------------------------------------
# Application Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
