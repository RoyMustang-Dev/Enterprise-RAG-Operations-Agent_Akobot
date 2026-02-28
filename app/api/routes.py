"""
Vertical Slice API Gateway

This module exclusively handles HTTP routing, payload validation, and HTTP-level exception handling.
It delegates all complex business logic down to inner slices (`app.supervisor`, `app.ingestion`).
"""
import os
from dotenv import load_dotenv
load_dotenv(override=True)

import uuid
import json
import time
import logging
from typing import List, Dict, Any, Optional, Literal

from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form, Header, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.core.telemetry import ObservabilityLayer
from app.core.rate_limit import TokenBucketRateLimiter
from app.infra.database import init_ingestion_db, get_ingestion_job
from app.infra.job_tracker import JobTracker
from app.infra.hardware import HardwareProbe
from app.core.types import TelemetryLogRecord


# -----------------------------------------------------------------------------
# Global Ingestion Job Tracker Memory Node
# -----------------------------------------------------------------------------
ingestion_jobs: Dict[str, Dict[str, Any]] = {}
init_ingestion_db()

# Global rate limiter (in-memory; replace with Redis-backed limiter in production)
_rate_limiter = TokenBucketRateLimiter()

router = APIRouter()
logger = logging.getLogger(__name__)
telemetry = ObservabilityLayer()

# Lazy singleton orchestrator to avoid recreating heavy components on each request.
_CHAT_ORCHESTRATOR = None
_MULTIMODAL_ROUTER = None


def _get_orchestrator():
    global _CHAT_ORCHESTRATOR
    if _CHAT_ORCHESTRATOR is None:
        from app.supervisor.router import ExecutionGraph

        _CHAT_ORCHESTRATOR = ExecutionGraph()
    return _CHAT_ORCHESTRATOR


def _get_multimodal_router():
    global _MULTIMODAL_ROUTER
    if _MULTIMODAL_ROUTER is None:
        from app.multimodal.multimodal_router import MultimodalRouter
        _MULTIMODAL_ROUTER = MultimodalRouter()
    return _MULTIMODAL_ROUTER


def _chunk_text_for_stream(text: str, size: int = 120):
    if not text:
        return []
    chunks = []
    idx = 0
    while idx < len(text):
        chunks.append(text[idx:idx + size])
        idx += size
    return chunks


# -----------------------------------------------------------------------------
# Pydantic Schemas (FastAPI Inbound/Outbound Contracts)
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's raw prompt.")
    chat_history: Optional[List[Dict[str, Any]]] = Field(default=[], description="Previous conversational turns.")
    model_provider: Literal["groq", "openai", "anthropic", "gemini", "auto"] = Field(
        default="groq",
        description="Requested provider. Use 'auto' to enable provider auto-routing (when PROVIDER_AUTO_ROUTING=true).",
    )
    session_id: Optional[str] = Field(default=None, description="Optional client session identifier for telemetry correlation.")
    stream: Optional[bool] = Field(default=False, description="Enable server-sent events (SSE) streaming output.")
    reranker_profile: Optional[Literal["auto", "accurate", "fast", "off"]] = Field(
        default="auto",
        description="Reranker profile: auto (default), accurate (large), fast (base), or off."
    )
    reranker_model_name: Optional[str] = Field(
        default=None,
        description="Explicit reranker model override (e.g., BAAI/bge-reranker-base)."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Summarize Updated_Resume_DS.pdf.",
                "chat_history": [],
                "model_provider": "groq",
                "session_id": "session-123",
                "stream": False,
                "reranker_profile": "auto"
            }
        }
    }


class ChatResponse(BaseModel):
    session_id: Optional[str] = None
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    verifier_verdict: str
    is_hallucinated: bool
    optimizations: Dict[str, Any]
    chat_history: Optional[List[Dict[str, Any]]] = Field(default=[])
    latency_optimizations: Optional[Dict[str, Any]] = Field(default={})

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "session-123",
                "answer": "Aditya Mishra is a Machine Learning Engineer...",
                "sources": [{"source": "Updated_Resume_DS.pdf", "score": 0.87, "text": "..."}],
                "confidence": 0.95,
                "verifier_verdict": "SUPPORTED",
                "is_hallucinated": False,
                "optimizations": {"agent_routed": "rag_agent", "complexity_score": 0.4},
                "chat_history": [],
                "latency_optimizations": {"llm_time_ms": 12000.0}
            }
        }
    }


class IngestionResponse(BaseModel):
    status: str
    message: str
    job_id: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "accepted",
                "message": "Queued 2 files for vector extraction.",
                "job_id": "job-uuid"
            }
        }
    }


class IngestionStatusResponse(BaseModel):
    collection: str
    mode: Literal["local", "cloud"]
    total_vectors: int
    documents: List[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "collection": "enterprise_rag",
                "mode": "local",
                "total_vectors": 27,
                "documents": ["Updated_Resume_DS.pdf", "support agent.docx"]
            }
        }
    }


class FeedbackRequest(BaseModel):
    session_id: str
    rating: str
    feedback_text: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "session-123",
                "rating": "up",
                "feedback_text": "Great answer.",
                "metadata": {"case": "smoke"}
            }
        }
    }


# -----------------------------------------------------------------------------
# 1. Chat Generation Endpoint
# -----------------------------------------------------------------------------
@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Unified Chat (RAG + Files + Images)",
    description="Primary chat endpoint. Accepts JSON or multipart form-data. "
                "If files are included, they are ingested into a 24h ephemeral collection and merged into retrieval. "
                "Session reuse test: Upload files once with session_id, then send follow-up requests with the same "
                "session_id (and no files) to query previously uploaded content.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": ChatRequest.model_json_schema(),
                    "example": {
                        "query": "Summarize Updated_Resume_DS.pdf.",
                        "chat_history": [],
                        "model_provider": "groq",
                        "session_id": "session-123",
                        "stream": False
                    }
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "chat_history": {"type": "string"},
                            "model_provider": {"type": "string"},
                            "session_id": {"type": "string"},
                            "image_mode": {"type": "string", "enum": ["auto", "ocr", "vision"]},
                            "stream": {"type": "boolean"},
                            "reranker_profile": {"type": "string", "enum": ["auto", "accurate", "fast", "off"]},
                            "reranker_model_name": {"type": "string"},
                            "files": {"type": "array", "items": {"type": "string", "format": "binary"}},
                        },
                        "required": ["query"]
                    }
                }
            }
        }
    }
)
async def chat_endpoint(
    http_request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="x-tenant-id"),
    x_user_id: Optional[str] = Header(default=None, alias="x-user-id"),
):
    """
    **Primary RAG Generation Interface**

    Accepts a user query, triggers supervisor routing, executes grounded generation,
    runs independent verification, and returns structured response metadata.
    
    Optional headers:
    - `x-tenant-id`: tenant or collection namespace
    - `x-user-id`: user identity for telemetry
    """
    # Detect JSON vs multipart body
    content_type = (http_request.headers.get("content-type", "") or "").lower()
    query = None
    chat_history_list = []
    model_provider = "groq"
    session_id = None
    stream = False
    image_mode = "auto"
    files = None
    reranker_profile = "auto"
    reranker_model_name = None

    if content_type.startswith("application/json"):
        payload = await http_request.json()
        parsed = ChatRequest(**payload)
        query = parsed.query
        chat_history_list = parsed.chat_history or []
        model_provider = parsed.model_provider
        session_id = parsed.session_id or str(uuid.uuid4())
        stream = bool(parsed.stream)
        reranker_profile = parsed.reranker_profile or "auto"
        reranker_model_name = parsed.reranker_model_name
    else:
        form = await http_request.form()
        query = form.get("query")
        if not query:
            raise HTTPException(status_code=422, detail="Missing required field: query")

        chat_history = form.get("chat_history")
        if chat_history:
            try:
                chat_history_list = json.loads(chat_history)
                if not isinstance(chat_history_list, list):
                    chat_history_list = []
            except Exception:
                chat_history_list = []

        model_provider = form.get("model_provider") or "groq"
        session_id = form.get("session_id") or str(uuid.uuid4())
        image_mode = form.get("image_mode") or "auto"
        reranker_profile = form.get("reranker_profile") or "auto"
        reranker_model_name = form.get("reranker_model_name") or None
        stream_val = form.get("stream")
        if isinstance(stream_val, str):
            stream = stream_val.lower() in ["true", "1", "yes", "y"]
        else:
            stream = bool(stream_val)
        files = form.getlist("files")

    start = time.perf_counter()
    client_id = x_tenant_id or (http_request.client.host if http_request.client else "anonymous")
    _rate_limiter.consume(client_id)

    try:
        extra_collections = []
        router = _get_multimodal_router()

        # If files were uploaded, ingest and attach ephemeral session collection
        if files:
            max_mb = int(os.getenv("MAX_UPLOAD_MB", "20"))
            file_payloads = []
            for f in files:
                file_bytes = await f.read()
                if len(file_bytes) > max_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"File too large. Max allowed: {max_mb} MB.")
                file_payloads.append((f.filename, file_bytes))

            ingest_info = router.ingest_files_for_session(
                question=query,
                files=file_payloads,
                session_id=session_id,
                image_mode=image_mode,
            )
            extra_collections = [ingest_info["collection_name"]]
        else:
            # No files this turn; reuse prior session collection if available
            if session_id:
                try:
                    collection_name = router.session_vectors.get_session_collection(session_id)
                    if collection_name:
                        extra_collections = [collection_name]
                except Exception:
                    pass

        orchestrator = _get_orchestrator()
        result = await orchestrator.invoke(
            query,
            chat_history_list,
            session_id=session_id,
            tenant_id=x_tenant_id,
            model_provider=model_provider,
            extra_collections=extra_collections,
            reranker_profile=reranker_profile,
            reranker_model_name=reranker_model_name,
        )

        response = ChatResponse(
            session_id=session_id,
            answer=result.get("answer", "No answer generated."),
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            verifier_verdict=result.get("verifier_verdict", "UNVERIFIED"),
            is_hallucinated=result.get("is_hallucinated", False),
            optimizations=result.get("optimizations", {}),
            chat_history=result.get("chat_history", []),
            latency_optimizations=result.get("latency_optimizations", {}),
        )

        elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
        telemetry.emit(
            TelemetryLogRecord(
                timestamp=ObservabilityLayer.get_timestamp(),
                session_id=session_id,
                user_id=x_user_id or "anonymous_user",
                query=query,
                intent_detected=result.get("intent", "unknown"),
                routed_agent=result.get("optimizations", {}).get("agent_routed", "unknown"),
                latency_ms=elapsed_ms,
                llm_time_ms=float(result.get("latency_optimizations", {}).get("llm_time_ms", 0.0)),
                retrieval_time_ms=float(result.get("latency_optimizations", {}).get("retrieval_time_ms", 0.0)),
                rerank_time_ms=float(result.get("latency_optimizations", {}).get("rerank_time_ms", 0.0)),
                verifier_score=float(result.get("confidence", 0.0)),
                hallucination_score=bool(result.get("is_hallucinated", False)),
                hardware_used="gpu" if HardwareProbe.detect_environment().get("primary_device") in ["cuda", "mps"] else "cpu",
                complexity_score=float(result.get("optimizations", {}).get("complexity_score", 0.0)),
                metadata_filters_applied=result.get("optimizations", {}).get("metadata_filters", {}),
                reward_score=float(result.get("optimizations", {}).get("reward_score", 0.0)),
                tokens_input=int(result.get("optimizations", {}).get("tokens_input", 0)),
                tokens_output=int(result.get("optimizations", {}).get("tokens_output", 0)),
                temperature_used=float(result.get("optimizations", {}).get("temperature_used", 0.0)),
                answer_preview=(result.get("answer", "") or "")[:500],
            )
        )

        if stream:
            async def event_gen():
                for chunk in _chunk_text_for_stream(response.answer):
                    yield f"data: {chunk}\n\n"
                meta = {
                    "session_id": response.session_id,
                    "sources": response.sources,
                    "confidence": response.confidence,
                    "verifier_verdict": response.verifier_verdict,
                    "is_hallucinated": response.is_hallucinated,
                    "optimizations": response.optimizations,
                    "latency_optimizations": response.latency_optimizations,
                }
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

            return StreamingResponse(event_gen(), media_type="text/event-stream")

        return response
    except Exception as e:
        logger.error(f"Chat Execution Failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Generation Error.")


# -----------------------------------------------------------------------------
# 2. File Ingestion Endpoint
# -----------------------------------------------------------------------------
@router.post(
    "/ingest/files",
    response_model=IngestionResponse,
    tags=["Ingestion"],
    summary="Ingest Files",
    description="Upload PDF/DOCX/TXT files and enqueue ingestion into the vector store."
)
async def ingest_files_endpoint(
    http_request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Select one or more PDF, DOCX, or TXT documents to process."),
    mode: Literal["append", "overwrite"] = Form(
        "append",
        description="Select 'append' to merge extracted documents, or 'overwrite' to reset vector DB before ingestion.",
    ),
    x_tenant_id: Optional[str] = Header(default=None, alias="x-tenant-id"),
):
    """Upload documents and dispatch asynchronous ingestion."""
    try:
        client_id = x_tenant_id or (http_request.client.host if http_request.client else "anonymous")
        _rate_limiter.consume(client_id, cost=2)
        job_id = str(uuid.uuid4())

        ingestion_jobs[job_id] = JobTracker(job_id, {
            "status": "pending",
            "chunks_added": 0,
            "total_chunks": 0,
            "job_id": job_id,
            "logs": ["Job queued for file processing..."],
        })

        save_dir = os.path.join("data", "uploaded_docs")
        os.makedirs(save_dir, exist_ok=True)

        for uploaded_file in files:
            temp_path = os.path.join(save_dir, uploaded_file.filename)
            with open(temp_path, "wb") as f:
                while True:
                    chunk = await uploaded_file.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

        file_paths = [os.path.join(save_dir, f.filename) for f in files]

        # Prefer Celery+Redis when enabled
        celery_enabled = os.getenv("CELERY_ENABLED", "false").lower() == "true"
        if celery_enabled and os.getenv("CELERY_BROKER_URL"):
            try:
                from app.infra.celery_tasks import run_ingestion_files
                run_ingestion_files.delay(
                    job_id=job_id,
                    file_paths=file_paths,
                    metadatas=[{} for _ in files],
                    reset_db=(mode == "overwrite"),
                    tenant_id=x_tenant_id,
                )
            except Exception as e:
                logger.error(f"Ingestion Queue Error (Celery fallback to local): {e}")
                celery_enabled = False

        if not celery_enabled:
            from app.ingestion.pipeline import IngestionPipeline

            pipeline = IngestionPipeline(tenant_id=x_tenant_id)
            background_tasks.add_task(
                pipeline.run_ingestion,
                file_paths=file_paths,
                metadatas=[{} for _ in files],
                reset_db=(mode == "overwrite"),
                job_tracker=ingestion_jobs[job_id],
            )

        return IngestionResponse(
            status="accepted",
            message=f"Queued {len(files)} files for vector extraction.",
            job_id=job_id,
        )
    except Exception as e:
        logger.error(f"Ingestion Queue Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize upload sequence.")


def _run_crawler_background(url, max_depth, save_folder, mode, job_id, tenant_id=None):
    """Executes Playwright web scraping and vector encoding in the background."""

    def log_trace(msg):
        print(msg)
        if job_id in ingestion_jobs:
            ingestion_jobs[job_id]["logs"].append(msg)

    log_trace(f"\n[BACKGROUND THREAD] _run_crawler_background initiated for job {job_id} / depth {max_depth}")
    try:
        import sys
        import asyncio
        from urllib.parse import urlparse

        if sys.platform == "win32":
            log_trace("[BACKGROUND THREAD] Applying WindowsProactorEventLoopPolicy...")
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        from app.ingestion.crawler_service import CrawlerService
        from app.ingestion.pipeline import IngestionPipeline

        crawler = CrawlerService()
        pipeline = IngestionPipeline(tenant_id=tenant_id)
        reset_db = mode == "overwrite"
        is_first_batch = [True]

        async def process_live_batch(batch_items):
            log_trace(f"[BACKGROUND THREAD] Streaming {len(batch_items)} scraped pages to Vector Engine...")
            ingestion_jobs[job_id]["status"] = "crawling_and_extracting"

            import hashlib

            paths = []
            metas = []
            target_domain = urlparse(url).netloc
            domain_folder = os.path.join(save_folder, target_domain)
            os.makedirs(domain_folder, exist_ok=True)

            for item in batch_items:
                current_url = item[1]
                title = item[2]
                content = item[3]

                safe_name = hashlib.md5(current_url.encode()).hexdigest()
                path = os.path.join(domain_folder, safe_name + ".txt")

                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"== {title} ==\\n{content}")

                paths.append(path)
                metas.append(
                    {
                        "type": "url",
                        "source_url": current_url,
                        "source_domain": urlparse(current_url).netloc,
                        "document_type": "webpage",
                    }
                )

            reset = reset_db if is_first_batch[0] else False
            is_first_batch[0] = False

            def ingest_sync():
                pipeline.run_ingestion(
                    paths,
                    metadatas=metas,
                    reset_db=reset,
                    job_tracker=ingestion_jobs[job_id],
                    mark_completed=False,
                )

            await asyncio.to_thread(ingest_sync)

        target_domain = urlparse(url).netloc
        domain_folder = os.path.join(save_folder, target_domain)

        result = loop.run_until_complete(
            crawler.crawl_url(
                url=url,
                save_folder=domain_folder,
                simulate=False,
                recursive=(max_depth > 1),
                max_depth=max_depth,
                on_batch_extracted=process_live_batch,
            )
        )
        loop.close()

        if result.get("saved_files") or not is_first_batch[0]:
            ingestion_jobs[job_id]["status"] = "completed"
            ingestion_jobs[job_id]["logs"].append("Pipeline formal completion.")
        else:
            ingestion_jobs[job_id]["status"] = "failed"
            ingestion_jobs[job_id]["error"] = "No unstructured text output generated by crawler."

    except Exception as e:
        import traceback

        trace = traceback.format_exc()
        logger.error(f"Crawler Background Error:\n{trace}")
        ingestion_jobs[job_id]["status"] = "failed"
        ingestion_jobs[job_id]["error"] = str(e)


# -----------------------------------------------------------------------------
# 3. Crawler Ingestion Endpoint
# -----------------------------------------------------------------------------
@router.post(
    "/ingest/crawler",
    response_model=IngestionResponse,
    tags=["Ingestion"],
    summary="Ingest from Crawler",
    description="Crawl a URL and ingest content into the vector store. Uses Playwright + batch ingestion."
)
async def ingest_crawler_endpoint(
    http_request: Request,
    background_tasks: BackgroundTasks,
    url: str = Form(..., description="The root HTTPS path to extract data from."),
    max_depth: int = Form(1, description="Depth 1 = Single page. Depth 2 = linked pages."),
    mode: Literal["append", "overwrite"] = Form(
        "append", description="Select append to merge data, or overwrite to reset vector DB first."
    ),
    x_tenant_id: Optional[str] = Header(default=None, alias="x-tenant-id"),
):
    """Spawn asynchronous crawler + ingestion pipeline and return job id."""
    try:
        client_id = x_tenant_id or (http_request.client.host if http_request.client else "anonymous")
        _rate_limiter.consume(client_id, cost=2)
        job_id = str(uuid.uuid4())

        ingestion_jobs[job_id] = JobTracker(job_id, {
            "status": "pending",
            "chunks_added": 0,
            "total_chunks": 0,
            "job_id": job_id,
            "logs": [f"Scraping engine initialized for URL: {url}"],
        })

        save_folder = os.path.join("data", "crawled_docs")
        celery_enabled = os.getenv("CELERY_ENABLED", "false").lower() == "true"
        if celery_enabled and os.getenv("CELERY_BROKER_URL"):
            from app.infra.celery_tasks import run_crawler_job
            run_crawler_job.delay(
                job_id=job_id,
                url=url,
                max_depth=max_depth,
                save_folder=save_folder,
                mode=mode,
                tenant_id=x_tenant_id,
            )
        else:
            background_tasks.add_task(
                _run_crawler_background,
                url=url,
                max_depth=max_depth,
                save_folder=save_folder,
                mode=mode,
                job_id=job_id,
                tenant_id=x_tenant_id,
            )

        return IngestionResponse(status="accepted", message=f"Dispatched background crawler for {url}.", job_id=job_id)
    except Exception as e:
        logger.error(f"Crawler Dispatch Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to trigger web scraper.")


# -----------------------------------------------------------------------------
# 4. Ingestion Status Endpoints
# -----------------------------------------------------------------------------
@router.get(
    "/progress/{job_id}",
    tags=["Ingestion"],
    summary="Ingestion Progress",
    description="Check ingestion or crawler job progress."
)
async def check_progress_endpoint(job_id: str):
    """Poll asynchronous ingestion/crawler job progress."""
    if job_id in ingestion_jobs:
        return ingestion_jobs[job_id]

    # Fallback to persistent store for multi-worker setups
    stored = get_ingestion_job(job_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Ingestion Job ID not found.")
    return stored


@router.get(
    "/ingest/status",
    response_model=IngestionStatusResponse,
    tags=["Ingestion"],
    summary="Ingestion Status",
    description="Return vector store stats (collection, mode, total vectors, documents)."
)
async def ingestion_status_endpoint(x_tenant_id: Optional[str] = Header(default=None, alias="x-tenant-id")):
    """Returns current vector collection mode, total vectors, and source documents."""
    try:
        from app.retrieval.vector_store import QdrantStore

        store = QdrantStore(collection_name=x_tenant_id)
        stats = store.stats()
        return IngestionStatusResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to read vector status: {e}")
        raise HTTPException(status_code=500, detail="Unable to inspect ingestion status.")


# -----------------------------------------------------------------------------
# 5. Text-to-Speech Endpoint
# -----------------------------------------------------------------------------
@router.post(
    "/tts",
    tags=["Audio"],
    summary="Text-to-Speech (Coqui)",
    description="Generate a WAV audio response from input text using Coqui TTS."
)
async def tts_endpoint(text: str = Form(..., description="Text to synthesize into speech.")):
    """Generate a WAV file from text using local Coqui TTS."""
    try:
        from app.multimodal.tts import TextToSpeech

        engine = TextToSpeech()
        audio_path = engine.generate_audio(text)
        return FileResponse(audio_path, media_type="audio/wav", filename="speech.wav")
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail="TTS generation failed.")


# -----------------------------------------------------------------------------
# 6. Multimodal Audio Transcription Endpoint
# -----------------------------------------------------------------------------
class TranscriptionResponse(BaseModel):
    transcript: str

    model_config = {
        "json_schema_extra": {
            "example": {"transcript": "Hello, this is a test."}
        }
    }


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    tags=["Audio"],
    summary="Audio Transcription (Experimental)",
    description="Audio transcription endpoint (disabled by default)."
)
async def transcribe_audio_endpoint(
    http_request: Request,
    audio_file: UploadFile = File(..., description="WAV/MP3/M4A/WebM audio stream")
):
    """Proxy audio transcription via Groq Whisper."""
    try:
        client_id = http_request.client.host if http_request.client else "anonymous"
        _rate_limiter.consume(client_id)
        if os.getenv("ENABLE_TRANSCRIBE", "false").lower() != "true":
            raise HTTPException(status_code=501, detail="Audio transcription endpoint is disabled. Set ENABLE_TRANSCRIBE=true to enable.")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Missing GROQ API KEY for whisper transcription.")

        import aiohttp

        form = aiohttp.FormData()
        file_bytes = await audio_file.read()
        form.add_field("file", file_bytes, filename=audio_file.filename, content_type=audio_file.content_type)
        form.add_field("model", "whisper-large-v3-turbo")
        form.add_field("response_format", "json")

        headers = {"Authorization": f"Bearer {api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                data=form,
                timeout=25,
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return TranscriptionResponse(transcript=result.get("text", ""))

    except Exception as e:
        logger.error(f"STT Whisper Proxy Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Audio transcription failed.")


# -----------------------------------------------------------------------------
# 7. RLHF Telemetry Endpoint
# -----------------------------------------------------------------------------
@router.post(
    "/feedback",
    tags=["Feedback"],
    summary="Feedback",
    description="Record user feedback (thumbs up/down + optional text)."
)
async def rlhf_feedback_endpoint(request: FeedbackRequest):
    """Receive frontend feedback (thumbs up/down) and store asynchronously."""
    try:
        from app.rlhf.feedback_store import FeedbackStore

        FeedbackStore().record_feedback(
            session_id=request.session_id,
            rating=request.rating,
            feedback_text=request.feedback_text,
            metadata=request.metadata,
        )
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"Failed to record RLHF telemetry: {str(e)}")
        return {"status": "error", "message": "Feedback dropped."}
