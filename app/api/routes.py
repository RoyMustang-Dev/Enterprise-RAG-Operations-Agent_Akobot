"""
Vertical Slice API Gateway

This module exclusively handles HTTP routing, payload validation, and HTTP-level exception handling.
It delegates all complex business logic down to inner slices (`app.supervisor`, `app.ingestion`).
"""
import os
import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Literal

from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form
from pydantic import BaseModel, Field

from app.core.telemetry import ObservabilityLayer
from app.core.types import TelemetryLogRecord


# -----------------------------------------------------------------------------
# Global Ingestion Job Tracker Memory Node
# -----------------------------------------------------------------------------
ingestion_jobs: Dict[str, Dict[str, Any]] = {}

router = APIRouter()
logger = logging.getLogger(__name__)
telemetry = ObservabilityLayer()

# Lazy singleton orchestrator to avoid recreating heavy components on each request.
_CHAT_ORCHESTRATOR = None


def _get_orchestrator():
    global _CHAT_ORCHESTRATOR
    if _CHAT_ORCHESTRATOR is None:
        from app.supervisor.router import ExecutionGraph

        _CHAT_ORCHESTRATOR = ExecutionGraph()
    return _CHAT_ORCHESTRATOR


# -----------------------------------------------------------------------------
# Pydantic Schemas (FastAPI Inbound/Outbound Contracts)
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's raw prompt.")
    chat_history: Optional[List[Dict[str, Any]]] = Field(default=[], description="Previous conversational turns.")
    model_provider: Literal["groq", "openai", "anthropic", "gemini"] = Field(
        default="groq",
        description="Requested provider. Non-groq values are accepted but currently routed through the Groq-backed stack.",
    )
    session_id: Optional[str] = Field(default=None, description="Optional client session identifier for telemetry correlation.")


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    verifier_verdict: str
    is_hallucinated: bool
    optimizations: Dict[str, Any]
    chat_history: Optional[List[Dict[str, Any]]] = Field(default=[])
    latency_optimizations: Optional[Dict[str, Any]] = Field(default={})


class IngestionResponse(BaseModel):
    status: str
    message: str
    job_id: str


class IngestionStatusResponse(BaseModel):
    collection: str
    mode: Literal["local", "cloud"]
    total_vectors: int
    documents: List[str]


class FeedbackRequest(BaseModel):
    session_id: str
    rating: str
    feedback_text: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# 1. Chat Generation Endpoint
# -----------------------------------------------------------------------------
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    **Primary RAG Generation Interface**

    Accepts a user query, triggers supervisor routing, executes grounded generation,
    runs independent verification, and returns structured response metadata.
    """
    session_id = request.session_id or str(uuid.uuid4())
    start = time.perf_counter()

    try:
        orchestrator = _get_orchestrator()
        result = await orchestrator.invoke(
            request.query,
            request.chat_history,
            session_id=session_id,
            model_provider=request.model_provider,
        )

        response = ChatResponse(
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
                query=request.query,
                intent_detected=result.get("intent", "unknown"),
                routed_agent=result.get("optimizations", {}).get("agent_routed", "unknown"),
                latency_ms=elapsed_ms,
                llm_time_ms=float(result.get("latency_optimizations", {}).get("llm_time_ms", 0.0)),
                retrieval_time_ms=float(result.get("latency_optimizations", {}).get("retrieval_time_ms", 0.0)),
                rerank_time_ms=float(result.get("latency_optimizations", {}).get("rerank_time_ms", 0.0)),
                verifier_score=float(result.get("confidence", 0.0)),
                hallucination_score=bool(result.get("is_hallucinated", False)),
                hardware_used="gpu" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
            )
        )

        return response
    except Exception as e:
        logger.error(f"Chat Execution Failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Generation Error.")


# -----------------------------------------------------------------------------
# 2. File Ingestion Endpoint
# -----------------------------------------------------------------------------
@router.post("/ingest/files", response_model=IngestionResponse)
async def ingest_files_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Select one or more PDF, DOCX, or TXT documents to process."),
    mode: Literal["append", "overwrite"] = Form(
        "append",
        description="Select 'append' to merge extracted documents, or 'overwrite' to reset vector DB before ingestion.",
    ),
):
    """Upload documents and dispatch asynchronous ingestion."""
    try:
        job_id = str(uuid.uuid4())

        ingestion_jobs[job_id] = {
            "status": "pending",
            "chunks_added": 0,
            "total_chunks": 0,
            "job_id": job_id,
            "logs": ["Job queued for file processing..."],
        }

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

        from app.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline()

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


def _run_crawler_background(url, max_depth, save_folder, mode, job_id):
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
        pipeline = IngestionPipeline()
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
@router.post("/ingest/crawler", response_model=IngestionResponse)
async def ingest_crawler_endpoint(
    background_tasks: BackgroundTasks,
    url: str = Form(..., description="The root HTTPS path to extract data from."),
    max_depth: int = Form(1, description="Depth 1 = Single page. Depth 2 = linked pages."),
    mode: Literal["append", "overwrite"] = Form(
        "append", description="Select append to merge data, or overwrite to reset vector DB first."
    ),
):
    """Spawn asynchronous crawler + ingestion pipeline and return job id."""
    try:
        job_id = str(uuid.uuid4())

        ingestion_jobs[job_id] = {
            "status": "pending",
            "chunks_added": 0,
            "total_chunks": 0,
            "job_id": job_id,
            "logs": [f"Scraping engine initialized for URL: {url}"],
        }

        save_folder = os.path.join("data", "crawled_docs")
        background_tasks.add_task(
            _run_crawler_background,
            url=url,
            max_depth=max_depth,
            save_folder=save_folder,
            mode=mode,
            job_id=job_id,
        )

        return IngestionResponse(status="accepted", message=f"Dispatched background crawler for {url}.", job_id=job_id)
    except Exception as e:
        logger.error(f"Crawler Dispatch Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to trigger web scraper.")


# -----------------------------------------------------------------------------
# 4. Ingestion Status Endpoints
# -----------------------------------------------------------------------------
@router.get("/progress/{job_id}")
async def check_progress_endpoint(job_id: str):
    """Poll asynchronous ingestion/crawler job progress."""
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail="Ingestion Job ID not found.")

    job = ingestion_jobs[job_id]
    if job.get("status") == "failed":
        raise HTTPException(status_code=500, detail=f"Job failed: {job.get('error', 'Unknown Error')}")

    return job


@router.get("/ingest/status", response_model=IngestionStatusResponse)
async def ingestion_status_endpoint():
    """Returns current vector collection mode, total vectors, and source documents."""
    try:
        from app.retrieval.vector_store import QdrantStore

        store = QdrantStore()
        stats = store.stats()
        return IngestionStatusResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to read vector status: {e}")
        raise HTTPException(status_code=500, detail="Unable to inspect ingestion status.")


# -----------------------------------------------------------------------------
# 5. Multimodal Audio Transcription Endpoint
# -----------------------------------------------------------------------------
class TranscriptionResponse(BaseModel):
    transcript: str


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_endpoint(audio_file: UploadFile = File(..., description="WAV/MP3/M4A/WebM audio stream")):
    """Proxy audio transcription via Groq Whisper."""
    try:
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
# 6. RLHF Telemetry Endpoint
# -----------------------------------------------------------------------------
@router.post("/feedback")
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
