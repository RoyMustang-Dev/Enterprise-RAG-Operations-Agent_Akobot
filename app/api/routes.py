"""
Vertical Slice API Gateway

This module exclusively handles HTTP routing, payload validation, and HTTP-level exception handling.
It delegates all complex business logic down to the inner slices (`app.supervisor`, `app.ingestion`).
By keeping this file logicless, we enforce the Single Responsibility Principle.
"""
import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Literal
from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form, Depends
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Global Ingestion Job Tracker Memory Node
# -----------------------------------------------------------------------------
ingestion_jobs: Dict[str, Dict[str, Any]] = {}

# We implement explicit mock dependencies here during Phase 2 scaffolding.
# These will be dynamically replaced as we build the inner slices in later Phases.

router = APIRouter()
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Pydantic Schemas (FastAPI Inbound/Outbound Contracts)
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's raw prompt.")
    chat_history: Optional[List[Dict[str, Any]]] = Field(default=[], description="Previous conversational turns.")
    model_provider: Literal["groq", "openai", "anthropic", "gemini"] = Field(
        default="groq", 
        description="The requested AI inference backend. Controls cost vs reasoning strength."
    )

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
    
    Accepts a user query, triggers the ReAct Supervisor routing, executes the grounded 
    response generation, evaluates the Sarvam verifiable assertions, and logs the execution.
    """
    try:
        from app.supervisor.router import ExecutionGraph
        orchestrator = ExecutionGraph()
        result = await orchestrator.invoke(request.query, request.chat_history)
        
        return ChatResponse(
            answer=result.get("answer", "No answer generated."),
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            verifier_verdict=result.get("verifier_verdict", "UNVERIFIED"),
            is_hallucinated=result.get("is_hallucinated", False),
            optimizations=result.get("optimizations", {}),
            chat_history=result.get("chat_history", []),
            latency_optimizations=result.get("latency_optimizations", {})
        )
    except Exception as e:
        logger.error(f"Chat Execution Failed: {str(e)}")
        # Mute raw stack traces from reaching the client per Enterprise standards
        raise HTTPException(status_code=500, detail="Internal Generation Error.")

# -----------------------------------------------------------------------------
# 2. File Ingestion Endpoint
# -----------------------------------------------------------------------------
@router.post("/ingest/files", response_model=IngestionResponse)
async def ingest_files_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Select one or more PDF, DOCX, or TXT enterprise documents to process into knowledge vectors."),
    mode: Literal["append", "overwrite"] = Form("append", description="Select 'Append' to merge extracted documents seamlessly, or 'Overwrite' to drop the Qdrant DB completely before extraction.")
):
    """
    **Enterprise Document Upload Pipeline**
    
    Accepts raw PDF/DOCX files, saves them to static storage, and dispatches them 
    to the asynchronous chunking/embedding pipeline over the BAAI engine.
    """
    try:
        job_id = str(uuid.uuid4())
        
        ingestion_jobs[job_id] = {
            "status": "pending",
            "chunks_added": 0,
            "total_chunks": 0,
            "job_id": job_id,
            "logs": ["Job queued for file processing..."]
        }
        
        # Ensure static upload directory exists physically mapping outside the app/ tree
        save_dir = os.path.join("data", "uploaded_docs")
        os.makedirs(save_dir, exist_ok=True)
        
        for uploaded_file in files:
            temp_path = os.path.join(save_dir, uploaded_file.filename)
            content = await uploaded_file.read()
            with open(temp_path, "wb") as f:
                f.write(content)
        
        # Disaggregating logic to the Ingestion pipeline natively
        file_paths = [os.path.join(save_dir, f.filename) for f in files]
        
        from app.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        
        # Trigger background pipeline execution structurally avoiding API timeouts
        background_tasks.add_task(
            pipeline.run_ingestion, 
            file_paths=file_paths, 
            metadatas=[{} for _ in files], 
            reset_db=(mode == "overwrite"),
            job_tracker=ingestion_jobs[job_id]
        )
        
        return IngestionResponse(
            status="accepted",
            message=f"Queued {len(files)} files for vector extraction.",
            job_id=job_id
        )
    except Exception as e:
        logger.error(f"Ingestion Queue Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize upload sequence.")

# -----------------------------------------------------------------------------
# 3. Crawler Ingestion Endpoint
# -----------------------------------------------------------------------------
def _run_crawler_background(url, max_depth, save_folder, mode, job_id):
    """Executes the Playwright headless web scraping and subsequent vector encoding fully in the background."""
    def log_trace(msg):
        print(msg)
        if job_id in ingestion_jobs:
            ingestion_jobs[job_id]["logs"].append(msg)
            
    log_trace(f"\\n[BACKGROUND THREAD] _run_crawler_background initiated for job {job_id} / depth {max_depth}")
    try:
        import sys
        import asyncio
        if sys.platform == "win32":
            log_trace("[BACKGROUND THREAD] Applying WindowsProactorEventLoopPolicy...")
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
        log_trace("[BACKGROUND THREAD] Creating new isolated asyncio loop...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        from app.ingestion.crawler_service import CrawlerService
        log_trace(f"[BACKGROUND THREAD] Instantiating CrawlerService for {url}...")
        crawler = CrawlerService()
        
        reset_db = True if mode == "overwrite" else False
        from app.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        is_first_batch = [True]
        
        async def process_live_batch(batch_items):
            log_trace(f"[BACKGROUND THREAD] Streaming {len(batch_items)} scraped pages natively to Vector Engine...")
            ingestion_jobs[job_id]["status"] = "crawling_and_extracting"
            
            paths = []
            metas = []
            import hashlib
            from urllib.parse import urlparse
            import json
            
            target_domain = urlparse(url).netloc
            domain_folder = os.path.join(save_folder, target_domain)
            
            if not os.path.exists(domain_folder):
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
                metas.append({"type": "url", "source_url": current_url})
                
            reset = reset_db if is_first_batch[0] else False
            is_first_batch[0] = False
            
            def ingest_sync():
                pipeline.run_ingestion(paths, metadatas=metas, reset_db=reset, job_tracker=ingestion_jobs[job_id], mark_completed=False)
                
            await asyncio.to_thread(ingest_sync)
            
        log_trace("[BACKGROUND THREAD] Blocking until Playwright crawler resolves completion...")
        from urllib.parse import urlparse
        target_domain = urlparse(url).netloc
        domain_folder = os.path.join(save_folder, target_domain)
        
        result = loop.run_until_complete(
            crawler.crawl_url(
                url=url, 
                save_folder=domain_folder,
                simulate=False,
                recursive=(max_depth > 1),
                max_depth=max_depth,
                on_batch_extracted=process_live_batch
            )
        )
        log_trace(f"[BACKGROUND THREAD] Playwright crawler successfully resolved: {result.get('status')}")
        loop.close()
        
        if result.get("saved_files") or not is_first_batch[0]:
            log_trace("[BACKGROUND THREAD] Vector Engine streaming execution formally complete!")
            ingestion_jobs[job_id]["status"] = "completed"
            ingestion_jobs[job_id]["logs"].append("Pipeline formal completion.")
        else:
             log_trace("[BACKGROUND THREAD] No unstructured text output physically extracted.")
             ingestion_jobs[job_id]["status"] = "failed"
             ingestion_jobs[job_id]["error"] = "No unstructured text output generated by Playwright crawler."

    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        logger.error(f"Crawler Background Error:\\n{trace}")
        log_trace(f"[BACKGROUND THREAD - FATAL] Target background engine fully crashed natively: {e}")
        print(trace)
        ingestion_jobs[job_id]["status"] = "failed"
        ingestion_jobs[job_id]["error"] = str(e)


@router.post("/ingest/crawler", response_model=IngestionResponse)
async def ingest_crawler_endpoint(
    background_tasks: BackgroundTasks,
    url: str = Form(..., description="The root HTTPS path to extract data from."),
    max_depth: int = Form(1, description="Depth 1 = Single page. Depth 2 = Links within page. (Beware depth 2 expands exponentially)."),
    mode: Literal["append", "overwrite"] = Form("append", description="Select 'Appendix' to merge data safely, or 'Overwrite' to drop the Qdrant DB completely before extraction.")
):
    """
    **External Dynamic Web Crawler**
    
    Spawns an asynchronous Playwright cluster natively mimicking human OS interaction.
    Dynamically bypasses bot-protections and extracts pure unformatted text logic.
    Provides real-time Background Task telemetry tracking logic to the /progress endpoint.
    """
    try:
        job_id = str(uuid.uuid4())
        
        ingestion_jobs[job_id] = {
            "status": "pending",
            "chunks_added": 0,
            "total_chunks": 0,
            "job_id": job_id,
            "logs": [f"Scraping Engine initialized for URL: {url}"]
        }
        
        save_folder = os.path.join("data", "crawled_docs")
        background_tasks.add_task(
            _run_crawler_background, 
            url=url, 
            max_depth=max_depth,
            save_folder=save_folder,
            mode=mode,
            job_id=job_id
        )
        
        return IngestionResponse(
            status="accepted",
            message=f"Dispatched background crawler for {url}.",
            job_id=job_id
        )
    except Exception as e:
        logger.error(f"Crawler Dispatch Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to trigger web scraper.")

# -----------------------------------------------------------------------------
# 4. Status Check Endpoint
# -----------------------------------------------------------------------------
@router.get("/progress/{job_id}")
async def check_progress_endpoint(job_id: str):
    """
    **Asynchronous Task Polling**
    """
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail="Ingestion Job ID not found natively.")

    job = ingestion_jobs[job_id]
    if job.get("status") == "failed":
         raise HTTPException(status_code=500, detail=f"Job failed explicitly: {job.get('error', 'Unknown Error')}")

    return job

# -----------------------------------------------------------------------------
# 5. Multimodal Audio Transcription Endpoint
# -----------------------------------------------------------------------------
class TranscriptionResponse(BaseModel):
    transcript: str

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_endpoint(
    audio_file: UploadFile = File(..., description="WAV/MP3/M4A Audio Stream")
):
    """
    **Whisper AI STT Gateway**
    
    Accepts raw binary audio buffers and natively executes hyper-fast transcription utilizing 
    the `whisper-large-v3-turbo` model over Groq API.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Missing GROQ API KEY explicitly for Whisper Transcription.")
            
        import aiohttp
        
        # Explicit MultiPart Form encoding required for binary proxying to Groq API
        form = aiohttp.FormData()
        
        # Read file dynamically
        file_bytes = await audio_file.read()
        form.add_field('file',
                       file_bytes,
                       filename=audio_file.filename,
                       content_type=audio_file.content_type)
        form.add_field('model', 'whisper-large-v3-turbo')
        form.add_field('response_format', 'json')
        
        headers = {"Authorization": f"Bearer {api_key}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                data=form,
                timeout=25
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                return TranscriptionResponse(transcript=result.get("text", ""))
                
    except Exception as e:
        logger.error(f"STT Whisper Proxy Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Audio Signal extraction natively failed.")

# -----------------------------------------------------------------------------
# 6. RLHF Telemetry Endpoint
# -----------------------------------------------------------------------------
@router.post("/feedback")
async def rlhf_feedback_endpoint(request: FeedbackRequest):
    """
    **Human-in-the-Loop Feedback Gateway**
    
    Receives frontend interactions (Thumbs up/down) and pipes them into the
    secure RLHF Audit Log for future AI self-correction cycles.
    """
    try:
        from app.rlhf.feedback_store import FeedbackStore
        FeedbackStore().record_feedback(
            session_id=request.session_id,
            rating=request.rating,
            feedback_text=request.feedback_text,
            metadata=request.metadata
        )
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"Failed to record RLHF telemetry: {str(e)}")
        # Do not throw 500. Feedback shouldn't crash the UI if the disk fails.
        return {"status": "error", "message": "Feedback dropped."}
