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

from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form, Header, Request, Body
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.core.telemetry import ObservabilityLayer
from app.core.rate_limit import TokenBucketRateLimiter
from app.infra.database import init_ingestion_db, get_ingestion_job, set_current_tenant
from app.infra.job_tracker import JobTracker
from app.infra.hardware import HardwareProbe
from app.core.types import TelemetryLogRecord, AgentType


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
_MODELSLAB_KEY = os.getenv("MODELSLAB_API_KEY", "")

def _mask_secret(text: str) -> str:
    if not text:
        return text
    if _MODELSLAB_KEY and _MODELSLAB_KEY in text:
        return text.replace(_MODELSLAB_KEY, "[REDACTED]")
    return text

# -----------------------------------------------------------------------------
# Global Bootstrapper (Create New Agent)
# -----------------------------------------------------------------------------
@router.post("/agents", summary="Global Persona Bootstrapper")
async def create_new_agent(
    bot_name: str = Form(...),
    brand_details: str = Form(...),
    welcome_message: str = Form(...),
    prompt_instructions: str = Form(""),
    agent_type: AgentType = Form(AgentType.ENTERPRISE_RAG),
    company_logo: Optional[UploadFile] = File(default=None)
):
    """
    Receives frontend multipart/form-data to define the overarching System Persona.
    Saves the logo securely to `app/static/logos` and utilizes a dynamic LLM hook 
    to expand raw instructions into a hyper-detailed ReAct prompt structure before 
    injecting it directly into the `PersonaCacheManager` singleton.
    """
    try:
        logo_path = ""
        # Secure the uploaded logo if provided
        if company_logo and company_logo.filename:
            logo_filename = os.path.basename(company_logo.filename)
            # Ensure path security avoiding traversal attacks
            logo_path = os.path.join("app", "static", "logos", logo_filename.replace("..", ""))
            os.makedirs(os.path.dirname(logo_path), exist_ok=True)
            
            try:
                with open(logo_path, "wb") as f:
                    f.write(await company_logo.read())
            except Exception as e:
                logger.error(f"[API ROUTE] Failed to save logo physically: {e}")
                raise HTTPException(status_code=500, detail="Failed to save physical logo artifact.")
                
        # Trigger the Expansion Hook
        from app.prompt_engine.bootstrapper import PersonaBootstrapper
        bootstrapper = PersonaBootstrapper()
        logger.info(f"Expanding Persona for {bot_name}...")
        expanded_prompt = bootstrapper.expand_persona(bot_name=bot_name, brand_details=brand_details, raw_instructions=prompt_instructions)
        
        # Persist and Sync Cache
        success = bootstrapper.persist_agent(
            bot_name=bot_name, 
            logo_path=logo_path, 
            brand_details=brand_details, 
            welcome_message=welcome_message, 
            raw_prompt=prompt_instructions, 
            expanded_prompt=expanded_prompt,
            agent_type=agent_type.value
        )
        
        if success:
            return {"status": "success", "message": f"Agent '{bot_name}' bootstrapped and cached globally.", "logo": logo_path}
        else:
            raise HTTPException(status_code=500, detail="Database persistence failed during bootstrapper cascade.")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API ERROR] Bootstrapper fault: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    model_provider: Literal["groq", "openai", "anthropic", "gemini", "modelslab", "auto"] = Field(
        default="auto",
        description="Requested provider. Use 'auto' to prefer Modelslab/Gemini when keys are present, with Groq fallback.",
    )
    session_id: Optional[str] = Field(default=None, description="Optional client session identifier for telemetry correlation.")
    stream: Optional[bool] = Field(default=False, description="Enable server-sent events (SSE) streaming output.")
    reranker_model_name: Optional[str] = Field(
        default=None,
        description="LLM-as-a-Judge reranker model. If omitted, uses provider defaults."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Summarize Updated_Resume_DS.pdf.",
                "model_provider": "auto",
                "session_id": "session-123",
                "stream": False,
                "reranker_model_name": "llama-3.1-8b-instant"
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
    active_persona: Optional[str] = Field(default=None, description="The bootstrapped persona mapped during execution.")

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


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize into speech.")

    model_config = {
        "json_schema_extra": {
            "example": {"text": "Hello! This is your assistant speaking."}
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
                "session_id (and no files) to query previously uploaded content.\n\n"
                "Swagger test matrix:\n"
                "1) JSON chat (no files)\n"
                "2) TXT/DOCX/PDF/MD upload + query\n"
                "3) Image OCR: image_mode=ocr\n"
                "4) Image Vision: image_mode=vision\n"
                "5) Session reuse: upload once, then query with same session_id\n",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "model_provider": {
                                "type": "string",
                                "enum": ["groq", "openai", "anthropic", "gemini", "modelslab", "auto"],
                                "default": "auto",
                                "description": "Provider selection. Use auto to let the system choose the best available provider."
                            },
                            "session_id": {"type": "string"},
                            "image_mode": {
                                "type": "string",
                                "enum": ["auto", "ocr", "vision"],
                                "default": "auto",
                                "description": "How to handle image inputs."
                            },
                            "stream": {"type": "boolean"},
                            "reranker_model_name": {
                                "type": "string",
                                "enum": ["llama-3.1-8b-instant"],
                                "default": "llama-3.1-8b-instant",
                                "description": "LLM-as-a-Judge reranker model."
                            },
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "Optional docs: .csv, .tsv, .xlsx, .docx, .txt, .md, .pdf"
                            },
                            "images": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "Optional images: .jpg, .jpeg, .png"
                            }
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
    set_current_tenant(x_tenant_id)
    chat_history_list = []
    files: List[UploadFile] = []
    image_files: List[UploadFile] = []
    query = None
    model_provider = None
    session_id = None
    image_mode = "auto"
    stream = False
    reranker_model_name = None
    force_session_context = False

    content_type = (http_request.headers.get("content-type", "") or "").lower()
    if content_type.startswith("application/json"):
        try:
            raw = await http_request.json()
            parsed = ChatRequest(**raw)
            query = parsed.query
            model_provider = parsed.model_provider
            session_id = parsed.session_id or str(uuid.uuid4())
            stream = bool(parsed.stream)
            reranker_model_name = parsed.reranker_model_name
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON payload: {e}")
    elif "multipart/form-data" in content_type:
        try:
            form = await http_request.form()
            query = form.get("query")
            model_provider = form.get("model_provider")
            session_id = form.get("session_id")
            image_mode = form.get("image_mode") or "auto"
            stream = str(form.get("stream", "false")).lower() == "true"
            reranker_model_name = form.get("reranker_model_name")

            raw_files = form.getlist("files") if hasattr(form, "getlist") else []
            files = []
            for f in raw_files:
                if isinstance(f, UploadFile) and getattr(f, "filename", None):
                    files.append(f)
                elif hasattr(f, "filename") and hasattr(f, "read") and getattr(f, "filename", None):
                    files.append(f)

            raw_images = form.getlist("images") if hasattr(form, "getlist") else []
            image_files = []
            for f in raw_images:
                if isinstance(f, UploadFile) and getattr(f, "filename", None):
                    image_files.append(f)
                elif hasattr(f, "filename") and hasattr(f, "read") and getattr(f, "filename", None):
                    image_files.append(f)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid multipart payload: {e}")
    else:
        raise HTTPException(status_code=415, detail="Unsupported Content-Type. Use JSON or multipart/form-data.")

    if not query:
        raise HTTPException(status_code=422, detail="Missing required field: query")
    if isinstance(query, str) and not query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")
    logger.info(
        f"[CHAT DEBUG] Parsed request: content_type={content_type} "
        f"query_len={len(query or '')} files={len(files)} images={len(image_files)} "
        f"session_id={session_id} image_mode={image_mode}"
    )
    if files:
        logger.info(f"[CHAT DEBUG] Uploaded files: {[f.filename for f in files]}")
    if image_files:
        logger.info(f"[CHAT DEBUG] Uploaded images: {[f.filename for f in image_files]}")

    # chat_history is auto-managed server-side via session history; client input is ignored

    model_provider = (model_provider or "auto").lower()
    session_id = session_id or str(uuid.uuid4())
    image_mode = image_mode or "auto"
    reranker_model_name = reranker_model_name or None
    reranker_profile = "llm_judge"
    if not reranker_model_name:
        reranker_model_name = "llama-3.1-8b-instant"

    start = time.perf_counter()
    client_id = x_tenant_id or (http_request.client.host if http_request.client else "anonymous")
    _rate_limiter.consume(client_id)

    try:
        extra_collections = []
        router = _get_multimodal_router()
        image_payloads = []

        # Validate uploads
        allowed_doc_exts = {".csv", ".tsv", ".xlsx", ".docx", ".txt", ".md", ".pdf"}
        allowed_img_exts = {".jpg", ".jpeg", ".png"}
        # If image files were mistakenly uploaded under "files", re-route them.
        if files:
            fixed_files = []
            for f in files:
                ext = os.path.splitext(f.filename or "")[1].lower()
                if ext in allowed_img_exts:
                    image_files.append(f)
                else:
                    fixed_files.append(f)
            files = fixed_files
        for f in files:
            ext = os.path.splitext(f.filename or "")[1].lower()
            if ext not in allowed_doc_exts:
                raise HTTPException(status_code=415, detail=f"Unsupported file type for docs: {f.filename}")
        for img in image_files:
            ext = os.path.splitext(img.filename or "")[1].lower()
            if ext not in allowed_img_exts:
                raise HTTPException(status_code=415, detail=f"Unsupported image type: {img.filename}")

        # If images are provided and user asks about image, short-circuit BEFORE ingestion.
        image_keywords = ["image", "photo", "picture", "screenshot", "describe", "what's in the image", "what is in the image"]
        q_lower = (query or "").lower()
        image_only = bool(image_files) and not files and (image_mode in ["vision", "ocr"] or any(k in q_lower for k in image_keywords))
        if image_only:
            max_mb = int(os.getenv("MAX_UPLOAD_MB", "20"))
            for f in image_files:
                file_bytes = await f.read()
                if len(file_bytes) > max_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"File too large. Max allowed: {max_mb} MB.")
                image_payloads.append((f.filename, file_bytes))
            response = router.answer_images(
                question=query,
                image_files=image_payloads,
                session_id=session_id,
                image_mode=image_mode,
            )
            logger.info("[MULTIMODAL] Short-circuiting RAG for vision-only query.")
            if stream:
                async def event_gen():
                    for chunk in _chunk_text_for_stream(response.get("answer", "")):
                        yield f"data: {chunk}\n\n"
                    meta = {
                        "session_id": response.get("session_id"),
                        "sources": response.get("sources", []),
                        "confidence": response.get("confidence", 0.0),
                        "verifier_verdict": response.get("verifier_verdict", "UNVERIFIED"),
                        "is_hallucinated": response.get("is_hallucinated", False),
                        "optimizations": response.get("optimizations", {}),
                        "latency_optimizations": response.get("latency_optimizations", {}),
                    }
                    yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                return StreamingResponse(event_gen(), media_type="text/event-stream")
            return response

        # If files were uploaded, ingest and attach ephemeral session collection
        if files or image_files:
            max_mb = int(os.getenv("MAX_UPLOAD_MB", "20"))
            file_payloads = []
            for f in files + image_files:
                file_bytes = await f.read()
                if len(file_bytes) > max_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"File too large. Max allowed: {max_mb} MB.")
                file_payloads.append((f.filename, file_bytes))
                # Preserve image bytes for direct vision path
                if f in image_files:
                    image_payloads.append((f.filename, file_bytes))

            ingest_info = router.ingest_files_for_session(
                question=query,
                files=file_payloads,
                session_id=session_id,
                image_mode=image_mode,
                tenant_id=x_tenant_id,
            )
            extra_collections = [ingest_info["collection_name"]]
            logger.info(
                f"[CHAT DEBUG] Ingested files: session_id={session_id} "
                f"collection={ingest_info.get('collection_name')} chunks_added={ingest_info.get('chunks_added')} "
                f"modes={ingest_info.get('modes')}"
            )
            if ingest_info.get("chunks_added", 0) <= 0:
                raise HTTPException(status_code=400, detail="Uploaded files/images produced no extractable content.")

        # If images are provided and user asks about image, route directly to vision/OCR response.
        if image_files and (image_mode in ["vision", "ocr"] or any(k in q_lower for k in image_keywords)):
            if not image_payloads:
                for f in image_files:
                    if hasattr(f, "read"):
                        image_payloads.append((f.filename, await f.read()))
            response = router.answer_images(
                question=query,
                image_files=image_payloads,
                session_id=session_id,
                image_mode=image_mode,
            )
            # If user explicitly asked about the image, return immediately (skip RAG pipeline).
            if image_mode in ["vision", "ocr"] or any(k in q_lower for k in image_keywords):
                logger.info("[MULTIMODAL] Short-circuiting RAG for vision-only query.")
                if stream:
                    async def event_gen():
                        for chunk in _chunk_text_for_stream(response.get("answer", "")):
                            yield f"data: {chunk}\n\n"
                        meta = {
                            "session_id": response.get("session_id"),
                            "sources": response.get("sources", []),
                            "confidence": response.get("confidence", 0.0),
                            "verifier_verdict": response.get("verifier_verdict", "UNVERIFIED"),
                            "is_hallucinated": response.get("is_hallucinated", False),
                            "optimizations": response.get("optimizations", {}),
                            "latency_optimizations": response.get("latency_optimizations", {}),
                        }
                        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                    return StreamingResponse(event_gen(), media_type="text/event-stream")
                return response
            if stream:
                async def event_gen():
                    for chunk in _chunk_text_for_stream(response.get("answer", "")):
                        yield f"data: {chunk}\n\n"
                    meta = {
                        "session_id": response.get("session_id"),
                        "sources": response.get("sources", []),
                        "confidence": response.get("confidence", 0.0),
                        "verifier_verdict": response.get("verifier_verdict", "UNVERIFIED"),
                        "is_hallucinated": response.get("is_hallucinated", False),
                        "optimizations": response.get("optimizations", {}),
                        "latency_optimizations": response.get("latency_optimizations", {}),
                    }
                    yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
                return StreamingResponse(event_gen(), media_type="text/event-stream")
            return response
        else:
            # No files this turn; reuse prior session collection if available
            if session_id:
                try:
                    collection_name = router.session_vectors.get_session_collection(session_id, tenant_id=x_tenant_id)
                    if collection_name:
                        extra_collections = [collection_name]
                        file_keywords = [
                            "file", "files", "document", "documents", "pdf", "excel", "xlsx", "csv", "tsv",
                            "spreadsheet", "sheet", "docx", "attached", "upload", "image", "photo", "picture"
                        ]
                        q_lower = (query or "").lower()
                        if any(k in q_lower for k in file_keywords):
                            force_session_context = True
                        logger.info(
                            f"[CHAT DEBUG] Reusing session collection: session_id={session_id} "
                            f"collection={collection_name}"
                        )
                except Exception:
                    pass

        # Note: We do NOT block normal RAG when no files are attached.
        # If files/images are present, we route through multimodal flows above.

        orchestrator = _get_orchestrator()

        if stream:
            # Provider-native streaming path
            import asyncio
            token_queue: asyncio.Queue[str] = asyncio.Queue()
            done_event = asyncio.Event()
            result_holder: Dict[str, Any] = {}
            streamed_any = {"value": False}

            def _on_token(tok: str):
                if tok:
                    streamed_any["value"] = True
                    try:
                        token_queue.put_nowait(tok)
                    except Exception:
                        pass

            async def _run_orchestrator():
                try:
                    result = await orchestrator.invoke(
                        query,
                        chat_history_list,
                        session_id=session_id,
                        tenant_id=x_tenant_id,
                        model_provider=model_provider,
                        extra_collections=extra_collections,
                        reranker_profile=reranker_profile,
                        reranker_model_name=reranker_model_name,
                        force_session_context=force_session_context,
                        streaming_callback=_on_token,
                    )
                    result_holder["result"] = result
                    elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
                    try:
                        telemetry.emit(
                            TelemetryLogRecord(
                                timestamp=ObservabilityLayer.get_timestamp(),
                                session_id=session_id,
                                user_id=x_user_id or "anonymous_user",
                                query=query,
                                intent_detected=(result.get("intent") or "unknown"),
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
                                tokens_input=int(result.get("optimizations", {}).get("tokens_input", 0) or 0),
                                tokens_output=int(result.get("optimizations", {}).get("tokens_output", 0) or 0),
                                temperature_used=float(result.get("optimizations", {}).get("temperature_used", 0.0) or 0.0),
                                answer_preview=(result.get("answer", "") or "")[:500],
                                model_provider=result.get("optimizations", {}).get("model_provider"),
                                model_selected=result.get("latency_optimizations", {}).get("model_selected"),
                                reranker_model_name=result.get("optimizations", {}).get("reranker_model_name"),
                                retrieval_scope=result.get("optimizations", {}).get("retrieval_scope"),
                                active_persona=result.get("active_persona"),
                                stream_enabled=True,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"[TELEMETRY] Failed to emit log record: {e}")
                except Exception as e:
                    result_holder["error"] = str(e)
                finally:
                    done_event.set()

            asyncio.create_task(_run_orchestrator())

            async def event_gen():
                buffer = ""
                last_flush = time.perf_counter()
                while True:
                    if done_event.is_set() and token_queue.empty():
                        break
                    try:
                        tok = await asyncio.wait_for(token_queue.get(), timeout=0.2)
                        if tok:
                            buffer += tok
                            now = time.perf_counter()
                            if len(buffer) >= 120 or (now - last_flush) > 0.4:
                                yield f"data: {buffer}\n\n"
                                buffer = ""
                                last_flush = now
                    except asyncio.TimeoutError:
                        # Periodically flush partial buffer to avoid long pauses.
                        if buffer and (time.perf_counter() - last_flush) > 0.6:
                            yield f"data: {buffer}\n\n"
                            buffer = ""
                            last_flush = time.perf_counter()
                        continue

                if buffer:
                    yield f"data: {buffer}\n\n"

                result = result_holder.get("result") or {}
                if not streamed_any["value"]:
                    for chunk in _chunk_text_for_stream(result.get("answer", "")):
                        yield f"data: {chunk}\n\n"

                meta = {
                    "session_id": session_id,
                    "sources": result.get("sources", []),
                    "confidence": result.get("confidence", 0.0),
                    "verifier_verdict": result.get("verifier_verdict", "UNVERIFIED"),
                    "is_hallucinated": result.get("is_hallucinated", False),
                    "optimizations": result.get("optimizations", {}),
                    "latency_optimizations": result.get("latency_optimizations", {}),
                    "active_persona": result.get("active_persona", None),
                }
                yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

            return StreamingResponse(event_gen(), media_type="text/event-stream")

        result = await orchestrator.invoke(
            query,
            chat_history_list,
            session_id=session_id,
            tenant_id=x_tenant_id,
            model_provider=model_provider,
            extra_collections=extra_collections,
            reranker_profile=reranker_profile,
            reranker_model_name=reranker_model_name,
            force_session_context=force_session_context,
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
            active_persona=result.get("active_persona", None)
        )

        elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
        try:
            telemetry.emit(
                TelemetryLogRecord(
                    timestamp=ObservabilityLayer.get_timestamp(),
                    session_id=session_id,
                    user_id=x_user_id or "anonymous_user",
                    query=query,
                    intent_detected=(result.get("intent") or "unknown"),
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
                    tokens_input=int(result.get("optimizations", {}).get("tokens_input", 0) or 0),
                    tokens_output=int(result.get("optimizations", {}).get("tokens_output", 0) or 0),
                    temperature_used=float(result.get("optimizations", {}).get("temperature_used", 0.0) or 0.0),
                    answer_preview=(result.get("answer", "") or "")[:500],
                    model_provider=result.get("optimizations", {}).get("model_provider"),
                    model_selected=result.get("latency_optimizations", {}).get("model_selected"),
                    reranker_model_name=result.get("optimizations", {}).get("reranker_model_name"),
                    retrieval_scope=result.get("optimizations", {}).get("retrieval_scope"),
                    active_persona=result.get("active_persona"),
                    stream_enabled=False,
                )
            )
        except Exception as e:
            logger.warning(f"[TELEMETRY] Failed to emit log record: {e}")

        return response
    except HTTPException:
        raise
    except Exception as e:
        import traceback; logger.error(f"Chat Execution Failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Generation Error.")
    finally:
        set_current_tenant(None)


# -----------------------------------------------------------------------------
# 2. File Ingestion Endpoint
# -----------------------------------------------------------------------------
@router.post(
    "/ingest/files",
    response_model=IngestionResponse,
    tags=["Ingestion"],
    summary="Ingest Files",
    description="Upload PDF/DOCX/TXT/MD/CSV/TSV/XLSX files and enqueue ingestion into the vector store.",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "Select one or more documents: .csv, .tsv, .xlsx, .docx, .txt, .md, .pdf"
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["append", "overwrite"],
                                "default": "append",
                                "description": "append merges; overwrite resets the vector DB before ingestion."
                            }
                        },
                        "required": ["files"]
                    }
                }
            }
        }
    }
)
async def ingest_files_endpoint(
    http_request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Select one or more PDF/DOCX/TXT/MD/CSV/TSV/XLSX documents to process."),
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

        allowed_exts = {".csv", ".tsv", ".xlsx", ".docx", ".txt", ".md", ".pdf"}
        for uploaded_file in files:
            ext = os.path.splitext(uploaded_file.filename or "")[1].lower()
            if ext not in allowed_exts:
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {uploaded_file.filename}")
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

        crawler = CrawlerService(tenant_id=tenant_id)
        pipeline = IngestionPipeline(tenant_id=tenant_id)
        reset_db = mode == "overwrite"
        is_first_batch = [True]

        async def process_live_batch(batch_items):
            log_trace(f"[BACKGROUND THREAD] Streaming {len(batch_items)} scraped pages to Vector Engine...")
            ingestion_jobs[job_id]["status"] = "crawling_and_extracting"

            import hashlib

            def _infer_page_type(url: str) -> str:
                path = (urlparse(url).path or "").lower()
                if "/product" in path:
                    return "product"
                if "/pricing" in path:
                    return "pricing"
                if "/docs" in path or "/documentation" in path:
                    return "docs"
                if "/blog" in path:
                    return "blog"
                if "/help" in path or "/support" in path:
                    return "support"
                if "/category" in path or "/listing" in path:
                    return "listing"
                return "general"

            def _quality_score(text: str) -> float:
                if not text:
                    return 0.0
                words = [w for w in text.split() if w]
                wc = len(words)
                if wc == 0:
                    return 0.0
                unique_ratio = len(set(words)) / max(1, wc)
                length_score = min(1.0, wc / 200)
                return round((0.6 * length_score) + (0.4 * unique_ratio), 3)

            def _is_thin(text: str) -> bool:
                if not text:
                    return True
                wc = len(text.split())
                if wc < 40:
                    return True
                lowered = text.lower()
                if wc < 80 and ("cookie" in lowered and "policy" in lowered):
                    return True
                return False

            paths = []
            metas = []
            target_domain = urlparse(url).netloc
            domain_folder = os.path.join(save_folder, target_domain)
            os.makedirs(domain_folder, exist_ok=True)

            for item in batch_items:
                current_url = item[1]
                title = item[2]
                content = item[3]
                if _is_thin(content):
                    continue
                quality = _quality_score(content)

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
                        "page_type": _infer_page_type(current_url),
                        "quality_score": quality,
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
    celery_enabled = os.getenv("CELERY_ENABLED", "false").lower() == "true"
    if celery_enabled and os.getenv("CELERY_BROKER_URL"):
        stored = get_ingestion_job(job_id)
        if stored is not None:
            return stored
        if job_id in ingestion_jobs:
            return ingestion_jobs[job_id]
        raise HTTPException(status_code=404, detail="Ingestion Job ID not found.")

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
        use_multi_tenant = os.getenv("QDRANT_MULTI_TENANT", "false").lower() == "true"
        if use_multi_tenant:
            store = QdrantStore()
            stats = store.stats(tenant_id=x_tenant_id)
        else:
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
    summary="Text-to-Speech (Modelslab)",
    description="Generate a speech audio response from input text using ModelsLab TTS. "
                "Accepts JSON ({\"text\": \"...\"}) or form-data (text=...).",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/TTSRequest"}
                },
                "application/x-www-form-urlencoded": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to synthesize into speech."}
                        },
                        "required": ["text"]
                    }
                }
            }
        }
    }
)
async def tts_endpoint(
    payload: Optional[TTSRequest] = Body(default=None),
    text: Optional[str] = Form(default=None, description="Text to synthesize into speech.")
):
    """Generate a speech audio response from text using ModelsLab (fallback to local Coqui if key missing)."""
    try:
        if payload is not None:
            text = payload.text
        if not text:
            raise HTTPException(status_code=422, detail="Missing required field: text")

        modelslab_key = os.getenv("MODELSLAB_API_KEY")
        if modelslab_key:
            import aiohttp
            import tempfile

            voice_id = "mia"
            language = "american english"
            
            # Simple Hindi character detection
            if any("\u0900" <= c <= "\u097F" for c in text):
                voice_id = "tara"
                language = "hindi"

            import asyncio
            import re

            # Chunk text to avoid timeouts and limits. Max ~450 chars per chunk.
            def _chunk_text(t: str, max_len=450) -> list[str]:
                chunks = []
                sentences = re.split(r'(?<=[.!?]) +', t)
                curr = ""
                for s in sentences:
                    if len(curr) + len(s) < max_len:
                        curr += s + " "
                    else:
                        if curr.strip(): chunks.append(curr.strip())
                        curr = s + " "
                if curr.strip(): chunks.append(curr.strip())
                return chunks or [t[:max_len]]

            text_chunks = _chunk_text(text)

            async def _fetch_chunk(session, c_text: str):
                payload = {
                    "key": modelslab_key,
                    "prompt": c_text,
                    "language": language,
                    "voice_id": voice_id,
                    "speed": "0.8",
                    "emotion": True,
                }
                async with session.post(
                    "https://modelslab.com/api/v6/voice/text_to_speech",
                    json=payload,
                    timeout=60,
                ) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    if result.get("status") == "error":
                        raise HTTPException(status_code=502, detail=_mask_secret(f"ModelsLab TTS Error: {result.get('message', result)}"))

                audio_url = None
                for k in ("output", "proxy_links", "future_links", "links"):
                    if isinstance(result.get(k), list) and result[k]:
                        audio_url = result[k][0]
                        break
                    if isinstance(result.get(k), dict):
                        for dk in ("audio_url", "url", "link"):
                            if result[k].get(dk):
                                audio_url = result[k].get(dk)
                                break
                if not audio_url and isinstance(result.get("audio_url"), str):
                    audio_url = result.get("audio_url")
                if not audio_url and isinstance(result.get("audio"), str):
                    audio_url = result.get("audio")
                if not audio_url and isinstance(result.get("fetch_result"), str):
                    audio_url = result.get("fetch_result")
                if not audio_url:
                    raise HTTPException(status_code=502, detail=_mask_secret(f"Modelslab TTS did not return an audio URL. Raw: {result}"))

                # Some URLs are not immediately ready; retry on 404/5xx.
                backoff = 3
                last_err = None
                for attempt in range(12):
                    try:
                        async with session.get(audio_url, timeout=60) as audio_resp:
                            if audio_resp.status == 404:
                                last_err = f"404 for {audio_url}"
                                await asyncio.sleep(backoff)
                                backoff = min(backoff * 1.5, 12)
                                continue
                            audio_resp.raise_for_status()
                            return await audio_resp.read(), audio_url
                    except Exception as e:
                        last_err = str(e)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 1.5, 12)
                raise HTTPException(status_code=502, detail=_mask_secret(f"Modelslab TTS audio fetch failed: {last_err}"))

            async with aiohttp.ClientSession() as session:
                tasks = [_fetch_chunk(session, c) for c in text_chunks]
                results = await asyncio.gather(*tasks)

            combined_audio_bytes = b""
            suffix = ".mp3"
            for audio_bytes, a_url in results:
                combined_audio_bytes += audio_bytes
                if a_url.endswith(".wav"): suffix = ".wav"

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(combined_audio_bytes)
            tmp.close()
            media_type = "audio/mpeg" if suffix == ".mp3" else "audio/wav"
            return FileResponse(tmp.name, media_type=media_type, filename=f"speech{suffix}")

        # Fallback to local Coqui if ModelsLab key is missing
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
    summary="Audio Transcription (Modelslab)",
    description="Audio transcription endpoint using ModelsLab STT. Accepts WAV/MP3 files and returns text."
)
async def transcribe_audio_endpoint(
    http_request: Request,
    audio_file: UploadFile = File(..., description="WAV or MP3 audio stream")
):
    """Audio transcription via ModelsLab STT (fallback to local Whisper if key missing)."""
    try:
        client_id = http_request.client.host if http_request.client else "anonymous"
        _rate_limiter.consume(client_id)
        if os.getenv("ENABLE_TRANSCRIBE", "true").lower() != "true":
            raise HTTPException(status_code=501, detail="Audio transcription endpoint is disabled. Set ENABLE_TRANSCRIBE=true to enable.")

        filename = audio_file.filename or ""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".wav", ".mp3"]:
            raise HTTPException(status_code=415, detail="Unsupported file type. Please upload WAV or MP3.")

        file_bytes = await audio_file.read()
        modelslab_key = os.getenv("MODELSLAB_API_KEY")
        if modelslab_key:
            try:
                import aiohttp
                import base64
                import asyncio

                # 1) Upload audio as base64 to get a URL
                mime = "audio/wav" if ext == ".wav" else "audio/mpeg"
                b64 = base64.b64encode(file_bytes).decode("utf-8")
                init_audio = f"data:{mime};base64,{b64}"
                upload_payload = {"key": modelslab_key, "init_audio": init_audio}

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://modelslab.com/api/v6/voice/base64_to_url",
                        json=upload_payload,
                        timeout=60,
                    ) as resp:
                        resp.raise_for_status()
                        upload_result = await resp.json()
                        if upload_result.get("status") == "error":
                            raise HTTPException(status_code=502, detail=_mask_secret(f"ModelsLab STT Upload Error: {upload_result.get('message', upload_result)}"))

                audio_url = None
                if isinstance(upload_result.get("output"), list) and upload_result["output"]:
                    audio_url = upload_result["output"][0]
                if not audio_url and isinstance(upload_result.get("audio_url"), str):
                    audio_url = upload_result.get("audio_url")
                if not audio_url and isinstance(upload_result.get("fetch_result"), str):
                    audio_url = upload_result.get("fetch_result")
                if not audio_url:
                    raise HTTPException(status_code=502, detail=_mask_secret(f"Modelslab STT upload failed. Raw: {upload_result}"))

                # 2) Request STT (v6 community endpoint)
                stt_payload = {
                    "key": modelslab_key,
                    "init_audio": audio_url,
                    "language": "en",
                    "timestamp_level": None,
                    "webhook": None,
                    "track_id": None,
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://modelslab.com/api/v6/voice/speech_to_text",
                        json=stt_payload,
                        timeout=120,
                    ) as resp:
                        resp.raise_for_status()
                        stt_result = await resp.json()
                        if stt_result.get("status") == "error":
                            raise HTTPException(status_code=502, detail=_mask_secret(f"ModelsLab STT Eval Error: {stt_result.get('message', stt_result)}"))

                status = stt_result.get("status")
                if isinstance(status, str) and status.lower() in ["error", "failed"]:
                    raise HTTPException(
                        status_code=502,
                        detail=_mask_secret(f"Modelslab STT error: {stt_result.get('message', stt_result)}"),
                    )

                if status == "processing" or stt_result.get("fetch_result"):
                    fetch_url = stt_result.get("fetch_result")
                    if fetch_url:
                        backoff = 3
                        last_err = None
                        for attempt in range(15):
                            await asyncio.sleep(backoff)
                            try:
                                async with aiohttp.ClientSession() as session:
                                    async with session.post(fetch_url, json={"key": modelslab_key}, timeout=60) as fresp:
                                        fresp.raise_for_status()
                                        stt_result = await fresp.json()
                                status = stt_result.get("status")
                                logger.info(f"[STT] fetch_result poll {attempt+1}/15 status={status}")
                                if isinstance(status, str) and status.lower() in ["error", "failed"]:
                                    raise HTTPException(
                                        status_code=502,
                                        detail=_mask_secret(f"Modelslab STT error: {stt_result.get('message', stt_result)}"),
                                    )
                                if status == "success" or "output" in stt_result or "text" in stt_result:
                                    break
                            except Exception as fe:
                                last_err = fe
                                logger.warning(f"[STT] fetch_result retry failed: {fe}")
                            backoff = min(backoff * 1.5, 15)
                        if status != "success" and not stt_result.get("output") and not stt_result.get("text"):
                            raise HTTPException(status_code=502, detail=_mask_secret(f"Modelslab STT fetch failed: {last_err or 'Timeout'}"))

                transcript = None
                if isinstance(stt_result.get("output"), list) and stt_result["output"]:
                    out_val = stt_result["output"][0]
                    if isinstance(out_val, str) and out_val.startswith("http"):
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(out_val, timeout=30) as txt_resp:
                                    txt_resp.raise_for_status()
                                    transcript = await txt_resp.text()
                        except Exception as e:
                            logger.warning(f"Failed to download STT text from {out_val}: {e}")
                    else:
                        transcript = out_val
                if not transcript and isinstance(stt_result.get("output"), str):
                    transcript = stt_result["output"]

                if not transcript and isinstance(stt_result.get("text"), str):
                    transcript = stt_result["text"]

                if not transcript:
                    raise HTTPException(status_code=502, detail=_mask_secret(f"Modelslab STT did not return transcript. Raw: {stt_result}"))

                return TranscriptionResponse(transcript=transcript)
            except Exception as e:
                logger.warning(f"[STT] ModelsLab failed; falling back to local STT. Error: {e}")

        # Fallback to local Whisper if ModelsLab key is missing or failed
        backend = os.getenv("STT_BACKEND", "local").lower()
        if backend == "local":
            try:
                from app.multimodal.stt import SpeechToText
                engine = SpeechToText()
                transcript = engine.transcribe(file_bytes, filename=filename)
                return TranscriptionResponse(transcript=transcript)
            except Exception as e:
                logger.warning(f"[STT] Local Whisper failed; falling back to Groq STT. Error: {e}")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Missing GROQ API KEY for whisper transcription.")

        import aiohttp

        form = aiohttp.FormData()
        form.add_field("file", file_bytes, filename=filename, content_type=audio_file.content_type)
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

# -----------------------------------------------------------------------------
# 8. System Metrics Endpoint
# -----------------------------------------------------------------------------
class HardwareMetrics(BaseModel):
    cpu_usage_percent: float
    memory_usage_percent: float
    active_jobs: int

class SystemMetricsResponse(BaseModel):
    hardware: HardwareMetrics
    tenant_id: str
    llm_observability: Optional[Dict[str, Any]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "hardware": {
                    "cpu_usage_percent": 15.4,
                    "memory_usage_percent": 42.1,
                    "active_jobs": 2
                },
                "tenant_id": "default",
                "llm_observability": {
                    "rag_agent_metrics": { "tokens": 40200, "errors": 0 },
                    "business_analyst_metrics": { "tokens": 85030, "errors": 2 }
                }
            }
        }
    }

@router.get(
    "/metrics",
    response_model=SystemMetricsResponse,
    tags=["System"],
    summary="System Metrics Diagnostics",
    description="Returns hardware diagnostic metrics and active queue states for scaling telemetry."
)
async def system_metrics_endpoint(x_tenant_id: Optional[str] = Header(default="default", alias="x-tenant-id")):
    import psutil
    active = len([j for j in ingestion_jobs.values() if isinstance(j, dict) and j.get("status") in ["pending", "crawling_and_extracting", "running", "extracting"]])
    
    hardware = HardwareMetrics(
        cpu_usage_percent=psutil.cpu_percent(interval=0.1),
        memory_usage_percent=psutil.virtual_memory().percent,
        active_jobs=active
    )
    
    llm_metrics = {
        "rag_agent_metrics": { "tokens": 0, "errors": 0 },
        "business_analyst_metrics": { "tokens": 0, "errors": 0 }
    }
    
    # Langfuse Observability Integration via non-blocking API payload
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            # Placeholder: In a real environment, query https://us.langfuse.com/api/public/metrics
            pass
        except Exception as e:
            logger.warning(f"[LANGFUSE] Failed to fetch Langfuse metrics: {e}")

    return SystemMetricsResponse(
        hardware=hardware,
        tenant_id=x_tenant_id,
        llm_observability=llm_metrics
    )


@router.get(
    "/metrics/sla",
    tags=["System"],
    summary="SLA Metrics Export",
    description="Returns latency and error-rate aggregates from recent audit logs for SLA dashboards."
)
async def sla_metrics_endpoint(
    x_tenant_id: Optional[str] = Header(default="default", alias="x-tenant-id"),
    limit: int = 200
):
    import json
    import statistics
    from pathlib import Path

    audit_path = Path("data") / "audit" / "audit_logs.jsonl"
    if not audit_path.exists():
        raise HTTPException(status_code=404, detail="Audit log file not found.")

    rows = []
    with audit_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    rows = rows[-max(1, min(limit, 5000)):]
    latencies = [r.get("latency_ms", 0.0) for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    errors = [r for r in rows if (r.get("answer_preview") or "").lower().startswith("error") or r.get("routed_agent") == "unknown"]

    def percentile(values, pct):
        if not values:
            return None
        values = sorted(values)
        idx = max(0, int(len(values) * pct) - 1)
        return values[idx]

    by_agent = {}
    for r in rows:
        agent = r.get("routed_agent") or "unknown"
        by_agent.setdefault(agent, 0)
        by_agent[agent] += 1

    return {
        "tenant_id": x_tenant_id,
        "samples": len(rows),
        "latency_ms": {
            "avg": round(statistics.mean(latencies), 3) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "error_rate": round(len(errors) / len(rows), 4) if rows else 0.0,
        "by_agent": by_agent,
    }

# -----------------------------------------------------------------------------
# 9. Decoupled Business Analyst Agent Endpoint
# -----------------------------------------------------------------------------
@router.post(
    "/business_analyst/chat",
    tags=["Analytics"],
    summary="Decoupled Data Analytics Chat Execution",
    description="Dedicated LangGraph REPL pipeline for explicit CSV file mathematical analysis.",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "session_id": {"type": "string"},
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "Upload one or more CSV/Excel files."
                            }
                        },
                        "required": ["query", "session_id", "files"]
                    }
                }
            }
        }
    }
)
async def analytics_chat_endpoint(
    request: Request,
    query: str = Form(...),
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
    async_mode: bool = Form(False),
    background_tasks: BackgroundTasks = None,
    x_tenant_id: str = Header(default="default", alias="x-tenant-id")
):
    import pandas as pd
    import io
    import os
    import json
    
    # Securely retrieve the Persona from Cache
    from app.prompt_engine.groq_prompts.config import PersonaCacheManager
    cache = PersonaCacheManager()
    persona = cache.get_persona()

    if os.getenv("ANALYTICS_REQUIRE_TENANT", "false").lower() == "true":
        if not x_tenant_id or x_tenant_id == "default":
            raise HTTPException(status_code=400, detail="x-tenant-id is required for Business Analyst Agent.")
    
    # We strictly mandate tabular data
    if not files:
        raise HTTPException(status_code=400, detail="The Business Analyst Agent requires at least one CSV/Excel file upload to perform mathematical tasks.")
        
    dataframes = []
    sources = []
    for file in files:
        contents = await file.read()
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(contents))
            elif file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(io.BytesIO(contents))
            else:
                continue
            dataframes.append(df)
            sources.append({"source": file.filename, "rows": int(len(df))})
        except Exception as e:
            logger.error(f"[ANALYTICS] File parsing failed for {file.filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to parse {file.filename} as tabular data.")
            
    if not dataframes:
         raise HTTPException(status_code=400, detail="No readable CSV or Excel dataframes were extracted from the uploaded files.")
         
    from app.agents.data_analytics.supervisor import DataAnalyticsSupervisor
    supervisor = DataAnalyticsSupervisor(dataframes=dataframes, sources=sources)
    
    logger.info(f"[{session_id}] Executing Business Analyst LangGraph on {len(dataframes)} dataframes.")
    if async_mode:
        from uuid import uuid4
        from app.infra.database import init_analytics_jobs_db, upsert_analytics_job
        job_id = str(uuid4())
        init_analytics_jobs_db()
        upsert_analytics_job(job_id, "queued", "{}", x_tenant_id)

        async def _run_job():
            try:
                payload = await supervisor.run(query=query, persona=persona, session_id=session_id)
                upsert_analytics_job(job_id, "completed", json.dumps(payload), x_tenant_id)
            except Exception as e:
                upsert_analytics_job(job_id, "failed", json.dumps({"error": str(e)}), x_tenant_id)

        if background_tasks:
            background_tasks.add_task(_run_job)
        else:
            await _run_job()
        return {"status": "accepted", "job_id": job_id}

    dashboard_payload = await supervisor.run(query=query, persona=persona, session_id=session_id)
    
    return {
        "status": "success",
        "agent": "BUSINESS_ANALYST",
        "data": dashboard_payload
    }

# -----------------------------------------------------------------------------
# Analytics Job Status
# -----------------------------------------------------------------------------
@router.get(
    "/analytics/progress/{job_id}",
    tags=["Analytics"],
    summary="Analytics Job Status",
    description="Check the status and result of an async Business Analyst job."
)
async def analytics_job_status(job_id: str, x_tenant_id: str = Header(default="default", alias="x-tenant-id")):
    from app.infra.database import init_analytics_jobs_db, fetch_analytics_job
    init_analytics_jobs_db()
    row = fetch_analytics_job(job_id, x_tenant_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    status, payload_json, updated_at = row
    try:
        payload = json.loads(payload_json) if payload_json else {}
    except Exception:
        payload = {"raw": payload_json}
    return {"job_id": job_id, "status": status, "updated_at": updated_at, "data": payload}

# -----------------------------------------------------------------------------
# 10. File Download Route (CSV Export)
# -----------------------------------------------------------------------------
@router.get(
    "/exports/{filename}",
    tags=["Exports"],
    summary="Download Generated Analytical Reports",
    description="Serves physically compiled CSV insights securely back to the frontend dashboard."
)
async def download_export(filename: str):
    import os
    from fastapi.responses import FileResponse
    
    export_dir = os.path.join(os.getcwd(), "data", "exports")
    file_path = os.path.join(export_dir, filename)
    
    if not os.path.exists(file_path):
         raise HTTPException(status_code=404, detail="File has expired or does not exist.")
    
    media_type = "text/csv"
    if filename.endswith(".xlsx"):
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )
