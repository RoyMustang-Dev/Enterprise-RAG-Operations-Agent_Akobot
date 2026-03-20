"""
V2 API Routes — Full V1 Feature Parity

Chat endpoint supports:
  - JSON body (simple queries)
  - multipart/form-data (file uploads + images + query in one request)
  - image_mode: auto | ocr | vision
  - stream: SSE token streaming
  - Session reuse: same session_id retrieves prior PageIndex tree

Response schema mirrors V1 ChatResponse exactly:
  verifier_verdict, is_hallucinated, active_persona, email_action, chat_history
"""
import os
import uuid
import time
import json
import logging
from typing import List, Dict, Any, Optional, Literal
import re

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.telemetry import ObservabilityLayer
from app.core.rate_limit import TokenBucketRateLimiter
from app.infra.database import (
    set_current_tenant,
    get_chat_history,
    save_chat_turn,
    get_session_cache,
    upsert_session_cache,
)

_rate_limiter = TokenBucketRateLimiter()
router = APIRouter()
logger = logging.getLogger(__name__)
telemetry = ObservabilityLayer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_upload_file(obj: Any) -> bool:
    return bool(obj) and hasattr(obj, "filename") and hasattr(obj, "read")


# ---------------------------------------------------------------------------
# Pydantic Schemas (Full V1 Parity)
# ---------------------------------------------------------------------------
class ChatRequestV2(BaseModel):
    query: str = Field(..., description="The user's raw prompt.")
    model_provider: Literal["groq", "openai", "modelslab", "auto"] = Field(
        default="auto",
        description="Provider selection. 'auto' uses ModelsLab (paid) with Groq fallback.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Client session identifier. Reuse to query previously ingested content.",
    )
    image_mode: Optional[Literal["auto", "ocr", "vision"]] = Field(
        default="auto",
        description="How to handle image inputs: auto (smart detect), ocr (text only), vision (semantic).",
    )
    stream: Optional[bool] = Field(default=False, description="Enable SSE token streaming.")
    reranker_model_name: Optional[str] = Field(
        default=None,
        description="LLM-as-a-Judge reranker model override (V2 uses PageIndex scoring, this is advanced).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the refund policy in the uploaded documents?",
                "model_provider": "auto",
                "session_id": "uuid-from-ingest",
                "image_mode": "auto",
                "stream": False,
            }
        }
    }


class ChatResponseV2(BaseModel):
    """Full V1 parity response schema."""
    session_id: Optional[str] = None
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    verifier_verdict: str = Field(default="NOT_RUN", description="Hallucination check result.")
    is_hallucinated: bool = Field(default=False)
    optimizations: Dict[str, Any]
    chat_history: Optional[List[Dict[str, Any]]] = Field(default=[])
    latency_optimizations: Optional[Dict[str, Any]] = Field(default={})
    active_persona: Optional[str] = Field(default=None)
    email_action: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Composio action result. email_action.connect_url is present when Gmail auth is needed.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "uuid",
                "answer": "The refund policy states...",
                "sources": [{"node_id": "0003", "title": "Refund Policy", "text": "..."}],
                "confidence": 0.97,
                "verifier_verdict": "SUPPORTED",
                "is_hallucinated": False,
                "optimizations": {"active_model": "gemini-2.5-flash", "tools_used": ["check_security", "search_pageindex", "synthesize_answer", "verify_answer"]},
                "email_action": None,
            }
        }
    }


class IngestionResponseV2(BaseModel):
    status: str
    message: str
    job_id: str
    session_id: Optional[str] = None
    nodes_indexed: Optional[int] = None


class IngestionStatusV2(BaseModel):
    status: str
    job_id: str
    payload: Optional[Dict[str, Any]] = None


class PageIndexStatusV2(BaseModel):
    tenant_id: Optional[str]
    session_id: Optional[str] = None
    total_sessions: int
    total_nodes: int
    sessions: List[Dict[str, Any]]


class CrawlerRequestV2(BaseModel):
    url: str = Field(..., description="Target URL to crawl.")
    max_depth: int = Field(default=1, description="Crawl depth (1=single page, 2=linked pages, etc.).")
    session_id: Optional[str] = Field(default=None, description="Optional session_id to append crawled data into.")
    mode: Literal["append", "overwrite"] = Field(
        default="append",
        description="append: add new nodes to existing session. overwrite: clear existing nodes first.",
    )


# Lazy singleton orchestrator
_MODULAR_ORCHESTRATOR = None


def _get_v2_orchestrator():
    global _MODULAR_ORCHESTRATOR
    if _MODULAR_ORCHESTRATOR is None:
        try:
            from app.v2.agents.modular_orchestrator import ModularOrchestrator
            _MODULAR_ORCHESTRATOR = ModularOrchestrator(agent_tag="rag_v2")
        except Exception as e:
            logger.error(f"[V2 ROUTES] Failed to load ModularOrchestrator: {e}")
    return _MODULAR_ORCHESTRATOR


def _chunk_text_for_stream(text: str, size: int = 120):
    if not text:
        return []
    chunks, idx = [], 0
    while idx < len(text):
        chunks.append(text[idx:idx + size])
        idx += size
    return chunks


# ---------------------------------------------------------------------------
# 1. Chat Generation Endpoint (V2) — Full V1 parity
# ---------------------------------------------------------------------------
@router.post(
    "/chat",
    response_model=ChatResponseV2,
    tags=["V2 Chat"],
    summary="Modular Tool-Calling Chat Agent (V2) — RAG + Files + Images",
    description=(
        "V2 Modular RAG Chat. Accepts JSON or multipart/form-data.\n\n"
        "**File Upload (inline docs):** Upload files with the query; they are indexed into PageIndex for that session.\n\n"
        "**Image support:** Upload images; uses OCR (EasyOCR) or Vision (Gemini) based on image_mode.\n\n"
        "**Session reuse:** Use the same session_id from /api/v2/ingest/* to query previously indexed content.\n\n"
        "**Streaming:** Set stream=true for SSE token streaming.\n\n"
        "**Swagger test matrix:**\n"
        "1) JSON chat (no files)\n"
        "2) Multipart with PDF/DOCX/CSV + query\n"
        "3) Image OCR: image_mode=ocr\n"
        "4) Image Vision: image_mode=vision\n"
        "5) Session reuse: ingest first, then chat with same session_id\n"
        "6) Email action: 'Send an email to X about Y' → returns connect_url if Gmail not connected\n"
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "User query (required)"},
                            "model_provider": {
                                "type": "string",
                                "enum": ["groq", "openai", "modelslab", "auto"],
                                "default": "auto",
                            },
                            "session_id": {"type": "string"},
                            "image_mode": {
                                "type": "string",
                                "enum": ["auto", "ocr", "vision"],
                                "default": "auto",
                            },
                            "stream": {"type": "boolean", "default": False},
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "Docs: .pdf, .docx, .txt, .md, .csv, .tsv, .xlsx",
                            },
                            "images": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "Images: .jpg, .jpeg, .png",
                            },
                        },
                        "required": ["query"],
                    }
                }
            }
        }
    },
)
async def chat_endpoint_v2(
    http_request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="x-tenant-id"),
    x_user_id: Optional[str] = Header(default=None, alias="x-user-id"),
):
    set_current_tenant(x_tenant_id)

    # ── Parse request: JSON or multipart ─────────────────────────────────────
    query = None
    session_id = None
    model_provider = "auto"
    image_mode = "auto"
    stream = False
    files: List[UploadFile] = []
    image_files: List[UploadFile] = []
    inline_filenames: List[str] = []

    content_type = (http_request.headers.get("content-type", "") or "").lower()

    if content_type.startswith("application/json"):
        try:
            raw = await http_request.json()
            parsed = ChatRequestV2(**raw)
            query = parsed.query
            model_provider = parsed.model_provider or "auto"
            session_id = parsed.session_id
            image_mode = parsed.image_mode or "auto"
            stream = bool(parsed.stream)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON payload: {e}")

    elif "multipart/form-data" in content_type:
        try:
            form = await http_request.form()
            query = form.get("query")
            model_provider = form.get("model_provider") or "auto"
            session_id = form.get("session_id")
            image_mode = form.get("image_mode") or "auto"
            stream = str(form.get("stream", "false")).lower() == "true"

            raw_files = form.getlist("files") if hasattr(form, "getlist") else []
            raw_images = form.getlist("images") if hasattr(form, "getlist") else []
            # Fallback for single file uploads where getlist may be empty
            if not raw_files:
                single_file = form.get("files") or form.get("file") or form.get("documents")
                if _is_upload_file(single_file):
                    raw_files = [single_file]
            if not raw_images:
                single_img = form.get("images") or form.get("image")
                if _is_upload_file(single_img):
                    raw_images = [single_img]

            if not raw_files and not raw_images:
                try:
                    keys = list(form.keys()) if hasattr(form, "keys") else []
                except Exception:
                    keys = []
                logger.warning(f"[V2 CHAT] Multipart received with no files/images. Keys={keys}")

            allowed_doc_exts = {".csv", ".tsv", ".xlsx", ".docx", ".txt", ".md", ".pdf"}
            allowed_img_exts = {".jpg", ".jpeg", ".png", ".webp"}

            for f in raw_files:
                if _is_upload_file(f) and f.filename:
                    ext = os.path.splitext(f.filename)[1].lower()
                    if ext in allowed_img_exts:
                        image_files.append(f)  # Auto-reclassify images placed in files
                    elif ext in allowed_doc_exts:
                        files.append(f)
                        inline_filenames.append(f.filename)
                    else:
                        raise HTTPException(status_code=415, detail=f"Unsupported file type: {f.filename}")

            for f in raw_images:
                if _is_upload_file(f) and f.filename:
                    ext = os.path.splitext(f.filename)[1].lower()
                    if ext in allowed_img_exts:
                        image_files.append(f)
                    else:
                        raise HTTPException(status_code=415, detail=f"Unsupported image type: {f.filename}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid multipart payload: {e}")
    else:
        raise HTTPException(status_code=415, detail="Unsupported Content-Type. Use application/json or multipart/form-data.")

    if not query or not str(query).strip():
        raise HTTPException(status_code=422, detail="Missing or empty required field: query")

    session_id = session_id or str(uuid.uuid4())
    client_id = x_tenant_id or (http_request.client.host if http_request.client else "anonymous")

    try:
        _rate_limiter.consume(client_id)
    except Exception as rle:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Slow down requests. Error: {rle}")

    # ── Email auth "done" → retry pending email ─────────────────────────────
    if (str(query or "").strip().lower()) in {"done", "connected", "ok", "okay", "yes, done"}:
        cached = get_session_cache(session_id=session_id, tenant_id=x_tenant_id) or {}
        pending = (cached.get("cache") or {}).get("pending_email") or {}
        if pending.get("to"):
            from app.tools.emailer import send_email_via_composio
            email_result = send_email_via_composio(
                user_id=x_tenant_id or "default",
                to=pending.get("to", []),
                subject=pending.get("subject", "Requested summary"),
                body=pending.get("body", ""),
            )
            if email_result.get("successful"):
                try:
                    upsert_session_cache(
                        session_id=session_id,
                        cache_payload={"pending_email": {"to": [], "subject": "", "body": ""}},
                        tenant_id=x_tenant_id,
                    )
                except Exception:
                    pass
            if email_result.get("successful"):
                answer = "Email sent."
            else:
                answer = email_result.get("error") or "Authentication required to send email."
            try:
                save_chat_turn(session_id=session_id, role="assistant", content=answer, tenant_id=x_tenant_id)
            except Exception:
                pass
            return ChatResponseV2(
                session_id=session_id,
                answer=answer,
                sources=[],
                confidence=0.9 if email_result.get("successful") else 0.6,
                verifier_verdict="NOT_RUN",
                is_hallucinated=False,
                chat_history=get_chat_history(session_id=session_id, tenant_id=x_tenant_id),
                email_action=email_result,
                optimizations={"tools_used": ["get_email_or_send"], "email_retry": True},
                latency_optimizations={},
                active_persona=None,
            )

    start = time.perf_counter()

    # ── Multimodal: Process images (OCR + Vision) ─────────────────────────────
    image_context = None
    image_sources = []
    max_mb = int(os.getenv("MAX_UPLOAD_MB", "20"))

    if image_files:
        try:
            from app.multimodal.file_parser import FileParser
            from app.multimodal.vision import VisionModel
            file_parser = FileParser()
            vision_model = VisionModel()

            for img_file in image_files:
                img_bytes = await img_file.read()
                if len(img_bytes) > max_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"Image {img_file.filename} exceeds {max_mb}MB limit.")

                ocr_text = ""
                vision_text = ""

                if image_mode in ("auto", "ocr"):
                    try:
                        ocr_text = file_parser.parse_image_text(img_bytes)
                    except Exception as e:
                        logger.warning(f"[V2 CHAT] OCR failed for {img_file.filename}: {e}")

                if image_mode in ("auto", "vision") or (image_mode == "auto" and not ocr_text):
                    try:
                        vision_text = vision_model.answer(img_bytes, question=query)
                    except Exception as e:
                        logger.warning(f"[V2 CHAT] Vision failed for {img_file.filename}: {e}")

                combined = ""
                if ocr_text and vision_text:
                    combined = f"[OCR TEXT]\n{ocr_text}\n\n[VISUAL DESCRIPTION]\n{vision_text}"
                elif ocr_text:
                    combined = ocr_text
                elif vision_text:
                    combined = vision_text

                if combined:
                    if image_context:
                        image_context += f"\n\n--- [{img_file.filename}] ---\n{combined}"
                    else:
                        image_context = f"[{img_file.filename}]\n{combined}"
                    image_sources.append({"source": img_file.filename, "score": 1.0, "text": combined[:240]})

            logger.info(f"[V2 CHAT] Processed {len(image_files)} image(s), extracted {len(image_context or '')} chars")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[V2 CHAT] Image processing error: {e}")

    # ── Multimodal: Ingest doc files into PageIndex for this session ──────────
    if files:
        try:
            from app.v2.ingestion.file_upload_v2 import FileUploadServiceV2
            from app.v2.retrieval.pageindex_tool import store_documents_in_tree_cache

            file_payloads = []
            for f in files:
                fb = await f.read()
                if len(fb) > max_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"File {f.filename} exceeds {max_mb}MB limit.")
                file_payloads.append(UploadFile(filename=f.filename, file=__import__("io").BytesIO(fb)))

            svc = FileUploadServiceV2(tenant_id=x_tenant_id)
            docs = await svc.process_files(file_payloads)
            if docs:
                node_count = store_documents_in_tree_cache(session_id=session_id, documents=docs, tenant_id=x_tenant_id, mode="append")
                logger.info(f"[V2 CHAT] Inline file ingest: {len(docs)} docs → {node_count} nodes for session={session_id}")
            else:
                filenames = [f.filename for f in files]
                logger.warning(f"[V2 CHAT] Inline file ingest produced no content for {filenames}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Inline upload failed to extract content from: {', '.join(filenames)}"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[V2 CHAT] Inline file ingest error: {e}")

    # ── Chat history (persisted) ──────────────────────────────────────
    history = get_chat_history(session_id=session_id, tenant_id=x_tenant_id)
    if query:
        save_chat_turn(session_id=session_id, role="user", content=str(query), tenant_id=x_tenant_id)
        history = get_chat_history(session_id=session_id, tenant_id=x_tenant_id)

    # ── Invoke V2 Orchestrator ─────────────────────────────────────────────────
    orchestrator = _get_v2_orchestrator()
    if not orchestrator:
        raise HTTPException(status_code=501, detail="V2 Orchestrator failed to initialize. Check API keys.")

    try:
        forced_sources = []
        if inline_filenames:
            try:
                from app.v2.retrieval.pageindex_tool import fetch_nodes_for_titles
                forced_sources = fetch_nodes_for_titles(
                    session_id=session_id,
                    titles=inline_filenames,
                    tenant_id=x_tenant_id,
                    limit=24,
                )
            except Exception as e:
                logger.warning(f"[V2 CHAT] Failed to fetch forced sources: {e}")
        result = await orchestrator.invoke(
            query=str(query),
            session_id=session_id,
            tenant_id=x_tenant_id,
            image_context=image_context,
            chat_history=history,
            forced_sources=forced_sources,
        )
    except Exception as e:
        import traceback
        logger.error(f"[V2 CHAT ERROR] {e}\n{traceback.format_exc()}")
        err_msg = str(e).lower()
        if any(w in err_msg for w in ["rate limit", "rate_limit", "429", "quota"]):
            friendly = "The AI service is temporarily rate-limited (too many requests). Please wait a moment and try again."
        elif any(w in err_msg for w in ["timeout", "timed out", "connect"]):
            friendly = "The AI service request timed out. Please try again — if this persists, check your API key quota."
        else:
            friendly = f"An internal error occurred processing your request. Please try again. (ref: {str(e)[:80]})"
        return ChatResponseV2(
            session_id=session_id,
            answer=friendly,
            sources=[],
            confidence=0.0,
            verifier_verdict="ERROR",
            is_hallucinated=False,
            chat_history=history,
            active_persona=None,
            email_action=None,
            optimizations={"error": str(e)[:120], "tools_used": []},
            latency_optimizations={},
        )
    finally:
        set_current_tenant(None)

    # ── Merge image sources into result sources ──────────────────────────────
    all_sources = image_sources + result.get("sources", [])
    email_action = result.get("email_action")

    # Persist assistant turn
    try:
        answer_text = result.get("answer", "")
        if answer_text:
            save_chat_turn(session_id=session_id, role="assistant", content=answer_text, tenant_id=x_tenant_id)
    except Exception:
        pass
    history = get_chat_history(session_id=session_id, tenant_id=x_tenant_id)

    # If email needs auth, store pending email for retry after "done"
    try:
        if email_action:
            status = (email_action.get("status") or "").lower()
            successful = bool(email_action.get("successful")) or status == "success"
            if successful:
                upsert_session_cache(
                    session_id=session_id,
                    cache_payload={"pending_email": {"to": [], "subject": "", "body": ""}},
                    tenant_id=x_tenant_id,
                )
            else:
                # Only store pending email for auth-required retries
                needs_retry = (status == "auth_required")
                if needs_retry:
                    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", str(query or ""))
                    subject = "Requested summary"
                    body = answer_text or ""
                    upsert_session_cache(
                        session_id=session_id,
                        cache_payload={"pending_email": {"to": emails, "subject": subject, "body": body}},
                        tenant_id=x_tenant_id,
                    )
    except Exception:
        pass

    # ── Streaming Response ────────────────────────────────────────────────────
    if stream:
        answer_text = result.get("answer", "")

        async def event_gen():
            for chunk in _chunk_text_for_stream(answer_text):
                yield f"data: {chunk}\n\n"
            meta = {
                "session_id": session_id,
                "sources": all_sources,
                "confidence": result.get("confidence", 0.0),
                "verifier_verdict": result.get("verifier_verdict", "NOT_RUN"),
                "is_hallucinated": result.get("is_hallucinated", False),
                "email_action": email_action,
                "optimizations": result.get("optimizations", {}),
                "latency_optimizations": result.get("latency_optimizations", {}),
                "chat_history": history,
                "active_persona": result.get("active_persona"),
            }
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    # ── Standard JSON Response ─────────────────────────────────────────────────
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    lat = result.get("latency_optimizations", {})
    lat["total_time_ms"] = elapsed

    return ChatResponseV2(
        session_id=session_id,
        answer=result.get("answer", "No answer generated."),
        sources=all_sources,
        confidence=result.get("confidence", 0.0),
        verifier_verdict=result.get("verifier_verdict", "NOT_RUN"),
        is_hallucinated=result.get("is_hallucinated", False),
        optimizations=result.get("optimizations", {}),
        chat_history=history,
        latency_optimizations=lat,
        active_persona=result.get("active_persona"),
        email_action=email_action,
    )


# ---------------------------------------------------------------------------
# 2. File Ingestion Endpoint (V2)
# ---------------------------------------------------------------------------
@router.post(
    "/ingest/files",
    response_model=IngestionResponseV2,
    tags=["V2 Ingestion"],
    summary="Ingest Files to PageIndex (V2)",
    description=(
        "Uploads PDF/DOCX/TXT/MD/CSV/XLSX files, extracts Markdown, and builds "
        "a PageIndex tree indexed by session_id.\n\n"
        "**mode=append** (default): adds new nodes to an existing session.\n"
        "**mode=overwrite**: clears existing nodes for that session first.\n\n"
        "Pass the returned session_id to `/api/v2/chat`."
    ),
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
                                "description": ".pdf, .docx, .txt, .md, .csv, .tsv, .xlsx",
                            },
                            "session_id": {"type": "string", "description": "Optional — reuse existing session."},
                            "mode": {
                                "type": "string",
                                "enum": ["append", "overwrite"],
                                "default": "append",
                            },
                        },
                        "required": ["files"],
                    }
                }
            }
        }
    },
)
async def ingest_files_v2_endpoint(
    http_request: Request,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(default=None),
    mode: str = Form(default="append"),
    x_tenant_id: Optional[str] = Header(default=None, alias="x-tenant-id"),
):
    job_id = str(uuid.uuid4())
    session_id = session_id or job_id

    try:
        from app.v2.ingestion.file_upload_v2 import FileUploadServiceV2
        from app.v2.retrieval.pageindex_tool import store_documents_in_tree_cache, clear_session_nodes
        from app.infra.database import upsert_ingestion_job

        # Overwrite mode: clear existing nodes before indexing
        if mode == "overwrite":
            try:
                clear_session_nodes(session_id=session_id, tenant_id=x_tenant_id)
                logger.info(f"[V2 INGEST] Overwrite mode — cleared session {session_id}")
            except Exception as e:
                logger.warning(f"[V2 INGEST] clear_session_nodes failed (may not exist yet): {e}")

        svc = FileUploadServiceV2(tenant_id=x_tenant_id)
        extracted_docs = await svc.process_files(files)

        if not extracted_docs:
            raise HTTPException(status_code=400, detail="No extractable content found in uploaded files. Supported: PDF, DOCX, TXT, MD, CSV, XLSX.")

        node_count = store_documents_in_tree_cache(session_id=session_id, documents=extracted_docs, tenant_id=x_tenant_id, mode=mode)

        logger.info(f"[V2 INGEST FILES] session={session_id} docs={len(extracted_docs)} nodes={node_count} mode={mode}")
        upsert_ingestion_job(
            job_id=job_id,
            status="completed",
            payload={
                "message": f"Indexed {len(extracted_docs)} document(s) → {node_count} PageIndex nodes (mode={mode}).",
                "session_id": session_id,
                "nodes_indexed": node_count,
            },
            tenant_id=x_tenant_id,
        )
        return IngestionResponseV2(
            status="completed",
            message=f"Indexed {len(extracted_docs)} document(s) into {node_count} nodes (mode={mode}).",
            job_id=job_id,
            session_id=session_id,
            nodes_indexed=node_count,
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"[V2 INGEST FILES ERROR] {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"V2 file ingestion failed: {str(e)[:200]}")


# ---------------------------------------------------------------------------
# 3. Crawler Ingestion Endpoint (V2)
# ---------------------------------------------------------------------------
@router.post(
    "/ingest/crawler",
    response_model=IngestionResponseV2,
    tags=["V2 Ingestion"],
    summary="Trigger Modular Crawler (V2)",
    description=(
        "Dispatches V2 crawler (via Celery if enabled, else synchronous). "
        "Crawled pages are indexed into PageIndex — query via /api/v2/chat with the returned session_id.\n\n"
        "**mode=append** (default): adds crawled nodes to existing session.\n"
        "**mode=overwrite**: clears existing nodes first."
    ),
)
async def crawl_endpoint_v2(
    params: CrawlerRequestV2,
    http_request: Request,
    x_tenant_id: Optional[str] = Header(default=None, alias="x-tenant-id"),
):
    job_id = str(uuid.uuid4())
    session_id = params.session_id or job_id

    try:
        celery_enabled = os.getenv("CELERY_ENABLED", "false").lower() == "true"
        from app.infra.database import upsert_ingestion_job
        from app.v2.retrieval.pageindex_tool import clear_session_nodes

        # Overwrite mode: clear existing session nodes
        if params.mode == "overwrite":
            try:
                clear_session_nodes(session_id=session_id, tenant_id=x_tenant_id)
            except Exception as e:
                logger.warning(f"[V2 CRAWLER] Overwrite clear failed: {e}")

        if celery_enabled and os.getenv("CELERY_BROKER_URL"):
            from app.infra.celery_tasks import run_crawler_job_v2
            upsert_ingestion_job(
                job_id=job_id,
                status="pending",
                payload={"message": f"V2 crawler queued for {params.url}.", "session_id": session_id},
                tenant_id=x_tenant_id,
            )
            run_crawler_job_v2.delay(
                job_id=job_id,
                url=params.url,
                max_depth=params.max_depth,
                session_id=session_id,
                tenant_id=x_tenant_id,
            )
            return IngestionResponseV2(
                status="accepted",
                message=f"V2 Crawler dispatched for {params.url}. Poll /api/v2/ingest/status/{job_id} for completion.",
                job_id=job_id,
                session_id=session_id,
            )
        else:
            from app.v2.ingestion.crawler_v2 import CrawlerService as CrawlerServiceV2
            from app.v2.retrieval.pageindex_tool import store_documents_in_tree_cache
            from urllib.parse import urlparse as _urlparse

            _domain = _urlparse(params.url).netloc or "unknown"
            _t = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (x_tenant_id or "global"))[:64] or "global"
            _save_folder = os.path.join("data", "crawled_docs_v2", _t, _domain)
            os.makedirs(_save_folder, exist_ok=True)

            crawler = CrawlerServiceV2(tenant_id=x_tenant_id)
            await crawler.crawl_url(
                url=params.url,
                save_folder=_save_folder,
                simulate=False,
                recursive=(params.max_depth > 1),
                max_depth=params.max_depth,
            )
            docs = [
                {"filename": r.get("url", "page"), "content": r.get("content", "")}
                for r in crawler.results_memory
                if r.get("status") == "success"
            ]
            node_count = store_documents_in_tree_cache(session_id=session_id, documents=docs, tenant_id=x_tenant_id, mode="append")
            upsert_ingestion_job(
                job_id=job_id,
                status="completed",
                payload={"message": f"Crawled {len(docs)} pages → {node_count} nodes.", "session_id": session_id, "nodes_indexed": node_count},
                tenant_id=x_tenant_id,
            )
            return IngestionResponseV2(
                status="completed",
                message=f"Crawled {len(docs)} pages → {node_count} nodes indexed (mode={params.mode}).",
                job_id=job_id,
                session_id=session_id,
                nodes_indexed=node_count,
            )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"[V2 CRAWLER ERROR] {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"V2 crawler failed: {str(e)[:200]}")


# ---------------------------------------------------------------------------
# 4. Ingestion Status
# ---------------------------------------------------------------------------
@router.get(
    "/ingest/status/{job_id}",
    response_model=IngestionStatusV2,
    tags=["V2 Ingestion"],
    summary="Get V2 Ingestion Job Status",
)
async def ingest_status_v2(
    job_id: str,
    x_tenant_id: Optional[str] = Header(default=None, alias="x-tenant-id"),
):
    from app.infra.database import get_ingestion_job
    job = get_ingestion_job(job_id=job_id, tenant_id=x_tenant_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found. It may have expired or not been created yet."
        )
    return IngestionStatusV2(
        status=job.get("status", "unknown"),
        job_id=job_id,
        payload={
            k: v for k, v in job.items()
            if k not in ("status", "job_id", "updated_at")
        } or None,
    )


@router.get(
    "/ingest/status",
    response_model=PageIndexStatusV2,
    tags=["V2 Ingestion"],
    summary="V2 Ingestion Status (PageIndex)",
    description="Return PageIndex session stats (node counts) for a tenant, optionally filtered by session_id.",
)
async def ingest_status_v2_summary(
    session_id: Optional[str] = None,
    x_tenant_id: Optional[str] = Header(default=None, alias="x-tenant-id"),
):
    from app.v2.retrieval.pageindex_store import fetch_pageindex_session_stats
    from app.v2.retrieval.pageindex_tool import get_session_node_count

    sessions = fetch_pageindex_session_stats(tenant_id=x_tenant_id)
    if session_id:
        node_count = get_session_node_count(session_id=session_id, tenant_id=x_tenant_id)
        sessions = [{"session_id": session_id, "nodes_indexed": node_count}]

    total_nodes = sum(int(s.get("nodes_indexed", 0)) for s in sessions)
    return PageIndexStatusV2(
        tenant_id=x_tenant_id,
        session_id=session_id,
        total_sessions=len(sessions),
        total_nodes=total_nodes,
        sessions=sessions,
    )
