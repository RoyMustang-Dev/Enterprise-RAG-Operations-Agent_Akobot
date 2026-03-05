# README Updates (Manual Apply)

Below are the exact sections to replace in `README.md`.

---

## ?? Project Overview (Replace Entire Section)
This project is a **production?grade Retrieval?Augmented Generation (RAG) + Multimodal system** for enterprise operations. It prioritizes **explicit orchestration**, **verifiable ingestion**, and **operational traceability** over unbounded chat behavior.

It ingests enterprise documents (PDF/DOCX/TXT/MD), images (PNG/JPEG), audio (WAV/MP3), and crawls modern websites with **sitemap bootstrapping + SPA detection**. Primary LLM calls route through **ModelsLab + Gemini**, with **Groq fallback**. Retrieval is grounded in Qdrant with strict audit logs, and responses can stream natively when providers support it.

### Layer?Wise Architecture Data Flow
*(The following diagram visually maps the precise physical inputs and structural outputs transmitted across each logical layer of the RAG system.)*

![Layer?Wise Execution Architecture](./assets/architecture_layered.png)

---

## ? Comprehensive Key Features (Replace Entire Section)
- **Provider?aware routing**: ModelsLab + Gemini as first?class providers, Groq as fallback (no code changes required).
- **Deterministic multi?agent orchestration**: LangGraph DAG with typed `AgentState` ensures deterministic routing and verification.
- **Enterprise ingestion**: Files, images, audio, and crawler outputs normalize into Qdrant with metadata.
- **SPA?aware crawler**: Canonicalization, query/fragment stripping, sitemap seeding, HTTP?first + Playwright fallback.
- **Retrieval quality controls**: Metadata filters, LLM reranking, token?budget trimming, thin?content filtering.
- **Multimodal RAG integration**: Vision (LLaVA/BLIP), OCR (EasyOCR), ModelsLab STT/TTS with local fallbacks.
- **Auditable telemetry**: JSONL logs capture routing, latency, filters, and output previews.
- **Provider?native streaming**: SSE streams from providers when supported, with safe fallback to chunked output.

---

## ??? Project Architecture (Replace Tree Block)
Use the following updated structure block:

```text
enterprise-rag-agent/
?
??? app/                      # Enterprise Vertical Slice Architecture
?   ??? api/                  # FastAPI endpoints and request/response schemas
?   ?   ??? __init__.py
?   ?   ??? routes.py         # /chat, /ingest/files, /ingest/crawler, /progress, /ingest/status, /tts, /transcribe, /feedback, /agents
?   ?
?   ??? core/                 # Cross-cutting primitives
?   ?   ??? __init__.py
?   ?   ??? rate_limit.py     # TokenBucket in-memory limiter
?   ?   ??? telemetry.py      # JSONL audit logger
?   ?   ??? types.py          # AgentState + TelemetryLogRecord
?   ?
?   ??? supervisor/           # Router + intent + planner
?   ?   ??? __init__.py
?   ?   ??? intent.py         # Intent classifier
?   ?   ??? source_scope.py   # Retrieval scope selector
?   ?   ??? planner.py        # Adaptive routing planner
?   ?   ??? router.py         # ExecutionGraph DAG
?   ?
?   ??? prompt_engine/        # Guardrails + prompts
?   ?   ??? bootstrapper.py   # Persona expansion + persistence
?   ?   ??? guard.py          # Prompt injection guard
?   ?   ??? rewriter.py       # Query rewrite
?   ?   ??? groq_prompts/     # Base prompts + few-shots + persona config
?   ?
?   ??? ingestion/            # Data pipeline
?   ?   ??? chunker.py        # Token-aware chunking
?   ?   ??? crawler_service.py# HTTP-first crawler + Playwright SPA fallback
?   ?   ??? loader.py         # PyMuPDF DOC/PDF extractor
?   ?   ??? pipeline.py       # Ingestion pipeline -> Qdrant
?   ?
?   ??? retrieval/            # Search mechanics
?   ?   ??? embeddings.py     # Gemini embeddings or BAAI local fallback
?   ?   ??? metadata_extractor.py
?   ?   ??? reranker.py       # LLM reranker
?   ?   ??? hybrid_search.py  # Optional BM25
?   ?   ??? vector_store.py   # Qdrant adapter
?   ?
?   ??? multimodal/           # Multimodality primitives
?   ?   ??? file_parser.py    # OCR + doc parsing
?   ?   ??? vision.py         # LLaVA/BLIP vision
?   ?   ??? session_vector.py # Ephemeral session collections
?   ?   ??? multimodal_router.py
?   ?
?   ??? agents/               # Execution workers
?   ?   ??? rag.py            # RAG DAG pipeline
?   ?   ??? coder.py          # Coder agent
?   ?   ??? smalltalk.py
?   ?
?   ??? reasoning/            # Core logic brain
?   ?   ??? complexity.py
?   ?   ??? synthesis.py
?   ?   ??? verifier.py
?   ?   ??? formatter.py
?   ?
?   ??? rlhf/                 # Feedback + reward scoring
?   ?   ??? feedback_store.py
?   ?   ??? reward_model.py
?   ?
?   ??? infra/                # Infrastructure
?       ??? llm_client.py     # Provider adapters (ModelsLab/Gemini/Groq)
?       ??? model_registry.py # Phase model routing
?       ??? history_budget.py # Token-aware chat history trimming
?       ??? token_budget.py   # Context budget for chunks
?       ??? model_bootstrap.py
?       ??? provider_router.py
?       ??? celery_app.py
?       ??? celery_tasks.py
?
??? scripts/
?   ??? *.ps1                 # Windows scripts
?   ??? mac/                  # macOS scripts
?
??? data/                     # Persistent data stores
??? assets/                   # Mermaid diagrams + PNG renders
??? frontend/                 # Streamlit UI client
??? .env / .env-copy
??? README.md
??? requirements.txt
```

---

## ??? Technology Stack (Replace Entire Table)
| Component | Tech | Reason for Choice/Location |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | Stable ecosystem for AI/ML + infra. |
| **Orchestration** | LangGraph | Typed state DAG for deterministic routing. |
| **Backend** | FastAPI | Async API with Swagger. |
| **Frontend** | Streamlit | Thin client for ops/UI testing. |
| **Primary LLMs** | ModelsLab + Gemini | Fast, reliable core inference. |
| **Fallback LLM** | Groq | Safe fallback when paid keys missing. |
| **Reranker** | LLM reranker (llama?3.1?8b?instant) | Fast semantic ranking. |
| **Embeddings** | Gemini embeddings or BAAI/bge?large | Cloud speed + local fallback. |
| **Vision** | LLaVA / BLIP | High?VRAM + low?VRAM fallback. |
| **OCR** | EasyOCR | Fast local OCR. |
| **STT/TTS** | ModelsLab + local fallback | External quality + offline safety. |
| **Vector Store** | Qdrant | Multi?tenant, scalable vector DB. |
| **Crawler** | HTTP + Playwright + Sitemap | Fast crawl + SPA fallback. |
| **Queue** | Celery + Redis | Offload long?running crawls/ingestion. |

---

## ?? Installation & Usage (Replace Entire Section)
**Runbooks & CI**
- Deployment runbook: `docs/final_delivery_runbook.md`
- CI gates: `docs/ci_gates.md`
- Phase test artifact template: `docs/phase_test_artifact_template.md`

### ?? Critical Prerequisites (Windows)
Install **Microsoft Visual C++ Redistributables (2015?2022)** before setup. Without this, torch/transformers/EasyOCR may fail (`WinError 127`). The packaged installer is in `assets/`.

### Phase A: Environment Setup (`.env`)
Copy `.env-copy` to `.env` and fill **all required keys**. The template includes step?by?step guidance for each variable.

### Phase B: Zero?Friction Application Execution
**Windows (PowerShell)**
1. `scripts/bootstrap_env.ps1` (first?time setup)
2. `scripts/start_stack.ps1` (API + Celery)
3. Optional: `scripts/start_api.ps1` or `scripts/start_celery_worker_dev.ps1`

**macOS (Bash)**
1. `scripts/mac/bootstrap_env.sh` (first?time setup)
2. `scripts/mac/start_stack.sh` (API + Celery)
3. Optional: `scripts/mac/start_api.sh` or `scripts/mac/start_celery_worker.sh`

### Phase C: FastAPI Swagger / Docs
Open Swagger at `http://localhost:8000/docs` after startup.

**Endpoints**
- `GET /` and `GET /api/v1/health`: Health checks
- `POST /api/v1/agents`: Bootstrap global persona
- `POST /api/v1/chat`: Unified RAG + files + images
  - `model_provider`: `auto|modelslab|gemini|groq`
  - `image_mode`: `auto|ocr|vision`
  - `stream`: `true|false` (provider?native streaming)
  - headers: `x-tenant-id`, `x-user-id`
- `POST /api/v1/ingest/files`: Upload docs for ingestion
- `POST /api/v1/ingest/crawler`: Crawl a URL + ingest
- `GET /api/v1/progress/{job_id}`: Ingestion progress
- `GET /api/v1/ingest/status`: Vector stats for tenant
- `POST /api/v1/tts`: Text?to?speech (ModelsLab + fallback)
- `POST /api/v1/transcribe`: Speech?to?text (ModelsLab + fallback)
- `POST /api/v1/feedback`: RLHF feedback

---

## ?? Integration Phases 1-9 (Replace Entire Section)
1. **Phase 1: API Gateway + Typed Contracts**
   - FastAPI routes, strict Pydantic schemas, uniform request validation.
   - Diagram: `./assets/architecture_phase1.png`
2. **Phase 2: Orchestrator & Routing**
   - LangGraph DAG, intent + source?scope classifiers, adaptive planner.
   - Diagram: `./assets/architecture_phase2.png`
3. **Phase 3: Ingestion Pipeline**
   - Files, images, and text normalized into token?aware chunks.
   - Diagram: `./assets/architecture_phase3_4.png`
4. **Phase 4: Crawler Engine**
   - HTTP?first crawl with canonicalization + sitemap seed + Playwright SPA fallback.
   - Diagram: `./assets/architecture_phase3_4.png`
5. **Phase 5: Retrieval + Reranking**
   - Qdrant search, metadata filtering, LLM reranker.
   - Diagram: `./assets/architecture_phase5_6.png`
6. **Phase 6: Synthesis + Verification**
   - ModelsLab + Gemini synthesis, LLaMA?3.3 verifier, correction loop.
   - Diagram: `./assets/architecture_phase5_6.png`
7. **Phase 7: API?First Decoupling**
   - Streamlit becomes a thin client; FastAPI is the system boundary.
   - Diagram: `./assets/architecture_phase7.png`
8. **Phase 8: Telemetry + Audit**
   - JSONL telemetry for routing, latency, filters, and outputs.
   - Diagram: `./assets/architecture_phase8_9.png`
9. **Phase 9: Provider Fallback & Resilience**
   - ModelsLab + Gemini primary; Groq fallback without code edits.
   - Diagram: `./assets/architecture_phase8_9.png`

---

## ??? Phase 10: The Complete 11-Step RAG Agentic Architecture (Replace Section)
Below represents the current execution DAG. Multimodal ingestion is now integrated into the same flow and feeds ephemeral collections that are merged at retrieval.

1. **Prompt Injection & Safety Guard:** `llama-guard-4-12b` evaluates the user prompt and blocks or allows.
2. **Prompt Rewriter / Query Expansion:** `llama-3.1-8b-instant` produces optimized prompts (low/med/high) for routing and synthesis.
3. **Intent Detection Supervisor:** Classifies the request into RAG / smalltalk / code.
4. **Source Scope Classifier:** Chooses `kb_only`, `session_only`, or `both` based on context.
5. **Agent Dispatch:** Routes to Smalltalk, Coder, or RAG DAG.
6. **Dynamic Metadata Extraction:** Parses metadata filters into strict JSON for Qdrant.
7. **Vector Similarity Search:** Qdrant retrieval using Gemini embeddings or local BAAI fallback.
8. **LLM Reranking:** LLM reranker selects top?K chunks.
9. **Synthesis:** ModelsLab + Gemini (`gemini-2.5-flash`) produce grounded answer (JSON format).
10. **Verification + Correction Loop:** `llama-3.3-70b-versatile` verifies claims and triggers correction if hallucinated.
11. **Formatter + Telemetry + Streaming:** JSON response construction + audit logs + SSE streaming (native where available).

### Visual Architecture Diagram (The Execution DAG)
![11-Step RAG Execution Architecture](./assets/architecture_11_steps.png)

### Strict Step?By?Step Execution Flow
1. **Prompt Guard Security** ? Evaluates prompt injection risk and blocks/soft?allows accordingly.
   ![Step 2: Prompt Guard Security](./assets/step02_prompt_guard_security.png)
2. **Query Expansion Rewriter** ? Generates optimized prompts for downstream reasoning.
   ![Step 3: Query Expansion Rewriter](./assets/step03_query_expansion_rewriter.png)
3. **Semantic Intent Triage** ? Determines whether to route to RAG, code, or smalltalk.
   ![Step 4: Semantic Intent Triage](./assets/step04_semantic_intent_triage.png)
4. **DAG Path Divergence** ? Selects the agent based on intent + source scope.
   ![Step 5: DAG Path Divergence](./assets/step05_dag_path_divergence.png)
5. **Metadata Filter Extraction** ? Extracts structured filters for Qdrant payloads.
   ![Step 6: Metadata Filter Extraction](./assets/step06_metadata_filter_extraction.png)
6. **Qdrant Similarity Search** ? Runs dense retrieval + merges ephemeral session collections.
   ![Step 7: Qdrant Similarity Search](./assets/step07_qdrant_similarity_search.png)
7. **LLM Reranker** ? Filters down to the highest?signal chunks.
   ![Step 8: Cross-Encoder Reranker](./assets/step08_cross_encoder_reranker.png)
8. **Complexity Heuristic Analyzer** ? Selects reasoning effort and target model.
   ![Step 9: Complexity Heuristic Analyzer](./assets/step09_complexity_heuristic_analyzer.png)
9. **Reasoning Synthesis** ? Grounded generation using ModelsLab + Gemini.
   ![Step 10: Reasoning Synthesis](./assets/step10_reasoning_synthesis.png)
10. **Fact Verification** ? Verifies claims and triggers correction when needed.
    ![Step 11: Sarvam Fact Verifier](./assets/step11_sarvam_fact_verifier.png)
11. **Formatter + Telemetry** ? Builds final JSON response, writes audit logs, streams tokens when enabled.
    ![Step 12: JSON API Formatter](./assets/step12_json_api_formatter.png)

---

## ??? Phase 11: The Multimodality Engine (Replace Section)
**Status:** Multimodality is now merged into the main RAG pipeline. The diagram below reflects the integrated flow (images/audio feed ephemeral collections, then merge at retrieval).

### Architecture (Phase 11 Multimodal Flow)
![Phase 11 Architecture](./assets/architecture_phase11.png)
