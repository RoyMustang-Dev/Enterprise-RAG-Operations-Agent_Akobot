# Enterprise RAG Operations Agent

## 🚀 Project Overview
This project is a **production-grade Retrieval Augmented Generation (RAG) + Multimodal system** for enterprise operations. It prioritizes **explicit orchestration**, **verifiable ingestion**, and **operational traceability** over unbounded chat behavior.

It ingests enterprise documents (PDF/DOCX/TXT/MD), images (PNG/JPEG), audio (WAV/MP3), and crawls modern websites with **sitemap bootstrapping + SPA detection**. Primary LLM calls route through **ModelsLab + Gemini**, with **Groq fallback**. Retrieval is grounded in Qdrant with strict audit logs, and responses can stream natively when providers support it.

### Layer-Wise Architecture Data Flow
*(The following diagram visually maps the precise physical inputs and structural outputs transmitted across each logical layer of the RAG system.)*

![Layer-Wise Execution Architecture](./assets/architecture_layered.png)

## ✨ Comprehensive Key Features
- **Provider-aware routing**: ModelsLab + Gemini as first-class providers, Groq as fallback (no code changes required).
- **Deterministic multi-agent orchestration**: LangGraph DAG with typed `AgentState` ensures deterministic routing and verification.
- **Enterprise ingestion**: Files, images, audio, and crawler outputs normalize into Qdrant with metadata.
- **SPA-aware crawler**: Canonicalization, query/fragment stripping, sitemap seeding, HTTP-first + Playwright fallback.
- **Retrieval quality controls**: Metadata filters, LLM reranking, token-budget trimming, thin-content filtering.
- **Multimodal RAG integration**: Vision (LLaVA/BLIP), OCR (EasyOCR), ModelsLab STT/TTS with local fallbacks.
- **Auditable telemetry**: JSONL logs capture routing, latency, filters, and output previews.
- **Provider-native streaming**: SSE streams from providers when supported, with safe fallback to chunked output.

## 🏗️ Project Architecture

```text
enterprise-rag-agent/         # Root project directory
│
├── app/                      # Enterprise Vertical Slice Architecture (Main Application)
│   ├── api/                  # FastAPI endpoints and request/response schemas
│   │   ├── __init__.py       # Package initializer for API router
│   │   ├── routes.py         # /chat, /ingest, /tts, /transcribe, /agents endpoints
│   │
│   ├── core/                 # Cross-cutting primitives & foundational utilities
│   │   ├── __init__.py       # Package initializer for core components
│   │   ├── rate_limit.py     # TokenBucket in-memory limiter for API protection
│   │   ├── telemetry.py      # JSONL audit logger for tracing LLM execution
│   │   ├── types.py          # AgentState + TelemetryLogRecord strict schema definitions
│   │
│   ├── supervisor/           # Router + intent classification + execution planning
│   │   ├── __init__.py       # Package initializer for supervisor
│   │   ├── intent.py         # Intent classifier (RAG vs Code vs Smalltalk)
│   │   ├── source_scope.py   # Retrieval scope selector (KB only vs Session vs Both)
│   │   ├── planner.py        # Adaptive routing planner for dynamic DAG generation
│   │   ├── router.py         # ExecutionGraph DAG (LangGraph orchestrator)
│   │
│   ├── prompt_engine/        # Guardrails + dynamic prompt formulation
│   │   ├── bootstrapper.py   # Persona expansion + persistence to SQLite DB
│   │   ├── guard.py          # Prompt injection guard (Llama-Guard-4)
│   │   ├── rewriter.py       # Query rewrite for optimized multi-faceted search
│   │   ├── groq_prompts/     # Base prompts + few-shots + persona configurations
│   │
│   ├── ingestion/            # Data pipeline for populating vector memory
│   │   ├── chunker.py        # Token-aware text chunking + sliding window algorithm
│   │   ├── crawler_service.py# HTTP-first crawler + Playwright SPA physical render fallback
│   │   ├── loader.py         # PyMuPDF DOCX/PDF/TXT enterprise extraction toolkit
│   │   ├── pipeline.py       # Main ingestion pipeline mapping directly to Qdrant
│   │
│   ├── retrieval/            # Search mechanics and vector retrieval logic
│   │   ├── embeddings.py     # Gemini embeddings wrapper or BAAI physical local fallback
│   │   ├── metadata_extractor.py # Regex/LLM logic to extract structured filters (dates, authors)
│   │   ├── reranker.py       # LLM Cross-Encoder reranking optimization loop
│   │   ├── hybrid_search.py  # Optional BM25 dense-sparse semantic fusion logic
│   │   ├── vector_store.py   # Qdrant DB adapter mapping for CRUD abstractions
│   │
│   ├── multimodal/           # Multimodality primitives (Image/Audio/OCR/Voice)
│   │   ├── file_parser.py    # EasyOCR + PyTorch advanced document parsing layout module
│   │   ├── vision.py         # LLaVA / BLIP vision backend tensor integrations
│   │   ├── session_vector.py # Ephemeral transient session collections for live file uploads
│   │   ├── multimodal_router.py # Triages and explicitly routes audio/vision multi-modal payloads
│   │
│   ├── agents/               # Specialized execution workers / Sub-agents
│   │   ├── rag.py            # Primary RAG DAG explicit retrieval pipeline execution
│   │   ├── coder.py          # Complex coder assistant for programmatic structural tasks
│   │   ├── smalltalk.py      # Fallback chatter agent strictly respecting the Bootstrapped Persona
│   │
│   ├── reasoning/            # Core logic brain for generation, parsing, and strict-validation
│   │   ├── complexity.py     # Heuristic engine dynamically analyzing required computational loop depth
│   │   ├── synthesis.py      # Core RAG grounded Generation synthesis node (Outputs answers)
│   │   ├── verifier.py       # Final Fact-verification engine asserting against hallucinations
│   │   ├── formatter.py      # Cleans textual output and encapsulates strictly into JSON payloads
│   │
│   ├── rlhf/                 # Reinforcement Learning from Human Feedback infrastructure
│   │   ├── feedback_store.py # SQL logic for tracking and organizing explicit / implicit user rewards
│   │   ├── reward_model.py   # Calculates systemic mathematical telemetry adjustments based on voting models
│   │
│   ├── infra/                # External Infrastructure bindings, memory loops, & hardware constraints
│       ├── llm_client.py     # Unified Provider adapters (ModelsLab/Gemini/Groq/Sarvam native bridging)
│       ├── model_registry.py # Dictates explicit phase routing model registry mapping (Which LLM performs what)
│       ├── history_budget.py # Token-aware algorithmic chat history context truncation trimming (FIFO)
│       ├── token_budget.py   # Context budget allocation limiter for retrieved RAG physical chunks
│       ├── model_bootstrap.py# Downloads HF / Local AI assets automatically upon boot initialization
│       ├── provider_router.py# Robust automatic fail-over mechanism selecting the next active provider
│       ├── celery_app.py     # Celery initialization, Redis handshakes & global asynchronous configuration
│       ├── celery_tasks.py   # Isolated asynchronous Background tasks mapping (web SPA crawling, PDF parsing)
│
├── scripts/                  # OS-Level Shell bootstrap / execution scripts
│   ├── *.ps1                 # Windows PowerShell execution directives for automated deployment
│   ├── mac/                  # macOS Bash Unix execution directives for automated deployment
│
├── data/                     # Persistent local SQLite databases / File stores (RLHF, Persona Caches)
├── assets/                   # Compiled Architecture Mermaid physical diagrams (.mmd) + PNG renders
├── frontend/                 # Decoupled Streamlit Python UI client testing frontend harness
├── .env / .env-copy          # Environment variable secret maps and operational configuration definitions
├── README.md                 # Primary project documentation and structural architectural blueprint
├── requirements.txt          # System Python pip dependency constraints for deterministic building
```

## 🛠️ Technology Stack

| Component | Tech | Reason for Choice/Location |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | Stable ecosystem for AI/ML + infra. |
| **Orchestration** | LangGraph | Typed state DAG for deterministic routing. |
| **Backend** | FastAPI | Async API with Swagger. |
| **Frontend** | Streamlit | Thin client for ops/UI testing. |
| **Core LLM (Reasoning)** | Sarvam / ModelsLab | Deep reasoning framework for primary synthesis (gemini-2.5-flash). |
| **Vision LLM (Multimodal)** | Gemini | Spatial interpretation (gemini-3-pro-image-preview). |
| **Routing & Fallback LLM** | Groq | High-speed routing & filtering (llama-3.1-8b-instant) and verification (llama-3.3-70b-versatile). |
| **Embeddings** | Gemini (Cloud) / BAAI (Local) | Dual-plane hardware-aware embeddings. |
| **OCR** | EasyOCR | Fast local PyTorch OCR without C++ Paddle dependencies. |
| **STT/TTS** | ModelsLab + Whisper | Multi-lingual voice arrays (Tara/Mia) + Local Whisper fallback. |
| **Vector Store** | Qdrant | Multi-tenant, scalable vector DB. |
| **Crawler** | HTTP + Playwright + Sitemap | Fast crawl + SPA fallback. |
| **Queue** | Celery + Redis | Offload long-running crawls/ingestion. |

## 🚀 Installation & Usage

**Runbooks & CI**
- Deployment runbook: `docs/final_delivery_runbook.md`
- CI gates: `docs/ci_gates.md`
- Phase test artifact template: `docs/phase_test_artifact_template.md`

### ⚠️ Critical Prerequisites (Windows)
Install **Microsoft Visual C++ Redistributables (2015-2022)** before setup. Without this, torch/transformers/EasyOCR may fail (`WinError 127`). The packaged installer is in `assets/`.

### Phase A: Environment Setup (`.env`)
You must define the exact `.env` configuration file in the primary project root mapping the necessary variables and boolean limits before booting.
1. Copy `.env-copy` to `.env`.
2. Open `.env` and fill the variables. Below is the step-by-step breakdown:
   - **CORE_LLM_PROVIDER, VISION_LLM_PROVIDER**: Explicitly routes the reasoning tasks. Provide strings like `modelslab`, `groq`, or `sarvam`.
   - **MODELSLAB_API_KEY**: Used for ultra-fast Gemini queries & Audio synthesis. Acquire at `https://modelslab.com/` (Dashboard -> API Keys).
   - **GEMINI_API_KEY**: Direct Google Vision endpoint. Acquire at `https://aistudio.google.com/` (Get API Key).
   - **GROQ_API_KEY**: Required for micro-models (Llama-Guard, intent routing). Acquire at `https://console.groq.com/` (API Keys section).
   - **SARVAM_API_KEY**: Optional enterprise logic routing. Acquire at `https://www.sarvam.ai/`.
   - **QDRANT_URL / QDRANT_API_KEY**: Vector persistence. Acquire at `https://cloud.qdrant.io/`. Create a cluster and copy the URL and API Key over.
   - **CELERY_BROKER_URL**: Background tasks. Maps to a Redis endpoint (e.g. `redis://localhost:6379/0`). Can be run via Docker `docker run -p 6379:6379 redis` or a cloud provider.

*For an exhaustive breakdown of every specific sub-setting (like Image Mode Fallbacks, TTS constraints, etc.), please read the inline documentation meticulously provided inside `.env-copy`.*

### Phase B: Zero-Friction Application Execution
We utilize automated initialization scripts to bypass complex virtual environments securely:

**Windows PowerShell Execution:**
1. `scripts/bootstrap_env.ps1`: Generates a pristine `venv`, destroys zombie python threads, force-reinstalls `requirements.txt`, and probes Windows OS for PyTorch CUDA C++ `.dll` patching automatically. Run this ONLY ONCE during your first setup!
2. `scripts/start_stack.ps1`: The Master execution terminal. Natively spans two separate background threads launching both `start_api.ps1` (FastAPI at port 8000) and `start_celery_worker_dev.ps1` (Redis Queue) simultaneously.
   - *If you wish to scale manually: Call `scripts/start_api.ps1` and `scripts/start_celery_worker_dev.ps1` in separate terminal windows.*

**macOS Bash Execution:**
1. `scripts/mac/bootstrap_env.sh`: Safely formats a localized Mac `venv`, installs architecture requirements, and downloads HuggingFace pre-cache binaries mapped for Apple Silicon/Metal acceleration. Run this ONLY ONCE during your first setup!
2. `scripts/mac/start_stack.sh`: Deploys the unified stack using `&` background jobs ensuring Uvicorn and Celery attach themselves locally without crashing the terminal.
   - *If you wish to scale manually: Call `scripts/mac/start_api.sh` and `scripts/mac/start_celery_worker.sh` in separate terminal windows.*

### Phase C: FastAPI Swagger / Docs
Because the system is Headless API-First, explore exactly how the system connects via **`http://localhost:8000/docs`**.

**Core Operational Endpoints Explained:**
- `GET /api/v1/health`: Basic diagnostic ping returning CPU metrics and online status.
- `GET /api/v1/metrics`: Returns system resource availability (Memory/CPU) alongside current queued asynchronous processing jobs for active orchestrators.
- `POST /api/v1/agents`: 
  - **Function**: Bootstraps the global Persona into the SQLite database.
  - **Options**: Accepts multipart/form-data for Logo uploads (`company_logo`), brand details (`brand_details`), and raw `prompt_instructions` (which gets automatically expanded by the React engine to formulate the final Persona array natively).
- `POST /api/v1/chat`: 
  - **Function**: The 11-Step multi-agent RAG DAG core input vector.
  - **Options**:
    - `query`: The user text payload.
    - `files`: File uploads arrays (processes PDF/Docx iteratively into ephemeral Qdrant pools).
    - `images`: Image uploads tracking (JPG/PNG) directly routed to Vision backends.
    - `model_provider`: Override the `.env` default specifically for this request (`auto`, `modelslab`, `groq`, `gemini`).
    - `image_mode`: `auto` (determine by VRAM constraints), `ocr` (strict PyTorch text), `vision` (Contextual analysis via LLM LLAVA/BLIP).
    - `stream`: Native SSE token streaming Boolean toggle.
    - `reranker_model_name`: Override the default Cross-Encoder on the fly (defaults to `llama-3.1-8b-instant`).
- `POST /api/v1/ingest/files`: 
  - **Function**: Maps physical files natively to Qdrant Vector Stores asynchronously directly for the global knowledge base.
  - **Options**: `files` (array), `mode` (`append` or `overwrite`).
- `POST /api/v1/ingest/crawler`: 
  - **Function**: Executes a headless Playwright array targeting JS-rendered SPA sites.
  - **Options**: `start_url` (origin), `max_depth` (limit heuristic crawl depth), `mode` (`append`/`overwrite`).
- `GET /api/v1/progress/{job_id}`: Polls the Celery pipeline for Ingestion status (Pending/Success).
- `GET /api/v1/ingest/status`: Returns current physical Vector bounds, extraction count metrics, and physical document indexes for your specific Tenant inside the Qdrant Cloud.
- `POST /api/v1/tts`: Direct passthrough logic executing ModelsLab TTS generation exclusively mapping to `mia` (English) or `tara` (Hindi) depending on the textual boundary.
- `POST /api/v1/transcribe`: Audio transcription extracting physical Text via ModelsLab STT (or Whisper/Groq fallbacks) from `.wav` or `.mp3` payloads.
- `POST /api/v1/feedback`: Used to submit numeric or textual feedback linked to a specific session turn, fueling the RLHF database tables.
## 📈 Integration Phases 1-9

1. **Phase 1: API Gateway + Typed Contracts**
   - FastAPI routes, strict Pydantic schemas, uniform request validation.
   - Diagram: `./assets/architecture_phase1.png`
2. **Phase 2: Orchestrator & Routing**
   - LangGraph DAG, intent + source-scope classifiers, adaptive planner.
   - Diagram: `./assets/architecture_phase2.png`
3. **Phase 3: Ingestion Pipeline**
   - Files, images, and text normalized into token-aware chunks.
   - Diagram: `./assets/architecture_phase3_4.png`
4. **Phase 4: Crawler Engine**
   - HTTP-first crawl with canonicalization + sitemap seed + Playwright SPA fallback.
   - Diagram: `./assets/architecture_phase3_4.png`
5. **Phase 5: Retrieval + Reranking**
   - Qdrant search, metadata filtering, LLM reranker.
   - Diagram: `./assets/architecture_phase5_6.png`
6. **Phase 6: Synthesis + Verification**
   - ModelsLab + Gemini synthesis, LLaMA-3.3 verifier, correction loop.
   - Diagram: `./assets/architecture_phase5_6.png`
7. **Phase 7: API-First Decoupling**
   - Streamlit becomes a thin client; FastAPI is the system boundary.
   - Diagram: `./assets/architecture_phase7.png`
8. **Phase 8: Telemetry + Audit**
   - JSONL telemetry for routing, latency, filters, and outputs.
   - Diagram: `./assets/architecture_phase8_9.png`
9. **Phase 9: Provider Fallback & Resilience**
   - ModelsLab + Gemini primary; Groq fallback without code edits.
   - Diagram: `./assets/architecture_phase8_9.png`

## 🏗️ Phase 10: The Complete 11-Step RAG Agentic Architecture

Below represents the current execution DAG. Multimodal ingestion is now integrated into the same flow and feeds ephemeral collections that are merged at retrieval.

1. **Prompt Injection & Safety Guard:** `llama-guard-4-12b` evaluates the user prompt and blocks or allows.
2. **Prompt Rewriter / Query Expansion:** `llama-3.1-8b-instant` produces optimized prompts (low/med/high) for routing and synthesis.
3. **Intent Detection Supervisor:** Classifies the request into RAG / smalltalk / code.
4. **Source Scope Classifier:** Chooses `kb_only`, `session_only`, or `both` based on context.
5. **Agent Dispatch:** Routes to Smalltalk, Coder, or RAG DAG.
6. **Dynamic Metadata Extraction:** Parses metadata filters into strict JSON for Qdrant.
7. **Vector Similarity Search:** Qdrant retrieval using Gemini embeddings or local BAAI fallback.
8. **LLM Reranking:** LLM reranker selects top-K chunks.
9. **Synthesis:** ModelsLab + Gemini (`gemini-2.5-flash`) produce grounded answer (JSON format).
10. **Verification + Correction Loop:** `llama-3.3-70b-versatile` verifies claims and triggers correction if hallucinated.
11. **Formatter + Telemetry + Streaming:** JSON response construction + audit logs + SSE streaming (native where available).

### Visual Architecture Diagram (The Execution DAG)
![11-Step RAG Execution Architecture](./assets/architecture_11_steps.png)

### Strict Step-By-Step Execution Flow
1. **Prompt Guard Security** - Evaluates prompt injection risk and blocks/soft-allows accordingly.
   ![Step 2: Prompt Guard Security](./assets/step02_prompt_guard_security.png)
2. **Query Expansion Rewriter** - Generates optimized prompts for downstream reasoning.
   ![Step 3: Query Expansion Rewriter](./assets/step03_query_expansion_rewriter.png)
3. **Semantic Intent Triage** - Determines whether to route to RAG, code, or smalltalk.
   ![Step 4: Semantic Intent Triage](./assets/step04_semantic_intent_triage.png)
4. **DAG Path Divergence** - Selects the agent based on intent + source scope.
   ![Step 5: DAG Path Divergence](./assets/step05_dag_path_divergence.png)
5. **Metadata Filter Extraction** - Extracts structured filters for Qdrant payloads.
   ![Step 6: Metadata Filter Extraction](./assets/step06_metadata_filter_extraction.png)
6. **Qdrant Similarity Search** - Runs dense retrieval + merges ephemeral session collections.
   ![Step 7: Qdrant Similarity Search](./assets/step07_qdrant_similarity_search.png)
7. **LLM Reranker** - Filters down to the highest-signal chunks.
   ![Step 8: Cross-Encoder Reranker](./assets/step08_cross_encoder_reranker.png)
8. **Complexity Heuristic Analyzer** - Selects reasoning effort and target model.
   ![Step 9: Complexity Heuristic Analyzer](./assets/step09_complexity_heuristic_analyzer.png)
9. **Reasoning Synthesis** - Grounded generation using ModelsLab + Gemini.
   ![Step 10: Reasoning Synthesis](./assets/step10_reasoning_synthesis.png)
10. **Fact Verification** - Verifies claims and triggers correction when needed.
    ![Step 11: Sarvam Fact Verifier](./assets/step11_sarvam_fact_verifier.png)
11. **Formatter + Telemetry** - Builds final JSON response, writes audit logs, streams tokens when enabled.
    ![Step 12: JSON API Formatter](./assets/step12_json_api_formatter.png)

## 👁️ Phase 11: The Multimodality Engine

**Status:** Multimodality is now merged into the main RAG pipeline. The diagram below reflects the integrated flow (images/audio feed ephemeral collections, then merge at retrieval).

### Architecture (Phase 11 Multimodal Flow)
![Phase 11 Architecture](./assets/architecture_phase11.png)

## 🛑 Operational Challenges & Advanced Resolutions
*The following documents critical friction points tackled during stabilization of the Enterprise Stack.*

### 1. The Windows DLL Hell & Execution Precedence
**Challenge:** Initially mapping complex mathematical libraries (`CUDA`, `CuBLAS`) across multiple submodules fundamentally broke the FastAPI server at boot executing a terminal `WinError 127: The specified procedure could not be found`. PyTorch was colliding with leftover physical DLLs from other libraries fighting for `cu11` vs `cu12`.
**Resolution:** We mapped a strict initialization sequence inside `main.py`, forcing `import torch` at line zero before any other C++-bound sub-libraries initialized. We further expanded `HardwareProbe` to violently unshift explicit NVIDIA driver paths globally into the `os.add_dll_directory` array before mapping any secondary models.

### 2. Silencing Hostile Multimodal Subprocesses
**Challenge:** While transitioning OCR logic routines, deep Subprocess architectures and HuggingFace configurations flooded the parent terminal with uncatchable verbose STDOUT warnings mapping to C++ compiler issues resulting in incredibly messy server boots.
**Resolution:** Before initializing model loads inside `file_parser.py` and `model_bootstrap.py`, we explicitly bound deep low-level OS structures wrapping standard streams in `os.dup2` file descriptor null routing. This intercepted and silenced C-level logging natively shielding the FastAPI HTTP interface visibility cleanly.

### 3. Native Reranker Logit Truncation Collisions
**Challenge:** The local BAAI Cross-Encoder executed flawlessly, predicting distances on 30 vector nodes. However, the legacy script arbitrarily bounded `score > 0.35` straight against the raw generated predictions (Logits), which mathematically represent wildly unbounded raw values.
**Resolution:** We integrated absolute `math.exp()` normalizations. The raw model bounds were directly encapsulated inside a **Sigmoid Activation Function** `probability = 1 / (1 + math.exp(-logit))` instantly correcting the unbounded logits into pristine bounded decimals `0.0 -> 1.0`.

### 4. Stateless LangGraph Execution Paths
**Challenge:** Early iterations of the FastAPI payload successfully accepted conversational `chat_history` lists from the Streamlit UI, however, the Agentic Orchestrator routinely initialized an empty semantic array into the graph on execution.
**Resolution:** The core `invoke` and `ainvoke` mappings were rewritten to natively unpack the user query and forcefully append both the user's string and the LLM's final generated logic sequentially back into the global `AgentState` dictionary before executing the final return constraint.
