# AKO AI Support Agent: Master Implementation Guide

This document (`README-implementation.md`) serves as the definitive Single Source of Truth for the architecture, routing logic, model selection, and execution roadmap for transforming the Enterprise RAG POC into a production-grade, API-first Multi-Tenant SaaS platform.

It aggregates all design decisions, zero-cost strategies, and feature pipelines discussed and implemented across all project phases.

---

## 1. Executive Summary: The SaaS Multimodal Shift

The objective is to deliver a fully headless, white-label AI SaaS platform that allows external clients to create their own custom support bots. 

**Key Transformation Highlights:**
*   **UI Decoupling:** The monolithic Streamlit application is sequestered. The core system exclusively exposes robust **FastAPI endpoints (REST/OpenAPI)**.
*   **Traceability & Audit Logging:** Every agent action, confidence score, and routing path is explicitly traced, surfaced to the frontend UI, and persistently logged to `data/audit/audit_logs.jsonl` for enterprise accountability.
*   **Multi-Tenancy:** Routes are designed to differentiate tenant data, allowing various clients to customize their Agent Name, Prompt instructions, and branding.
*   **Multimodal Capabilities:** The Chat APIs natively accept `multipart/form-data`, meaning the AI can process standard Text, Voice Notes (.wav/.mp3), Images (.jpg, .png), and temporary Document URLs.
*   **Action-Oriented Agents:** The core `SupervisorAgent` doesn't just answer questions; it routes logic to specialized sub-agents capable of executing Webhooks (e.g., generating Tickets, processing Refunds, engaging Human Handoffs).

---

## 2. API Architecture & Endpoint Design

The backend is structured into modular FastAPI routers (`backend/routers/`). The auto-generated Swagger documentation sits at `/docs`.

### A. Bot Setup & Configuration (`/api/v1/settings`)
*   **`POST /configure`:** Accepts the initial client dashboard payload (Bot Name, Company Logo, Brand Details, System Prompt).

### B. Knowledge Base Ingestion (`/api/v1/ingest`)
*   **`POST /files`:** Handles manual uploads (PDF/DOCX) and converts them into searchable vectors. Supports `append` or `start_fresh` modes.
*   **`POST /crawler`:** Triggers an advanced headless crawler against a provided URL to scrape a website and auto-generate the Knowledge Base (incorporating the `Aeko-test-crawl` repo logic).

### C. Multimodal Chat Execution (`/api/v1/chat`)
*   **`POST /`:** The primary conversation interface. Accepts `message` alongside optional `audio_file`, `image_file`, or `document_file`. Safely returns `<think>` blocks, confidence scores, and hallucination flags.

---

## 3. Explicit LLM Routing & Multi-Provider Strategy

To ensure enterprise stability, prevent vendor lock-in, and optimize costs, the backend implements a deterministic `LLMRouter` (`backend/generation/llm_router.py`).

We explicitly assign *which* model performs *which* task using Environmental Variables (`.env`), separating the "Core Brain" from "Situational Tools."

### How Routing Works:
1.  **Core Reasoning Engine (`CORE_LLM_PROVIDER`):** 
    *   This handles the vast majority of tasks: general chat logic, classifying RAG intents, invoking tool webhooks, and structuring the final response.
    *   *Default Setting:* `sarvam` (The user's preferred enterprise orchestrator).
    *   *Fallback Sequence:* If Sarvam is unavailable, it gracefully degrades: `Anthropic` -> `OpenAI` -> `Groq` -> `Local Ollama`.
2.  **Situational Multimodal Models:**
    *   If a client uploads an image, the `CORE` model might lack vision capabilities.
    *   The `/chat` endpoint explicitly overrides the core logic and requests a target provider: `target_provider="gemini"`.
    *   *The Result:* The `LLMRouter` triggers Gemini *only* for the image extraction sub-task, grabbing the text description, and handing it back to Sarvam to compile the final answer.

---

## 4. The Zero-Cost Model Execution Strategy

We are implementing a matrix of Free-Tier APIs and lightweight Local Models to deliver premium SaaS capabilities without paying per-token API overheads.

### The Categorical Deployment Matrix:

| Capability | Assigned Model | Deployment Type | Cost Tier | Purpose in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Embeddings** | `bge-large-en-v1.5` | 🏠 Local (Python) | 🟢 100% Free | Converts documents & text to vectors. Highly accurate (Must be L2 Normalized). |
| **Vector Index** | `Qdrant` / `Pinecone` | ☁️ Cloud DB | 🟢 Free Tier | Serverless Vector DB (Replaces FAISS). Pinecone = managed, Qdrant = managed + OSS. |
| **Core Brain** | `Sarvam API` | ☁️ Cloud API | 🟢 Free/Client Key | Main logic, agent routing, intent classification. |
| **Failsafe Brain**| `Groq` (Llama 3) | ☁️ Cloud API | 🟢 Free Tier | Lightning-fast text generator (800 tps). Used if main API fails. |
| **Vision (Image)** | `Gemini 1.5 Flash` | ☁️ Cloud API | 🟢 Free Tier | Situational LLM explicitly invoked to describe user-uploaded images/errors. |
| **Audio (STT)** | `faster-whisper` | 🏠 Local (Python) | 🟢 100% Free | Instant offline transcription of Voice Notes (`.wav` -> text). |

---

## 5. Enterprise Data Retrieval Upgrade (Vector DB & Chunking)

The original prototype relied on a flat, local FAISS index (`faiss_store.py`) and basic word-splitting logic. To achieve true Enterprise status, the architecture is being upgraded:

### A. Qdrant vs. Pinecone Database Migration
The local FAISS file is a bottleneck and prevents elastic horizontal scaling. We are transitioning to an Enterprise-grade Cloud Vector DB. 

**Is Pinecone a Downgrade?**
**Absolutely not.** Pinecone is a $1.5 Billion+ enterprise industry standard used by Notion and Zapier. Moving from FAISS to Pinecone or Qdrant is a massive *upgrade*.

**Comparison for a Completely Free Solution:**
*   **Qdrant:** Offers a "1GB Free Forever" Cloud Cluster, which is generous enough to store ~1 million 768-dimensional vectors. It is completely open-source (Rust), meaning if the Cloud tier limits you, you can self-host the exact same DB via Docker for absolutely zero software cost.
*   **Pinecone:** Offers a fully managed "Serverless Starter" free tier featuring 2GB of index storage (~300k - 1 Million vectors) and 2 Million writes/month. It is closed-source and fully managed.
*   **Decision:** We will support **Qdrant** as the primary `VectorStore` adapter since it provides maximum flexibility (Cloud + Local Docker hosting) and fits the "completely free" SaaS scaling ethos beautifully, while keeping Pinecone as an interchangeable backend module.

### B. Intelligent Chunking & Embedding Normalization
*   **Token-Aware Recursive Splitter:** We are replacing the blind "512 word limit" semantic chunker (which unknowingly chops off data because 512 words exceeds 512 tokens). We will use LangChain's `RecursiveCharacterTextSplitter` paired with the BGE HuggingFace tokenizer to chunk perfectly by paragraphs/sentences.
*   **Cosine Normalization:** The `bge-large-en-v1.5` embeddings explicitly require `normalize_embeddings=True` to execute Cosine Similarity correctly. We will enforce this in `embedding_model.py` to prevent retrieval skew.

### C. Advanced Retrieval (Phase 2 RAG)
*   Implementation of **Cross-Encoder Re-ranking** (e.g., retrieving 30 chunks and having an AI reranker strictly score the top 5).
*   Dynamic LLM-based Metadata Extraction (replacing hardcoded filename filters with dynamic `$eq` JSON payload queries).

---

## 6. Real-time Implementation Status Tracker

*   **Phase 1-4: Foundation & Execution Architectures ✅ (COMPLETED)**
    *   FastAPI modular routing (`/chat`, `/ingest`, `/settings`).
    *   Deterministic multi-provider logic implemented natively via `app/api/routes.py`.
    *   Integrated confidence scores, hallucination flags, and true routing optimizations.
    *   Successfully persisted complex JSONL logging natively in `data/audit/audit_logs.jsonl`.
    *   Migrated from FAISS to Qdrant `VectorStore`.
    *   Implement Token-Aware Recursive text splitting.
    *   Enforce L2 Normalization on the BGE Embedding model natively inside Qdrant endpoints.
    *   Integrated async Background Crawlers for dynamic recursive ingestion.

---

## 7. Cloud API Fallback Stack (Active Routing Logic)

If the active Enterprise RAG system is deployed on a weak virtual server (e.g., 2 CPU cores, 4GB RAM), running an 8B LLM locally will permanently crash it with Out-Of-Memory (OOM) errors. 

To prevent this, the architecture natively intercepts `.env` API keys and physically bypasses local PyTorch execution, transforming the deployment into a highly scalable, serverless Cloud Router natively processing inputs exclusively across API endpoints.

### 1. Vectorization: Embedding APIs
Cloud embedding APIs offer lightning-fast vector array calculations mapped to identical dimensional bounds without requiring local GPUs.
| Provider | Model | Cost (per 1M tokens) | Highlights |
| :--- | :--- | :--- | :--- |
| **OpenAI** | `text-embedding-3-small` | **$0.02** | The industry standard for cost-to-performance. Maps a 10,000-page wiki for less than $0.50 in under a minute. |
| **Cohere** | `embed-english-v3.0` | $0.10 | Excellent multilingual support and native Multimodal (Text + Image) embeddings. |
| **Voyage AI** | `voyage-3` | $0.06 | Exceptional at specialized domains (finance, law, code). Massive 200M free token initial tier. |

*Architectural Default:* OpenAI `text-embedding-3-small` (if `OPENAI_API_KEY` is present).

### 2. Core Brain LLM (Reasoning & Orchestration)
The "Core Brain" requires high intelligence, massive context windows, and reliable structured JSON output for strict LangGraph array routings.
| Provider | Model | Cost (per 1M In/Out) | Context Window | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Groq** | `llama-3.3-70b-versatile` | Free Tier | **131K tokens** | **Best Raw Speed.** Perfect for rapid reasoning natively via LPU logic gates. |
| **Anthropic** | `claude-3-5-sonnet` | $3.00 / $15.00 | 200K tokens | **Highest Raw Intelligence.** Dominates in dynamic coding synthesis and hallucination-free reasoning. |

*Architectural Default:* `llama-3.3-70b-versatile` offers the ultimate speed-to-performance ratio for complex RAG synthesis.

### 3. Native Multimodal Transcriptions
If the user uploads documents or sends physical media bounds, these models parse the input cleanly into the central Pipeline stream natively.
| Provider | Model | Cost | Best Use Case |
| **OpenAI** | `whisper-large-v3-turbo` | $0.00 | **Mass Scaling Background Transcription.** Feed it hundreds of `.wav` voice nodes asynchronously via Groq's high-throughput servers. |
| **Local CPU** | `faster-whisper` | ~$0.00 | **Precision Data.** Flawless at offline environments where streaming API boundaries are strictly isolated behind physical edge servers directly analyzing arrays. |

---

## 8. LLM Provider Status: Sarvam vs Groq

Based on the capabilities extracted from the Groq API and our initial implementation of `Sarvam AI`, Groq explicitly powers the real-time SaaS engine natively.

### Original Baseline: Sarvam AI (`sarvam-m`)
- **Capabilities:** Core reasoning, highly optimized for Indian enterprise context (indic languages). 
- **Latency:** Moderate (standard HTTP round trips).
- **Current System Status:** Deployed expressly inside the execution DAG as the **Independent Fact Verifier** (`app/reasoning/verifier.py`), scoring claims line-by-line natively against the Vector Context chunks.

### The Real-Time SaaS Core: Groq
Groq utilizes LPU (Language Processing Unit) architecture, dramatically reducing Time-To-First-Token (TTFT) and achieving massive generation speeds (often 800+ tokens per second).

#### 1. The Core Reasoning Brain: `llama-3.3-70b-versatile`
- **Context Window:** 131,072 tokens.
- **Why it matters:** Llama 3.3 70B runs the primary intelligent synthesis loops natively parsing JSON extraction responses through the `app/reasoning/synthesis.py` node bounds dynamically natively via LPU execution.

#### 2. The Speedy Failsafe / Supervisor Intent: `llama-3.1-8b-instant`
- **Context Window:** 131,072 tokens.
- **Why it matters:** Deployed strictly dynamically across the semantic Router Supervisor in `app/supervisor/intent.py` natively triaging user intent boundaries instantly natively (Smalltalk vs. Execution Analytics vs. Embedded Context Extraction).

#### 3. Voice Transcription (STT): `whisper-large-v3-turbo`
- **Why it matters:** We can offload STT processing directly to Groq to save local CPU overhead dynamically transcribing user audio nodes asynchronously.

#### 4. The Native JSON Metadata Agent: `qwen-2.5-coder-32b`
- **Why it matters:** Qwen models are exceptional at parsing dynamic prompt states into strict explicit API arrays. Deployed dynamically inside `app/retrieval/metadata_extractor.py` translating the user's natural language into exact Qdrant Filter objects.

---

## 9. Comprehensive System Capabilities (API Models)

The deployed Groq API key unlocks access to a series of highly requested, cutting-edge, and even preview-tier models dynamically mapped across the Enterprise Application Architecture dynamically bounds.

### 1. The Core Reasoning Engines (General Purpose)
*   **`llama-3.3-70b-versatile` (Meta):** A 70 billion parameter model functioning as the current state-of-the-art open-source logic engine natively routing context schemas.
*   **`openai/gpt-oss-120b` & `openai/gpt-oss-20b` (OpenAI):** Highly efficient open-weight Mixture-of-Experts (MoE) models optimized for function calling and logic array responses.

### 2. The Speedy Assistants (Low Latency / High Throughput)
*   **`llama-3.1-8b-instant` (Meta):** An ultra-fast, lightweight 8B model natively executing the Supervisor Agent triage.
*   **`qwen/qwen3-32b` (Alibaba Cloud):** The latest Qwen architecture uniquely excels at coding and structured JSON generation explicitly deployed into Metadata Extraction parameters natively.

### 3. Agentic & Tool-Use Specialists (Compound Systems)
*   **`groq/compound` & `groq/compound-mini` (Groq):** Advanced composite AI systems specifically designed to autonomously solve problems using external tools natively handling edge routines dynamically across complex UI states.



## 8. Complete System Folder Architecture

```text
app/
├── api/             # HTTP layer ONLY (FastAPI endpoints, swagger definitions)
├── core/            # Cross-cutting primitives (Types, exceptions, telemetry mapping)
├── supervisor/      # ReAct brain (Router map, Intent Detector, Fallback logic)
├── prompt_engine/   # Guardrails & Prompts (Llama-Guard, Rewriter, Templates)
├── ingestion/       # Full Data Pipeline (Crawler, Loader, Chunker, Sync->Async loops)
├── retrieval/       # Search Mechanics (Qdrant DB, BAAI Embeddings, Cross-Encoder Reranker, Metadata Extractor)
├── agents/          # Execution Workers (RAG Agent DAG, Code Agent, Smalltalk Bypass)
├── reasoning/       # Core Logic Brain (Llama-70B Synthesis, Sarvam Verifier, Citation Formatter)
├── rlhf/            # Data Flywheel (Feedback Store, Reward Model logs)
├── safety/          # Safeguards (Hallucination flags, Content Filters)
└── infra/           # Systems Infrastructure (Rate Limits, Hardware GPU Probing, DB Init)

data/                # 📂 ROOT DATA PERSISTENCE (Strictly Ignored by Git Tracker)
├── uploaded_docs/   # User uploaded manual PDFs/DOCXs
├── crawled_docs/    # Scraped Playwright structured JSON output domains
├── qdrant_storage/  # Local Vector DB persist flat-file directory
└── audit/           # append-only audit_logs.jsonl tracing all node vertices
```

---

## 9. The Complete 11-Step RAG Agentic Architecture

Below represents the exhaustive integration of the enterprise RAG standards we implemented natively into the LangGraph execution flow, strictly isolating vector similarity from intelligent reasoning.

1. **Prompt Injection & Safety Guard:** Protects the execution graph from system prompt extraction and RAG poisoning using `gpt-oss-safeguard-20b`.
2. **Prompt Rewriter / Query Expansion:** Mathematically expands ambiguous user queries (e.g., "What is it?" -> "What is the refund policy?") using historical context.
3. **Intent Detection Supervisor:** A high-speed classifier (`llama-3.1-8b-instant`) strictly routing the execution state without blocking the async API loop.
4. **Agent Dispatch / Smalltalk Bypass:** Routes trivial greetings to a lightweight responder, bypassing the expensive vector database.
5. **Dynamic Metadata Extraction:** Leverages `qwen-32b` to parse the user's natural language into strict JSON `$eq` filters, mapping directly to Qdrant payloads.
6. **Vector Similarity Search (Top 30):** Executes a high-recall L2 Cosine Distance search utilizing `BAAI/bge-large-en-v1.5` embeddings on the GPU.
7. **Cross-Encoder Reranking (Top 5):** Evaluates the top 30 chunks through a rigorous semantic cross-encoder algorithm, discarding hallucination risks and returning strictly the Top 5.
8. **Core Reasoning Synthesis:** The `llama-3.3-70b-versatile` master logical engine ingests the 5 verified chunks and structures a coherent JSON payload.
9. **Independent Fact Verifier:** A sovereign model (`Sarvam M`) audits the generated text line-by-line exclusively against the source chunks, redacting unsourced claims.
10. **Formatter & Citation Modeler:** Injects physical markdown URL and Document links natively into the structured stream response for Streamlit UI rendering.
11. **Telemetry & RLHF Auditing:** Traps latency matrices, token bounds, and hallucination verdicts natively into the `audit_logs.jsonl` pipeline.

### Visual Architecture Diagram (The Execution DAG)

![11-Step RAG Execution Architecture](./assets/architecture_11_steps.png)
