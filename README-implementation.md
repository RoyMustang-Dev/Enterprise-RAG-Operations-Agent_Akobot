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

## 6. Implementation Roadmap (Status Tracker)

*   **Phase 6: Exposing API Interfaces ✅ (COMPLETED)**
    *   FastAPI modular routing (`/chat`, `/ingest`, `/settings`).
    *   Deterministic `LLMRouter` multi-provider logic implemented.
*   **Phase 7: Full Traceability & Audit Logging ✅ (COMPLETED)**
    *   Integrated confidence scores, hallucination flags, and true routing optimizations into the Streamlit UI without truncation.
    *   Successfully persisted complex JSONL logging natively in `backend/generation/rag_service.py`.
*   **Phase 8: Enterprise Data Ingestion Upgrade ⏳ (IN PROGRESS)**
    *   Migrate from FAISS to Qdrant/Pinecone `VectorStore`.
    *   Implement Token-Aware Recursive text splitting.
    *   Enforce L2 Normalization on the BGE Embedding model.
*   **Phase 9: External Production Crawler Integration**
    *   Audit and replace legacy scraping scripts with the robust `Aeko-test-crawl` repository.
*   **Phase 10: Multimodal Processing Pipelines**
    *   Implement `faster-whisper` transcription for Audio inputs.
    *   Implement target Gemini Vision processing for Image inputs.
*   **Phase 11: Action-Oriented CRM Agents**
    *   Implement `TicketingAgent`, `RefundAgent`, and `HumanEscalationAgent` Webhooks.
