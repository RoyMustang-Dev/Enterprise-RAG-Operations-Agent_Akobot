# Enterprise RAG Operations Agent

## 🚀 Project Overview
This project is a **production-grade Retrieval-Augmented Generation (RAG) system** designed to operate as an autonomous knowledge agent for enterprise environments. Unlike simple chatbots, this system focuses on **explicit orchestration**, **verifiable data ingestion**, and **operational transparency**.

It allows users to ingest complex enterprise documents (PDF, DOCX, TXT) and crawl dynamic websites, building a unified knowledge base that can be queried with high precision.

## ✨ Key Features
- **Multi-Agent Orchestration**: Uses LangGraph for robust semantic routing and agent management.
- **Multi-Source Ingestion**: Seamlessly handles PDF, DOCX, TXT, and URL-based content.
- **Dynamic Crawler**: Uses **Playwright** to handle Single Page Applications (SPAs) and JavaScript-heavy sites.
- **Specialized Agents**: Independent RAG, Analytical, and Smalltalk agents to prevent processing bottlenecks and crashes.
- **Interactive UI**: A Streamlit-based frontend for easy testing and operation.

## 🏗️ Project Architecture

```text
enterprise-rag-agent/
│
├── app/             # Enterprise Vertical Slice Architecture
│   ├── api/             # HTTP layer ONLY (FastAPI endpoints, swagger definitions)
│   ├── core/            # Cross-cutting primitives (Types, exceptions, telemetry mapping)
│   ├── supervisor/      # ReAct brain (Router map, Intent Detector, Fallback logic)
│   ├── prompt_engine/   # Guardrails & Prompts (Llama-Guard, Rewriter, Templates)
│   ├── ingestion/       # Full Data Pipeline (Crawler, Loader, Chunker, Sync->Async loops)
│   ├── retrieval/       # Search Mechanics (Qdrant DB, BAAI Embeddings, Cross-Encoder Reranker, Metadata Extractor)
│   ├── agents/          # Execution Workers (RAG Agent DAG, Code Agent, Smalltalk Bypass)
│   ├── reasoning/       # Core Logic Brain (Llama-70B Synthesis, Sarvam Verifier, Citation Formatter)
│   ├── rlhf/            # Data Flywheel (Feedback Store, Reward Model logs)
│   ├── safety/          # Safeguards (Hallucination flags, Content Filters)
│   └── infra/           # Systems Infrastructure (Rate Limits, Hardware GPU Probing, DB Init)
│
├── frontend/                 # User Interface
│   └── app.py                # Streamlit Dashboard
│
├── data/                     # Root Data Persistence Storage
│   ├── crawled_docs/         # Scraped Playwright output files
│   ├── uploaded_docs/        # User uploaded manual PDFs/DOCXs
│   ├── qdrant_storage/       # Local Vector DB persist directory
│   └── audit/                # audit_logs.jsonl tracing all node vertices
│
└── tests/                    # System Verification
```

## 🛠️ Technology Stack

| Component | Tech | Reason for Choice |
| :--- | :--- | :--- |
| **Language** | Python 3.11 | Industry standard for AI/ML engineering. |
| **Orchestration** | LangGraph | State-based multi-agent orchestration for robust NLP routing. |
| **Frontend** | Streamlit | Rapid prototyping and interactive data visualization. |
| **Backend API** | FastAPI | High-performance, async-native REST API (Headless SaaS Architecture). |
| **Core Brain LLM**| Groq (`llama-3.3-70b-versatile`) | LPU architecture running at 800+ TPS. Rivals GPT-4o reasoning. |
| **Intent/Speed LLM**| Groq (`llama-3.1-8b-instant`) | Near-instantaneous intent routing and metadata extraction logic. |
| **Independent Verifier**| Sarvam AI (`sarvam-m`) | Secondary LLM layer to mathematically verify fact citations. |
| **Embeddings** | BAAI/bge-large-en-v1.5 + Reranker | State-of-the-art open-source semantic generation. |
| **Vector Store** | Qdrant / Pinecone | Scalable cloud-first vector index (Replaced flat FAISS files). |
| **PDF Processing** | PyMuPDF (fitz) | Fastest and most accurate text extraction for PDFs. |
| **Web Crawling** | **Playwright** + BeautifulSoup | Handles client-side JS rendering for modern SPAs. |

## 🚀 Installation & Usage

### Prerequisites
- Python 3.10+
- OS: Windows/Linux/Mac

### Setup
1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd Enterprise-RAG-Operations-Agent_POC
    ```

2.  **Create and activate virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    playwright install chromium
    ```

4.  **Environment Setup (`.env`):**
    ```env
    GROQ_API_KEY=your_groq_key
    SARVAM_API_KEY=your_sarvam_key
    HF_TOKEN=your_huggingface_read_token
    # PINECONE_API_KEY=your_pinecone_key (Optional: If using managed vector cloud)
    ```

### Running the Application

To run the full stack, you need to open two separate terminals.

5.  **Start the Backend API (Uvicorn):**
    Open your first terminal, ensure your virtual environment is activated, and run the FastAPI server:
    ```bash
    # Activate environment (if not already active)
    venv\Scripts\activate      # Windows
    # source venv/bin/activate # Mac/Linux
    
    # Start the backend server
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    ✅ **Swagger API Docs:** Access the interactive endpoints at `http://localhost:8000/docs`

6.  **Start the Frontend Portal (Streamlit):**
    Open a *second* terminal window, activate the virtual environment again, and launch the UI:
    ```bash
    # Activate environment in the new terminal
    venv\Scripts\activate      # Windows
    # source venv/bin/activate # Mac/Linux
    
    # Start the Streamlit dashboard
    streamlit run frontend/app.py
    ```
    ✅ **Agent Dashboard UI:** Access the portal at `http://localhost:8501`

---

## 🏗️ The Complete 11-Step RAG Agentic Architecture

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
