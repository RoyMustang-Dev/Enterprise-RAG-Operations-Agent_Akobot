# V2 Pipeline — Tests & Current State Report
**Generated:** 2026-03-18 20:54 IST  
**Environment:** Windows 10, NVIDIA GeForce GTX 1650 Ti, 12 CPU cores  
**Stack:** FastAPI (port 8000), Celery Worker (Redis broker, pool=solo), V2 PageIndex pipeline  

---

## Bugs Fixed This Session

### Bug 1 — `celery_tasks.py` IndentationError (CRITICAL)
**File:** `app/infra/celery_tasks.py` lines 215-216  
**Root cause:** Extra leading spaces (2 extra) on the `from` and `crawler =` lines inside `run_crawler_job_v2`, causing `IndentationError` on Python module import. This caused the Celery worker to exit immediately on startup.  
**Fix:** Removed the extra indentation.  
```diff
-          from app.v2.ingestion.crawler_v2 import CrawlerService as CrawlerServiceV2
-          crawler = CrawlerServiceV2(tenant_id=tenant_id)
+        from app.v2.ingestion.crawler_v2 import CrawlerService as CrawlerServiceV2
+        crawler = CrawlerServiceV2(tenant_id=tenant_id)
```

### Bug 2 — Unsupported `session_id` kwarg in `crawl_url` call
**File:** `app/infra/celery_tasks.py` line 225  
**Root cause:** The Celery V2 crawler task passed `session_id=session_id` to `CrawlerService.crawl_url()`, but that method does not accept `session_id` as a parameter. This caused every V2 crawler Celery task to fail with `TypeError` immediately after the crawl started.  
**Fix:** Removed the unsupported kwarg. The `session_id` is correctly used *after* crawling to key the PageIndex tree cache.  
```diff
-                session_id=session_id
```

### Bug 3 — Missing `upsert_ingestion_job` / `get_ingestion_job` in `database.py`
**File:** `app/infra/database.py`  
**Root cause:** Routes and Celery tasks imported these functions, but they didn't exist in `database.py`. This would have caused `ImportError` at runtime.  
**Status:** Already added by user prior to this session (confirmed present, syntax clean).

---

## V2 End-to-End Stress Test Results

### Test 1 — Celery Broker Reachability
| Check | Result |
|---|---|
| Redis connection | ✅ Connected (redis-13212.c328.europe-west3-1.gce.cloud.redislabs.com) |
| Worker startup | ✅ `celery@MetaData ready.` |
| Task registration | ✅ `ingestion.run_files`, `ingestion.run_crawler`, `ingestion.run_files_v2`, `ingestion.run_crawler_v2` |

---

### Test 2 — V2 File Ingestion + Chat
**Endpoint:** `POST /api/v2/ingest/files`  
**Content:** Synthetic Markdown (Akobot Product Overview)

| Metric | Result |
|---|---|
| HTTP Status | ✅ 200 |
| Nodes Indexed | ✅ **5 nodes** |
| Session ID | `55c80a07-68cb-4350-91c9-c225be35e25e` |

**Chat test using file session_id:**  
`POST /api/v2/chat → {"query": "What are the main features and target users of Akobot?"}`

| Metric | Result |
|---|---|
| HTTP Status | ✅ 200 |
| Answer | "The main features of Akobot include AI agent creation, custom RAG pipelines, and multi-modal AI tools such as image generation and video synthesis (node_id: 0003). The target users of Akobot are prima..." |
| Sources returned | ✅ **5 PageIndex nodes** (real node_ids) |
| Confidence | ✅ **0.98** |
| Active Model | `llama-3.3-70b-versatile` |
| Rate Limited | ✅ False |
| Tools Used | `check_security`, `rewrite_query`, `search_pageindex` |

---

### Test 3 — V2 Crawler (Celery) → Job Lifecycle → Chat
**Endpoint:** `POST /api/v2/ingest/crawler`  
**Target:** `https://akobot.ai/` (max_depth=1)

#### Job Lifecycle
| Phase | Status |
|---|---|
| Job dispatch | ✅ HTTP 200 `status: accepted` |
| Initial poll | ✅ `status: pending` → Celery task picked up |
| After completion | ✅ **`status: completed`** |
| Nodes indexed | ✅ **75 PageIndex nodes** from 1 crawled page |

**Crawler job ID:** `c085c0dd-3bff-4404-9b99-cafeee1d8189`  
**Session ID:** `c085c0dd-3bff-4404-9b99-cafeee1d8189`  

#### Chat using Crawler Session ID
`POST /api/v2/chat → {"query": "What is Akobot? Summarize its main features.", "session_id": "c085c0dd..."}`

| Metric | Result |
|---|---|
| HTTP Status | ✅ 200 |
| Answer | "Akobot is an AI-powered platform that enables the creation of custom AI agents for various use cases. Its main features include automating 80%+ of interactions with AI agents, providing multi-flow int..." |
| Sources | ✅ **5 PageIndex nodes** (real node_ids from crawled site) |
| Confidence | ✅ **0.98** |
| Active Model | `llama-3.3-70b-versatile` |
| Rate Limited | ✅ False (ModelsLab credits restored, Groq quota available) |

---

## System State After Fixes

### What's Working
| Component | Status |
|---|---|
| FastAPI server (port 8000) | ✅ Running, healthy |
| Celery worker (Redis broker) | ✅ Running, ready, consuming tasks |
| V2 file ingestion | ✅ Synchronous, indexes into PageIndex tree cache |
| V2 crawler (Celery async) | ✅ pending → completed, stores 75 nodes |
| V2 chat (file session) | ✅ HTTP 200, confidence 0.98, 5 sources |
| V2 chat (crawler session) | ✅ HTTP 200, confidence 0.98, 5 sources |
| ModelsLab | ✅ `modelslab: true` confirmed in /health |
| Groq (primary LLM) | ✅ `llama-3.3-70b-versatile` active, not rate limited |
| PageIndex tree cache | ✅ In-memory + SQLite persistence (`pageindex_store.py`) |
| Composio integration | ✅ Gated on COMPOSIO_API_KEY, fails gracefully |
| Circuit breaker (per agent::provider) | ✅ Keyed isolation implemented |
| Auth hooks | ✅ `app/middleware/auth_hooks.py` clean integration points |
| SLA gate | ✅ Wired to `/api/v1/metrics/sla` thresholds |

### File Changes Made This Session (by assistant)
| File | Change |
|---|---|
| `app/infra/celery_tasks.py` | Fixed IndentationError on lines 215-216 (Bug 1) |
| `app/infra/celery_tasks.py` | Removed unsupported `session_id` kwarg from `crawl_url` call (Bug 2) |

### File Changes Made by User (tracked for context)
| File | Change |
|---|---|
| `app/v2/retrieval/pageindex_tool.py` | Added `_cache_key()`, tenant_id support, SQLite persistence via `pageindex_store` |
| `app/v2/agents/modular_orchestrator.py` | Passed `tenant_id` to `get_session_node_count` and `search_tree` |
| `app/v2/api/routes_v2.py` | Added `IngestionStatusV2` schema, `upsert_ingestion_job` calls, `/ingest/status/{job_id}` endpoint |
| `app/infra/celery_tasks.py` | Added `run_crawler_job_v2` task with PageIndex tree storage and job status updates |
| `app/infra/database.py` | Added `upsert_ingestion_job` and `get_ingestion_job` functions |
| `app/middleware/auth_hooks.py` | Added auth hook extension points |
| `app/middleware/auth.py` | Wired auth_hooks module |
| `app/infra/circuit_breaker.py` | Per-agent circuit breaker keying (`agent::provider`) |
| `app/infra/llm_client.py` | Context propagation for circuit breaker |
| `app/supervisor/router.py` | Agent context set at entrypoints |
| `app/agents/data_analytics/supervisor.py` | BA agent circuit breaker context |
| `app/agents/support_data_analytics.py` | Support DA circuit breaker context |
| `scripts/py/ba_support_gate.py` | SLA checks wired |

---

## V2 Replacement Readiness

| Component | Readiness |
|---|---|
| V2 Chat + File Ingestion | ✅ **REPLACEMENT READY** |
| V2 Crawler (Celery async) | ✅ **REPLACEMENT READY** (pending→completed confirmed, post-fix) |
| Composio Live Integration | ⚠️ Authentication required (Gmail OAuth) |
| ModelsLab Vision Tools | ✅ Credits restored, `modelslab: true` in /health |

---

## Raw Test Data
- `Stress Test/v2_stress_raw_results.json` — Full E2E stress test Phase 1+2 results
- `Stress Test/v2_crawler_retry_results.json` — Crawler retry (post-fix) confirming completion
- `logs/celery_worker.log` — Celery worker startup log (shows `ready` with task list)


---

---

# V2 Architecture Deep-Dive & Client Handover Reference

> All V2 code lives exclusively inside `app/v2/`. It does not touch or break anything in `app/` (V1). V1 and V2 run simultaneously on the same FastAPI server, mounted at `/api/v1` and `/api/v2` respectively.

---

## 1. Crawler Pipeline — Step by Step

```
User
 └─ POST /api/v2/ingest/crawler   { url, max_depth }
         │
         ├─ [If CELERY_ENABLED=true]
         │       routes_v2.py
         │        ├─ Generates job_id + session_id (UUID4)
         │        ├─ Calls upsert_ingestion_job() → SQLite (status=pending)
         │        ├─ run_crawler_job_v2.delay() → Redis task queue
         │        └─ Returns HTTP 200 { status: accepted, job_id, session_id }
         │
         │       Celery Worker (background process)
         │        ├─ Picks up task from Redis queue
         │        ├─ Instantiates CrawlerService (playwright + asyncio)
         │        ├─ Launches N parallel headless Chromium workers (N = CPU cores, from HardwareProbe)
         │        ├─ Crawls the target URL recursively up to max_depth
         │        │     ├─ Fetches each page (playwright for JS-heavy, aiohttp for static)
         │        │     ├─ Deduplicates pages via URL canonicalization + content hash
         │        │     ├─ Respects robots.txt
         │        │     ├─ Extracts clean text (selectolax HTML parser, removes nav/header/footer)
         │        │     └─ Appends result to self.results_memory (in-memory list)
         │        ├─ Saves full crawl report to disk:
         │        │     data/crawled_docs_v2/{tenant}/{domain}/metadata.json
         │        │     (contains: URL, session_id, pages, content, timestamps, links)
         │        ├─ Converts results_memory into documents [{filename, content}]
         │        ├─ Calls store_documents_in_tree_cache(session_id, docs, tenant_id)
         │        │     ├─ Parses each document with page_index_md.py → Markdown tree
         │        │     ├─ Flattens tree into node list [{node_id, title, text, depth}]
         │        │     ├─ Stores in-memory: _TREE_CACHE["tenant::session_id"] = nodes
         │        │     └─ Persists to SQLite: pageindex_nodes table via upsert_pageindex_nodes()
         │        └─ Calls upsert_ingestion_job() → SQLite (status=completed, nodes_indexed=N)
         │
         └─ [If CELERY_ENABLED=false]
                 runs synchronously in the same FastAPI process
                 (same logic as above but awaited directly — no Redis/Celery)
```

**Where crawled data is saved (after this session's fix):**
| Storage | Location | Persists after restart? |
|---|---|---|
| In-memory tree cache | `_TREE_CACHE["tenant::session_id"]` | ❌ No — rebuilt from SQLite on next query |
| SQLite nodes | `data/tenants/{tenant}/app_data.db` → `pageindex_nodes` table | ✅ Yes |
| JSON crawl report | `data/crawled_docs_v2/{tenant}/{domain}/metadata.json` | ✅ Yes |

**Poll for completion:**
```
GET /api/v2/ingest/status/{job_id}
→ { status: "pending"|"completed"|"failed", payload: { nodes_indexed, message, session_id } }
```

---

## 2. File Upload Pipeline — Step by Step

```
User
 └─ POST /api/v2/ingest/files   (multipart/form-data)
         │
         routes_v2.py
          ├─ Generates session_id (or uses provided one)
          ├─ Calls FileUploadServiceV2.process_files(files)
          │     ├─ .pdf   → PyMuPDF (fitz)  →  text per page
          │     ├─ .docx  → python-docx     →  paragraphs
          │     ├─ .txt / .md → raw UTF-8 read
          │     ├─ .csv   → pandas          →  Markdown table
          │     └─ .xlsx  → openpyxl        →  sheet-by-sheet Markdown table
          │     Returns: [{ filename, content (Markdown string) }]
          │
          ├─ Calls store_documents_in_tree_cache(session_id, docs, tenant_id)
          │     ├─ page_index_md.py parses Markdown into hierarchy:
          │     │     H1 → root node (depth=1)
          │     │     H2 → child node (depth=2) … and so on
          │     │     Paragraph text → appended to most recent heading node
          │     ├─ _flatten_tree() collapses tree to flat node list
          │     ├─ Stored in _TREE_CACHE["tenant::session_id"]
          │     └─ Persisted in SQLite pageindex_nodes table
          └─ Returns HTTP 200 { status: completed, session_id, nodes_indexed }
```

**Supported file types:** PDF, DOCX, TXT, MD, CSV, XLSX  
**Note:** File uploads are synchronous — response returns immediately when indexing is done. No polling needed.

---

## 3. Complete V2 RAG Flow — Phase by Phase

```
User → POST /api/v2/chat  { query, session_id, model_provider }

[Rate Limiter] TokenBucketRateLimiter (per IP/tenant) → HTTP 429 if exceeded

ModularOrchestrator Agent Loop (max 6 turns, Groq LLM decides which tools to call):

  Turn 1 — Security Gate (ALWAYS first)
    LLM calls: check_security({ prompt: query })
    → PromptInjectionGuard.evaluate(query) using Groq Llama Guard
    → Returns { is_malicious, action: "allow"|"block" }
    → If block → STOP, return polite refusal (no other tools called)

  Turn 2 — Query Rewriting (for complex/ambiguous queries)
    LLM calls: rewrite_query({ query })
    → PromptRewriter rewrites for better semantic coverage
    → Model: llama-3.1-8b-instant

  Turn 3 — Knowledge Retrieval
    LLM calls: search_pageindex({ query, top_k: 5 })
    → Checks _TREE_CACHE["tenant::session_id"] (in-memory)
    → If cache miss, loads from SQLite pageindex_nodes table
    → _title_rank() scores nodes: title match = 3× weight, body = 1×
    → Returns top-k nodes with { node_id, title, text, score }

  Turn 4 (optional) — External Actions (only if user asks for email/CRM/etc)
    LLM calls: use_composio_integration({ action, payload })
    → ComposioToolRouterClient routes to Gmail/HubSpot/Calendar
    → Returns { status, result } or { status: unavailable } if no API key

  Final Turn — Answer Synthesis
    LLM receives all tool results, writes final text answer
    Cites node_ids when referencing PageIndex sources

Response:
{
  session_id, answer,
  sources: [{ node_id, title, text }],
  confidence: 0.98 (sources found) | 0.75 (no sources) | 0.0 (blocked),
  optimizations: { active_model, tools_used, blocked_by_guard, rate_limited }
}
```

### Models Used Per Phase

| Phase | Model | Provider |
|---|---|---|
| Security Guard | `llama-guard-3-8b` | Groq |
| Query Rewriting | `llama-3.1-8b-instant` | Groq |
| Orchestrator (primary) | `llama-3.3-70b-versatile` | Groq |
| Orchestrator fallback 1 | `llama-3.1-8b-instant` | Groq (on rate-limit) |
| Orchestrator fallback 2 | `gemma2-9b-it` | Groq (on rate-limit) |
| Orchestrator fallback 3 | `mixtral-8x7b-32768` | Groq (on rate-limit) |
| Composio actions | No LLM — direct REST call | Composio |

---

## 4. Concurrency & Multi-User Security

### How Concurrent Requests Work
- **FastAPI is fully async** — concurrent requests never block each other (each gets its own asyncio coroutine)
- **Celery workers** handle background crawl jobs — each job is isolated, no shared state between jobs
- **Rate limiter** (`TokenBucketRateLimiter`) prevents any single tenant from flooding the system

### Data Isolation — No Cross-Tenant Leakage

| Layer | Isolation Mechanism |
|---|---|
| PageIndex tree (in-memory) | `_TREE_CACHE["{tenant_id}::{session_id}"]` — compound key |
| SQLite app_data.db | Each tenant: `data/tenants/{tenant_id}/app_data.db` |
| SQLite crawler_data.db | Each tenant: `data/tenants/{tenant_id}/crawler_data.db` |
| Crawled JSON on disk | `data/crawled_docs_v2/{tenant_id}/{domain}/` |
| ChromaDB (V1 vector store) | Isolated collection per tenant: `collection_{tenant_id}` |
| Circuit breakers | Keyed per `{agent}::{provider}` — tenant failures don't cascade |

### Security Features

| Feature | Where |
|---|---|
| Prompt Injection Guard | Every V2 chat request, before any processing |
| Rate limiting | `TokenBucketRateLimiter`, per tenant/IP |
| Tenant header isolation | All endpoints read `x-tenant-id`, scope all DB/cache ops |
| Distributed lock | `app/infra/locks.py` — prevents concurrent write races |
| Auth hooks | `app/middleware/auth_hooks.py` — implement JWT/OIDC here |
| Circuit breaker | Per-agent-per-provider, prevents cascade failures |
| CORS | Configured in `app/main.py` — restrict origins before production |

---

## 5. Viewing the PageIndex Tree / Node Structure

There is no built-in UI for this, but here are three ways to inspect it:

### Option A — Query the Status API
```
GET /api/v2/ingest/status/{job_id}
→ Returns nodes_indexed count
```

### Option B — Direct SQLite Query
```bash
sqlite3 data/tenants/global/app_data.db
.headers on
.mode column
SELECT node_id, title, depth, substr(text,1,80) FROM pageindex_nodes WHERE session_id='YOUR_SESSION_ID';
```
Each row is a node: `node_id`, heading `title`, `depth` (0=root, 1=H1, 2=H2…), and text snippet.

### Option C — Python Debug Script
```python
from app.v2.retrieval.pageindex_store import fetch_pageindex_nodes
nodes = fetch_pageindex_nodes("YOUR_SESSION_ID", tenant_id="global")
for n in nodes:
    print("  " * n["depth"] + f"[{n['node_id']}] {n['title']}")
```

### Option D — Open the Crawl JSON
After a crawl: `data/crawled_docs_v2/{tenant}/{domain}/metadata.json` — full page content in human-readable JSON.

> **For the future:** A `GET /api/v2/pageindex/tree/{session_id}` endpoint can expose the nested node tree as JSON. It's a small addition if clients need it.

---

## 6. Langfuse Observability

| Pipeline | Langfuse Status |
|---|---|
| V1 RAG agent (`app/agents/rag.py`) | ✅ Implemented — set env vars to activate |
| V1 Business Analyst (`app/agents/data_analytics/supervisor.py`) | ✅ Implemented |
| **V2 ModularOrchestrator** | ❌ **Not yet implemented** |

**To activate Langfuse for V1:** Add to `.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.langfuse.com
```

**For V2:** Adding Langfuse is a ~10-line change using `langfuse.openai` auto-instrumentation, wrapping the Groq client in `modular_orchestrator.py`.

---

## 7. Client Integration — API Reference

### Base URL
```
http://{your-server}:8000
```

### Authentication
Pass `x-tenant-id` on every request to scope data to your tenant:
```
x-tenant-id: my-client-name
```
For production: implement `resolve_client_token()` in `app/middleware/auth_hooks.py` to validate JWT/API keys.

---

### V2 Endpoints — Complete Reference

#### File Ingest
```http
POST /api/v2/ingest/files
Content-Type: multipart/form-data
x-tenant-id: my-tenant

# Response
{ "status": "completed", "job_id": "uuid", "session_id": "uuid", "nodes_indexed": 42 }
```

#### Web Crawler (async via Celery)
```http
POST /api/v2/ingest/crawler
Content-Type: application/json
x-tenant-id: my-tenant

{ "url": "https://docs.example.com", "max_depth": 2 }

# Response
{ "status": "accepted", "job_id": "uuid", "session_id": "uuid" }
```

#### Poll Crawler Job
```http
GET /api/v2/ingest/status/{job_id}
x-tenant-id: my-tenant

# Response
{ "status": "pending|completed|failed", "payload": { "nodes_indexed": 75 } }
```

#### Chat with Indexed Content
```http
POST /api/v2/chat
Content-Type: application/json
x-tenant-id: my-tenant

{ "query": "What is the refund policy?", "session_id": "uuid-from-ingest", "model_provider": "groq" }

# Response
{
  "session_id": "...",
  "answer": "According to the documentation...",
  "sources": [{ "node_id": "0003", "title": "Refund Policy", "text": "..." }],
  "confidence": 0.98,
  "optimizations": { "tools_used": ["check_security", "rewrite_query", "search_pageindex"], "active_model": "llama-3.3-70b-versatile" }
}
```

---

### V1 Endpoints (Unchanged)

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/chat` | Full multi-phase RAG pipeline (ChromaDB vector search + reranker) |
| `POST /api/v1/ingest/files` | V1 file ingestion to ChromaDB |
| `POST /api/v1/ingest/crawler` | V1 crawler (SQLite + ChromaDB) |
| `POST /api/v1/business-analyst` | XGBoost + LLM business analytics |
| `GET /api/v1/metrics/sla` | Live SLA metrics (error rate, p95 latency) |
| `GET /health` | System health check |
| `GET /docs` | Swagger UI — interactive API explorer |

---

## 8. Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ V2 primary | Main LLM provider for V2 orchestrator |
| `CELERY_ENABLED` | Recommended | `true` = crawl jobs run async via Redis |
| `CELERY_BROKER_URL` | If Celery | Redis connection string |
| `CELERY_AUTOSTART` | Optional | `true` = `start_stack.ps1` auto-starts worker |
| `COMPOSIO_API_KEY` | Optional | Enables email/CRM via Composio |
| `LANGFUSE_PUBLIC_KEY` | Optional | Langfuse tracing for V1 agents |
| `LANGFUSE_SECRET_KEY` | Optional | — |
| `OPENAI_API_KEY` | Optional | Fallback LLM if no Groq key |
| `GUARD_FAIL_OPEN` | Optional | `true`=allow on guard error; `false`=block (safer) |
| `CELERY_POOL` | Optional | `solo` (Windows), `prefork` (Linux/prod) |

---

## 9. Known Limitations & Pre-Handover Checklist

| Item | Status | Action |
|---|---|---|
| Langfuse for V2 | ❌ Not wired | Add `langfuse.openai` wrapper in `modular_orchestrator.py` |
| Composio Gmail | ⚠️ Auth required | Client completes OAuth via connect link in response |
| PageIndex Tree UI | ❌ No endpoint | Add `GET /api/v2/pageindex/tree/{session_id}` if needed |
| In-memory cache | ⚠️ Lost on restart | Auto-rebuilt from SQLite on next query — no data loss |
| JWT Auth | ⚠️ Hook ready | Implement `resolve_client_token()` in `auth_hooks.py` |
| Production CORS | ⚠️ Wide open | Set `allow_origins` in `app/main.py` before go-live |
| Groq Free Tier | ⚠️ 100k TPD | Upgrade to Dev Tier; fallback chain handles rate limits |
| Crawler disk persistence | ✅ Fixed | `data/crawled_docs_v2/{tenant}/{domain}/metadata.json` |
