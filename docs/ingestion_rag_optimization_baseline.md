# Baseline Results (Phase 0)

Timestamp: 2026-03-04

## Crawler Baseline (3 runs)
Target: `https://learnnect.com` (depth 4 via `benchmark_crawler.py`)

Runs:
- Run 1: 26.82s, pages 38, status success
- Run 2: 25.18s, pages 38, status success
- Run 3: 27.30s, pages 38, status success

Median: **26.82s**

Recorded in: `crawler_benchmark_results/benchmark_log.md`

## Ingestion Baseline (3 runs)
Files:
- `test-files/Updated_Resume_DS.pdf`
- `test-files/support agent.docx`

Tenant: `baseline-benchmark`  
Reset DB each run: `true`

Runs:
- Run 1: 32.35s, chunks 6
- Run 2: 43.43s, chunks 6
- Run 3: 26.23s, chunks 6

Median: **32.35s**

Notes:
- Embedding model (`BAAI/bge-large-en-v1.5`) loaded during runs (warm‑up visible in logs).

## RAG Baseline (3 queries)
Queries:
- “Summarize Updated_Resume_DS.pdf”
- “List key skills from the uploaded resumes”
- “Provide a brief summary of support agent.docx”

Result:
- **All 3 requests returned HTTP 500** (Internal Server Error)
- Occurred both with and without `x-tenant-id: baseline-benchmark`
- No latency metrics could be captured from responses

Action required:
- Inspect server error logs to identify the 500 cause before proceeding with RAG latency tuning.

## RAG Diagnosis Update (2026-03-04)
- Fixed `UnboundLocalError` in `app/api/routes.py` by initializing `chat_history` for JSON requests.
- API startup now loads `.env` in `scripts/start_stack.ps1` and `scripts/start_api.ps1`.
- Verified JSON `/api/v1/chat` returns **200** for a baseline request (`session_id=baseline-rag-001`).
- Added tolerant parsing for `chat_history` when provided as a JSON string in JSON payloads (prevents 422 in Swagger-style inputs).
- Verified `/api/v1/chat` returns **200** when `chat_history` is supplied as a JSON string or as a list.

## RAG Diagnosis Findings (2026-03-04)
- `DATA_NOT_FOUND` is caused by **zero retrieval chunks**.
- Root cause: `QDRANT_MULTI_TENANT=true` + `x-tenant-id=aditya-ds` results in a tenant filter that matches **no vectors**.
- `/api/v1/ingest/status`:
  - With `x-tenant-id=aditya-ds`: `total_vectors=0`
  - Without tenant header: `enterprise_rag` has `total_vectors=6` (`Updated_Resume_DS.pdf`, `support agent.docx`)
- Action: ingest documents with the desired `x-tenant-id`, or omit tenant header when querying the global `enterprise_rag` collection.

## RAG Diagnosis Findings (Routing Guard) (2026-03-04)
- Query `summarize the resumes in the knowledge base` routed to **out_of_scope_bypass**.
- This happens when the intent classifier returns `out_of_scope`, which short-circuits the RAG pipeline before retrieval.
- Result: generic “unable to answer” response with empty sources even though `enterprise_rag` contains 6 vectors.

### Intent Classifier Data Quality Issue
- The few-shot file `app/prompt_engine/groq_prompts/few_shots/intent_classifier_llama_3.1_8b_instant.json`
  contains **misaligned examples** (e.g., “top 5 movies of 2022” labeled as `rag_question`).
- This skews the classifier toward `out_of_scope` for legitimate enterprise KB queries.
- Fix direction (for later): replace few-shots with enterprise KB-style questions (docs, resumes, policies, SOPs).

## RAG Diagnosis Update (Heuristic Override) (2026-03-04)
- Added a safe router override: if intent returns `out_of_scope` but the query contains enterprise KB keywords
  (e.g., `resume`, `policy`, `document`, `kb`), route to `rag_agent`.
- Result: query `summarize the resumes in the knowledge base` now routes to RAG and returns grounded content.

## RAG Diagnosis Update (No Hardcoded Keywords) (2026-03-04)
- Removed the hardcoded enterprise keyword override in the router.
- Added **LLM-based source scope classification** (`kb_only` | `session_only` | `both`) to decide how retrieval is performed.

## RAG Diagnosis Update (Source Scope Validation) (2026-03-04)
Test:
- Query: `Use only the attached file. Summarize it.`
- Upload: `test-files/Updated_Resume_DS.pdf`
- Expected: `session_only`

Result:
- `retrieval_scope=session_only`
- Sources from the uploaded file only

## RAG Diagnosis Update (Synthesis Fallback) (2026-03-04)
- Added an extractive fallback when the LLM response fails to parse or the API errors.
- This prevents `Internal Generation Error validating tokens.` from surfacing to users.

## RAG Diagnosis Update (Few-shot Repair) (2026-03-04)
- Corrected a mislabelled few-shot: “top 5 movies of 2022” now labeled `out_of_scope`.
- Replaced overly specific KB examples with **general** enterprise KB prompts:
  - `Summarize the uploaded document.`
  - `Summarize the documents in the knowledge base.`
  - `List the key points from the uploaded files.`
  - `Provide a brief summary of the attached document.`
  - `Summarize the internal policy document.`
- Result: `summarize the resumes in the knowledge base` continues to route to RAG and returns grounded content.

## RAG Diagnosis Findings (File Guard) (2026-03-04)
- When `ENFORCE_FILE_SESSION=true`, any query containing file keywords (e.g., `pdf`, `xlsx`, `attached`) returns:
  `400: No uploaded file detected for this session.`
- This is expected behavior for file-specific queries when no session collection is attached.

## RAG Diagnosis Update (File Guard Removed) (2026-03-04)
- Removed the hard file-keyword guard so **normal RAG works even when no session files are attached**.
- Result: queries like `summarize the uploaded document` now return a graceful response instead of `400`.

## Multimodal RAG Validation (2026-03-04)
Test:
- Query: `Check the attached resume and recommend the best Learnnect courses for this person.`
- Upload: `test-files/Updated_Resume_DS.pdf`

Result:
- **200 OK**
- Response uses **both** session resume content and Learnnect KB to recommend courses.

## RAG Latency Snapshot (No File Keywords) (2026-03-04)
Query: `summarize the knowledge base content`
- Status: **200**
- Retrieval: ~**0.89s**
- Rerank: ~**0.18s**
- LLM time: ~**7.29s**
- Total wall time: ~**13.9s**
- Agent: `rag_agent`, Model: `llama-3.3-70b-versatile`

Primary latency driver is **LLM generation** (not retrieval).

## Active Config Snapshot (from `.env`)
- `CRAWLER_FAST_HTTP=true`
- `CRAWLER_HTTP_MIN_CHARS=200`
- `CRAWLER_WAIT_SELECTORS=.content,.main,.article-body`
- `CRAWLER_SCROLL=true`
- `CRAWLER_SCROLL_STEPS=2`
- `CRAWLER_SCROLL_PAUSE_MS=200`
- `CRAWLER_DOMAIN_CONCURRENCY=10`
- `ENFORCE_FILE_SESSION=true`

## Phase 3: Synthesis + Routing Optimization (2026-03-04)
### Changes Implemented
- Added **session cache** (1-week TTL) with auto-refresh and cache-hit tracking.
- Added **cache notice + expiry** metadata for client transparency.
- Token-aware **chat history trimming** based on input budget.
- **Guard escalation**: if prompt guard flags malicious, re-check with 70B to reduce false positives.
- Removed `response_format` requirement in synthesis to reduce Groq API failures under load.
- Replaced intent few-shots with **general enterprise-ready examples** (no domain-specific bias).
- Telemetry logging is now **fail-safe** (will not crash `/chat` on log errors).

### Phase 3 Test Results
1) **Normal RAG**  
Query: `What is Learnnect and what do they provide? Be descriptive.`  
Session: `phase3-normal-004`  
Result: **200 OK**  
- `retrieval_scope=kb_only`  
- `cache_hit=false`  
- `cache_notice` present  
- LLM response returned (no fallback)

2) **Cache Reuse**  
Same query + same session: `phase3-normal-004`  
Result: **200 OK**  
- `cache_hit=true`  
- Retrieval time dropped to ~**28ms**

3) **Multi-modal RAG (file + KB)**  
Query: `Check the attached resume and recommend best Learnnect courses for this person.`  
File: `test-files/Updated_Resume_DS.pdf`  
Session: `phase3-mm-006`  
Result: **200 OK**  
- `retrieval_scope=both`  
- Sources include **uploaded resume**  
- LLM response returned (no 500)

## Phase 4: Ingestion Throughput + Indexing Efficiency (2026-03-04)
### Changes Implemented
- **Parallelized file load + chunking** using ThreadPoolExecutor.
- **Pre-initialized embedding model** before parallel CPU workers to prevent `meta` tensor failures.

### Phase 4 Test Results (Isolation)
Ingestion files:
- `test-files/Updated_Resume_DS.pdf`
- `test-files/support agent.docx`

Tenant: `phase4-benchmark`  
Reset DB: `true`  
Result: **6 chunks in 22.77s**

Notes:
- CPU embedding path used in local test.
- No crashes after embedding model init change.

## Phase 5: Reverted (2026-03-04)
Phase 5 changes were rolled back to the Phase 4 baseline per request.

Post-rollback fixes retained:
- **Tokenizer caching in chunker** to eliminate repeated HuggingFace downloads during ingestion.
- **Out-of-scope override** when tenant/session context exists to prevent KB bypass.

Explicit removals (per request):
- Removed `/metrics` endpoint and Prometheus instrumentation.
- Removed dynamic retrieval/rerank logic.
- Removed token‑budget context trimming.

## Phase 5: Reintroduced (Implementation Plan + In Progress)
Now re‑adding:
- `/metrics` endpoint and Prometheus instrumentation.
- Dynamic retrieval + rerank depth logic.
- Token‑budget context trimming.

## Planned Next Phase: Client Ops + Deployment (Discuss/Decide)
Scope (from `docs/client_ops_research.md`):
- **Point 2**: Provide public API base URL + docs URL + endpoint list, and deployment expectations.
- **Point 3**: Concurrency behavior under load (100+ users), isolation risks, and hardening plan.

Status:
- Not implemented yet. This will be discussed and scheduled as a separate phase.
